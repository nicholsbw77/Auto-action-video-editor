import logging
import os
import shutil
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from core.audio_analyzer import AudioAnalyzer
from core.models import AnalysisResult, CutPoint, ExportSettings
from ffmpeg.filter_graph import FilterGraphBuilder
from ffmpeg.runner import FFmpegRunner

logger = logging.getLogger("autoeditor.workers")


class AnalysisWorker(QThread):
    progress = pyqtSignal(str)  # status message
    finished = pyqtSignal(object)  # AnalysisResult or None
    error = pyqtSignal(str)

    def __init__(self, audio_path: str, parent=None):
        super().__init__(parent)
        self._audio_path = audio_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit("Analyzing audio...")
            analyzer = AudioAnalyzer()
            result = analyzer.analyze(
                self._audio_path,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                self.finished.emit(None)
                return
            self.progress.emit("Analysis complete")
            self.finished.emit(result)
        except Exception as e:
            logger.exception("Analysis failed")
            self.error.emit(str(e))


class ExportWorker(QThread):
    progress = pyqtSignal(float)  # 0.0 to 1.0
    stage = pyqtSignal(str)  # stage description
    finished = pyqtSignal(bool)  # success
    error = pyqtSignal(str)

    def __init__(
        self,
        cuts: list[CutPoint],
        ffmpeg_runner: FFmpegRunner,
        filter_builder: FilterGraphBuilder,
        settings: ExportSettings,
        source_paths: list[str],
        audio_path: str | None,
        temp_dir: str,
        use_separate_audio: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._cuts = cuts
        self._runner = ffmpeg_runner
        self._filter_builder = filter_builder
        self._settings = settings
        self._source_paths = source_paths
        self._audio_path = audio_path
        self._temp_dir = temp_dir
        self._use_separate_audio = use_separate_audio
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.stage.emit("Grouping segments...")
            groups = self._filter_builder.group_cuts(self._cuts)
            intermediate_files = []

            total_groups = len(groups)
            for gi, group in enumerate(groups):
                if self._cancelled:
                    self.finished.emit(False)
                    return

                self.stage.emit(f"Encoding group {gi + 1}/{total_groups}...")
                batches = self._filter_builder.split_batches(group)

                for bi, batch in enumerate(batches):
                    if self._cancelled:
                        self.finished.emit(False)
                        return

                    source_mapping = self._filter_builder.build_source_mapping(batch)
                    video_graph = self._filter_builder.build_xfade_graph(batch, source_mapping)
                    audio_graph = self._filter_builder.build_audio_graph(
                        batch, source_mapping, self._use_separate_audio,
                    )

                    # Build combined filter
                    combined = video_graph
                    maps = ["[vout]"]
                    if audio_graph:
                        combined += ";\n" + audio_graph
                        maps.append("[aout]")

                    # Determine input files for this batch
                    unique_sources = sorted(set(c.source_index for c in batch))
                    inputs = [self._source_paths[s] for s in unique_sources]

                    out_path = str(Path(self._temp_dir) / f"batch_{gi}_{bi}.mp4")

                    extra_args = [
                        "-c:v", self._settings.video_codec,
                        "-pix_fmt", "yuv420p",
                    ]
                    if self._settings.video_codec == "libx264":
                        extra_args += ["-crf", str(self._settings.video_crf), "-preset", "medium"]
                    else:
                        extra_args += ["-cq", str(self._settings.video_crf), "-preset", "p4"]

                    extra_args += ["-c:a", "aac", "-b:a", "192k"]

                    cmd = self._runner.build_command(
                        inputs=inputs, output=out_path,
                        filter_complex=combined, maps=maps,
                        extra_args=extra_args,
                    )

                    total_dur = sum(c.end - c.start for c in batch)
                    group_index = gi  # capture for lambda closure
                    result = self._runner.run(
                        cmd, total_duration=total_dur,
                        progress_callback=lambda p, g=group_index: self.progress.emit(
                            (g + p) / total_groups
                        ),
                        cancel_check=lambda: self._cancelled,
                    )

                    if result.returncode != 0:
                        # Try CPU fallback if GPU failed
                        if self._settings.video_codec == "h264_nvenc":
                            self.stage.emit("GPU failed, retrying with CPU...")
                            extra_args_cpu = [
                                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                                "-pix_fmt", "yuv420p",
                                "-c:a", "aac", "-b:a", "192k",
                            ]
                            cmd = self._runner.build_command(
                                inputs=inputs, output=out_path,
                                filter_complex=combined, maps=maps,
                                extra_args=extra_args_cpu,
                            )
                            result = self._runner.run(cmd, total_duration=total_dur)
                            if result.returncode != 0:
                                self.error.emit(f"Encoding failed: {result.stderr[-300:]}")
                                return
                        else:
                            self.error.emit(f"Encoding failed: {result.stderr[-300:]}")
                            return

                    intermediate_files.append(out_path)

            # Concatenate intermediates if needed
            if len(intermediate_files) == 1:
                shutil.move(intermediate_files[0], self._settings.output_path)
            else:
                self.stage.emit("Concatenating segments...")
                concat_file = str(Path(self._temp_dir) / "concat.txt")
                with open(concat_file, "w") as f:
                    f.write(self._filter_builder.build_concat_file(intermediate_files))

                cmd = self._runner.build_command(
                    inputs=[],
                    output=self._settings.output_path,
                    extra_args=[
                        "-f", "concat", "-safe", "0", "-i", concat_file,
                        "-c", "copy",
                    ],
                )
                result = self._runner.run(cmd)
                if result.returncode != 0:
                    self.error.emit(f"Concat failed: {result.stderr[-300:]}")
                    return

            # Add separate audio if needed
            if self._use_separate_audio and self._audio_path:
                self.stage.emit("Adding audio track...")
                final_with_audio = self._settings.output_path
                temp_video = str(Path(self._temp_dir) / "video_only.mp4")
                shutil.move(final_with_audio, temp_video)

                cmd = self._runner.build_command(
                    inputs=[temp_video, self._audio_path],
                    output=final_with_audio,
                    extra_args=[
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        "-map", "0:v", "-map", "1:a",
                        "-shortest",
                    ],
                )
                result = self._runner.run(cmd)
                if result.returncode != 0:
                    self.error.emit(f"Audio mux failed: {result.stderr[-300:]}")
                    return

            self.stage.emit("Export complete!")
            self.finished.emit(True)

        except Exception as e:
            logger.exception("Export failed")
            self.error.emit(str(e))


class AudioExtractWorker(QThread):
    """Extract audio from video on a background thread (Critical #2 fix)."""
    finished = pyqtSignal(str)  # extracted audio path
    error = pyqtSignal(str)

    def __init__(self, video_path: str, output_path: str, runner: FFmpegRunner, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._output_path = output_path
        self._runner = runner

    def run(self):
        try:
            cmd = self._runner.build_command(
                inputs=[self._video_path], output=self._output_path,
                extra_args=["-vn", "-ac", "1", "-ar", "22050"],
            )
            result = self._runner.run(cmd)
            if result.returncode != 0:
                self.error.emit(f"Audio extraction failed: {result.stderr[-200:]}")
                return
            self.finished.emit(self._output_path)
        except Exception as e:
            logger.exception("Audio extraction failed")
            self.error.emit(str(e))


class TrimWorker(QThread):
    """Run trim export on a background thread (Critical #1 fix)."""
    progress = pyqtSignal(int)  # 0-100
    finished = pyqtSignal(bool)  # success
    error = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        regions: list,
        output_dir: str,
        is_concat: bool,
        reencode: bool,
        source_name: str,
        runner: FFmpegRunner,
        session_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._regions = regions
        self._output_dir = output_dir
        self._is_concat = is_concat
        self._reencode = reencode
        self._source_name = source_name
        self._runner = runner
        self._session_dir = session_dir
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            segment_files = []
            for i, region in enumerate(self._regions):
                if self._cancelled:
                    self.finished.emit(False)
                    return

                out_name = f"{self._source_name}_trim_{i + 1}.mp4"
                if self._is_concat:
                    out_path = os.path.join(self._session_dir, out_name)
                else:
                    out_path = os.path.join(self._output_dir, out_name)

                if self._reencode:
                    extra = ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
                             "-c:a", "aac", "-b:a", "192k"]
                else:
                    extra = ["-c", "copy"]

                cmd = self._runner.build_command(
                    inputs=[self._video_path], output=out_path,
                    extra_args=["-ss", str(region.start), "-to", str(region.end)] + extra,
                )
                result = self._runner.run(cmd)
                if result.returncode != 0:
                    self.error.emit(f"Failed to export {region.label}")
                    return

                segment_files.append(out_path)
                self.progress.emit(int((i + 1) / len(self._regions) * (100 if not self._is_concat else 80)))

            if self._is_concat and len(segment_files) > 1:
                concat_file = os.path.join(self._session_dir, "concat.txt")
                with open(concat_file, "w") as f:
                    for sf in segment_files:
                        f.write(f"file '{sf}'\n")

                final_output = os.path.join(self._output_dir, f"{self._source_name}_trimmed.mp4")
                cmd = self._runner.build_command(
                    inputs=[], output=final_output,
                    extra_args=["-f", "concat", "-safe", "0", "-i", concat_file,
                                "-c:v", "libx264", "-crf", "18",
                                "-c:a", "aac", "-b:a", "192k"],
                )
                result = self._runner.run(cmd)
                if result.returncode != 0:
                    self.error.emit("Failed to concatenate regions")
                    return
            elif self._is_concat and len(segment_files) == 1:
                # Single region in concat mode: move from temp to output
                final_output = os.path.join(self._output_dir, f"{self._source_name}_trimmed.mp4")
                shutil.move(segment_files[0], final_output)

            self.progress.emit(100)
            self.finished.emit(True)

        except Exception as e:
            logger.exception("Trim export failed")
            self.error.emit(str(e))
