import logging
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
                    result = self._runner.run(
                        cmd, total_duration=total_dur,
                        progress_callback=lambda p: self.progress.emit(
                            (gi + p) / total_groups
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
                                self.finished.emit(False)
                                return
                        else:
                            self.error.emit(f"Encoding failed: {result.stderr[-300:]}")
                            self.finished.emit(False)
                            return

                    intermediate_files.append(out_path)

            # Concatenate intermediates if needed
            if len(intermediate_files) == 1:
                import shutil
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
                        "-c:v", self._settings.video_codec,
                        "-crf" if self._settings.video_codec == "libx264" else "-cq",
                        str(self._settings.video_crf),
                        "-c:a", "aac", "-b:a", "192k",
                    ],
                )
                result = self._runner.run(cmd)
                if result.returncode != 0:
                    self.error.emit(f"Concat failed: {result.stderr[-300:]}")
                    self.finished.emit(False)
                    return

            # Add separate audio if needed
            if self._use_separate_audio and self._audio_path:
                self.stage.emit("Adding audio track...")
                final_with_audio = self._settings.output_path
                temp_video = str(Path(self._temp_dir) / "video_only.mp4")
                import shutil
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
                    self.finished.emit(False)
                    return

            self.stage.emit("Export complete!")
            self.finished.emit(True)

        except Exception as e:
            logger.exception("Export failed")
            self.error.emit(str(e))
            self.finished.emit(False)
