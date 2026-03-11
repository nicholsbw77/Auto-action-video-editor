import os
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QProgressBar, QTextEdit, QLineEdit, QGroupBox, QMessageBox,
)
from PyQt6.QtCore import Qt
from gui.widgets import FileListWidget

logger = logging.getLogger("autoeditor.gui")


class AutoEditTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_path: str | None = None
        self._analysis_worker = None
        self._export_worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        # Video files
        video_header = QHBoxLayout()
        video_header.addWidget(QLabel("Video Files:"))
        self._add_video_btn = QPushButton("Add Video(s)")
        self._add_video_btn.clicked.connect(self._add_videos)
        video_header.addWidget(self._add_video_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_videos)
        video_header.addWidget(self._clear_btn)
        video_header.addStretch()
        input_layout.addLayout(video_header)

        self._file_list = FileListWidget()
        self._file_list.setMaximumHeight(120)
        input_layout.addWidget(self._file_list)

        # Mode toggle
        mode_layout = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._single_radio = QRadioButton("Single Video")
        self._multi_radio = QRadioButton("Multi-Clip Assembly")
        self._single_radio.setChecked(True)
        self._mode_group.addButton(self._single_radio)
        self._mode_group.addButton(self._multi_radio)
        self._single_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._single_radio)
        mode_layout.addWidget(self._multi_radio)
        mode_layout.addStretch()
        input_layout.addLayout(mode_layout)

        # Audio source
        audio_layout = QHBoxLayout()
        self._load_audio_btn = QPushButton("Load Audio")
        self._load_audio_btn.clicked.connect(self._load_audio)
        audio_layout.addWidget(self._load_audio_btn)
        self._audio_label = QLabel("No audio loaded")
        audio_layout.addWidget(self._audio_label)
        self._use_video_audio = QCheckBox("Use video's own audio")
        self._use_video_audio.setToolTip("Use the audio track from the video file for analysis")
        audio_layout.addWidget(self._use_video_audio)
        audio_layout.addStretch()
        input_layout.addLayout(audio_layout)

        layout.addWidget(input_group)

        # Export settings
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Folder:"))
        self._output_folder = QLineEdit()
        self._output_folder.setReadOnly(True)
        out_layout.addWidget(self._output_folder)
        self._browse_output = QPushButton("Browse...")
        self._browse_output.clicked.connect(self._browse_output_folder)
        out_layout.addWidget(self._browse_output)
        export_layout.addLayout(out_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Filename:"))
        self._output_name = QLineEdit("output_edited.mp4")
        name_layout.addWidget(self._output_name)
        export_layout.addLayout(name_layout)

        self._info_label = QLabel("Resolution: -- | FPS: -- | Codec: --")
        export_layout.addWidget(self._info_label)

        layout.addWidget(export_group)

        # Go button and progress
        action_layout = QHBoxLayout()
        self._go_btn = QPushButton("Go")
        self._go_btn.setEnabled(False)
        self._go_btn.setMinimumHeight(40)
        self._go_btn.setToolTip("Add at least one video to start")
        self._go_btn.clicked.connect(self._start_pipeline)
        action_layout.addWidget(self._go_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_pipeline)
        action_layout.addWidget(self._cancel_btn)
        layout.addLayout(action_layout)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._stage_label = QLabel("")
        layout.addWidget(self._stage_label)

        # Log area
        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMaximumHeight(150)
        layout.addWidget(self._log_area)

    def set_dependencies(self, config, ffmpeg_runner, filter_builder, gpu_detector, temp_manager):
        """Called by MainWindow after construction to inject dependencies."""
        self._config = config
        self._runner = ffmpeg_runner
        self._filter_builder = filter_builder
        self._gpu_detector = gpu_detector
        self._temp_manager = temp_manager

    def _add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.ts);;All Files (*)",
        )
        for f in files:
            self._file_list.add_file(f)
        self._update_go_state()

    def _clear_videos(self):
        self._file_list.clear_all()
        self._update_go_state()

    def _load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.mp3 *.wav *.flac *.aac *.ogg);;All Files (*)",
        )
        if path:
            self._audio_path = path
            self._audio_label.setText(os.path.basename(path))
            self._update_go_state()

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_folder.setText(folder)

    def _on_mode_changed(self):
        is_multi = self._multi_radio.isChecked()
        self._use_video_audio.setEnabled(not is_multi)
        if is_multi:
            self._use_video_audio.setChecked(False)
            self._use_video_audio.setToolTip("Not available in multi-clip mode")
        else:
            self._use_video_audio.setToolTip("Use the audio track from the video file")
        self._update_go_state()

    def _update_go_state(self):
        has_videos = len(self._file_list.get_paths()) > 0
        is_multi = self._multi_radio.isChecked()
        has_audio = self._audio_path is not None
        use_own = self._use_video_audio.isChecked()

        if is_multi:
            ready = has_videos and len(self._file_list.get_paths()) >= 2 and has_audio
            if not ready:
                self._go_btn.setToolTip("Need 2+ videos and an audio file for multi-clip mode")
        else:
            ready = has_videos and (has_audio or use_own)
            if not ready:
                self._go_btn.setToolTip("Add a video and load audio (or check 'Use video's own audio')")

        self._go_btn.setEnabled(ready)

    def log(self, message: str):
        self._log_area.append(message)

    def set_progress(self, value: float):
        self._progress.setValue(int(value * 100))

    def set_stage(self, text: str):
        self._stage_label.setText(text)

    def _start_pipeline(self):
        paths = self._file_list.get_paths()
        if not paths:
            return

        self._go_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)

        # Always create a temp session
        self._temp_manager.create_session()

        # Update default filename from first video
        first_name = os.path.splitext(os.path.basename(paths[0]))[0]
        if self._output_name.text() == "output_edited.mp4":
            self._output_name.setText(f"{first_name}_edited.mp4")

        use_own_audio = self._use_video_audio.isChecked()

        if use_own_audio:
            audio_path = self._extract_audio(paths[0])
        else:
            audio_path = self._audio_path

        if not audio_path:
            self.log("Error: No audio source available")
            self._go_btn.setEnabled(True)
            self._cancel_btn.setEnabled(False)
            return

        from gui.workers import AnalysisWorker
        self._analysis_worker = AnalysisWorker(audio_path)
        self._analysis_worker.progress.connect(self.log)
        self._analysis_worker.finished.connect(
            lambda result: self._on_analysis_done(result, paths, audio_path, not use_own_audio)
        )
        self._analysis_worker.error.connect(self._on_error)
        self._analysis_worker.start()

    def _extract_audio(self, video_path: str) -> str | None:
        audio_out = str(self._temp_manager.session_dir / "extracted_audio.wav")
        cmd = self._runner.build_command(
            inputs=[video_path], output=audio_out,
            extra_args=["-vn", "-ac", "1", "-ar", "22050"],
        )
        result = self._runner.run(cmd)
        if result.returncode != 0:
            self.log(f"Audio extraction failed: {result.stderr[-200:]}")
            return None
        return audio_out

    def _on_analysis_done(self, result, paths, audio_path, use_separate_audio):
        if result is None:
            self.log("Analysis cancelled")
            self._reset_ui()
            return

        self.log(f"Analysis complete: {result.tempo:.0f} BPM, {len(result.beats)} beats")

        from core.cut_generator import CutGenerator
        gen = CutGenerator(
            min_cut=self._config.min_cut_duration,
            max_cut=self._config.max_cut_duration,
            beat_snap_ms=self._config.beat_snap_tolerance_ms,
        )
        cuts = gen.generate(result, num_sources=len(paths))
        self.log(f"Generated {len(cuts)} cut points")

        from core.export_manager import ExportManager
        em = ExportManager()
        probes = []
        for p in paths:
            try:
                probes.append(self._runner.probe(p))
            except Exception as e:
                self.log(f"Warning: Failed to probe {p}: {e}")

        output_folder = self._output_folder.text() or str(Path(paths[0]).parent)
        output_name = self._output_name.text() or "output_edited.mp4"
        output_path = str(Path(output_folder) / output_name)

        force_cpu = self._config.force_cpu
        codec = self._gpu_detector.get_video_encoder(force_cpu)
        crf = self._gpu_detector.get_crf(force_cpu)

        settings = em.resolve_settings(probes, output_path, codec, crf)
        self.log(f"Export: {settings.width}x{settings.height} @ {settings.fps}fps, {settings.video_codec}")

        from gui.workers import ExportWorker
        self._export_worker = ExportWorker(
            cuts=cuts, ffmpeg_runner=self._runner,
            filter_builder=self._filter_builder, settings=settings,
            source_paths=paths,
            audio_path=audio_path if use_separate_audio else None,
            temp_dir=str(self._temp_manager.session_dir),
            use_separate_audio=use_separate_audio,
        )
        self._export_worker.progress.connect(self.set_progress)
        self._export_worker.stage.connect(self.set_stage)
        self._export_worker.finished.connect(self._on_export_done)
        self._export_worker.error.connect(self._on_error)
        self._export_worker.start()

    def _on_export_done(self, success: bool):
        if success:
            self.log("Export completed successfully!")
            self.set_stage("Done!")
            QMessageBox.information(self, "Success", "Video export complete!")
        self._temp_manager.cleanup_session()
        self._reset_ui()

    def _on_error(self, message: str):
        self.log(f"Error: {message}")
        QMessageBox.critical(self, "Error", message)
        self._reset_ui()

    def _cancel_pipeline(self):
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.cancel()
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.cancel()

    def _reset_ui(self):
        self._go_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._update_go_state()
