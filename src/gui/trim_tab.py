import os
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QProgressBar, QGroupBox, QScrollArea, QFrame, QMessageBox,
)
from PyQt6.QtCore import Qt
from gui.widgets import TimecodeInput
from core.models import TrimRegion

logger = logging.getLogger("autoeditor.gui")


class TrimRegionWidget(QFrame):
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        layout = QHBoxLayout(self)

        self._label = QLabel(f"Region {index + 1}")
        layout.addWidget(self._label)

        self._start = TimecodeInput("Start:")
        layout.addWidget(self._start)

        self._end = TimecodeInput("End:")
        layout.addWidget(self._end)

        self._remove_btn = QPushButton("Remove")
        layout.addWidget(self._remove_btn)

    def get_region(self) -> TrimRegion:
        return TrimRegion(
            start=self._start.get_value(),
            end=self._end.get_value(),
            label=self._label.text(),
        )

    def set_max_time(self, duration: float):
        pass  # Validation done on export


class TrimTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: str | None = None
        self._video_duration: float = 0.0
        self._region_widgets: list[TrimRegionWidget] = []
        self._runner = None
        self._trim_worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Video loader
        file_group = QGroupBox("Video File")
        file_layout = QHBoxLayout(file_group)
        self._load_btn = QPushButton("Load Video")
        self._load_btn.clicked.connect(self._load_video)
        file_layout.addWidget(self._load_btn)
        self._file_label = QLabel("No video loaded")
        file_layout.addWidget(self._file_label)
        self._duration_label = QLabel("")
        file_layout.addWidget(self._duration_label)
        file_layout.addStretch()
        layout.addWidget(file_group)

        # Regions
        regions_group = QGroupBox("Trim Regions")
        regions_layout = QVBoxLayout(regions_group)

        self._regions_container = QVBoxLayout()
        regions_layout.addLayout(self._regions_container)

        self._add_region_btn = QPushButton("Add Region")
        self._add_region_btn.clicked.connect(self._add_region)
        regions_layout.addWidget(self._add_region_btn)

        layout.addWidget(regions_group)

        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)

        self._mode_group = QButtonGroup(self)
        self._separate_radio = QRadioButton("Export as separate files")
        self._concat_radio = QRadioButton("Concatenate into one file")
        self._separate_radio.setChecked(True)
        self._mode_group.addButton(self._separate_radio)
        self._mode_group.addButton(self._concat_radio)
        self._concat_radio.toggled.connect(self._on_concat_toggled)
        export_layout.addWidget(self._separate_radio)
        export_layout.addWidget(self._concat_radio)

        self._reencode_check = QCheckBox("Force re-encode for frame accuracy")
        self._reencode_check.setToolTip(
            "When unchecked, uses stream copy (fast but cuts only on keyframes). "
            "When checked, re-encodes for frame-accurate cuts."
        )
        export_layout.addWidget(self._reencode_check)

        layout.addWidget(export_group)

        # Export button and progress
        self._export_btn = QPushButton("Export Trim")
        self._export_btn.setEnabled(False)
        self._export_btn.setMinimumHeight(40)
        self._export_btn.clicked.connect(self._export_trim)
        layout.addWidget(self._export_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        layout.addStretch()

        # Add initial region
        self._add_region()

    def set_dependencies(self, config, ffmpeg_runner, temp_manager):
        self._config = config
        self._runner = ffmpeg_runner
        self._temp_manager = temp_manager

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.ts);;All Files (*)",
        )
        if path:
            self._video_path = path
            self._file_label.setText(os.path.basename(path))
            self._update_export_state()

    def _add_region(self):
        idx = len(self._region_widgets)
        widget = TrimRegionWidget(idx)
        widget._remove_btn.clicked.connect(lambda: self._remove_region(widget))
        self._region_widgets.append(widget)
        self._regions_container.addWidget(widget)
        self._update_export_state()

    def _remove_region(self, widget: TrimRegionWidget):
        if len(self._region_widgets) <= 1:
            return  # Keep at least one
        self._region_widgets.remove(widget)
        self._regions_container.removeWidget(widget)
        widget.deleteLater()
        # Renumber
        for i, w in enumerate(self._region_widgets):
            w._label.setText(f"Region {i + 1}")
        self._update_export_state()

    def _on_concat_toggled(self, checked: bool):
        if checked:
            self._reencode_check.setChecked(True)
            self._reencode_check.setEnabled(False)
            self._reencode_check.setToolTip(
                "Re-encode is required when concatenating to avoid corrupted frames"
            )
        else:
            self._reencode_check.setEnabled(True)
            self._reencode_check.setToolTip(
                "When unchecked, uses stream copy (fast but cuts only on keyframes)"
            )

    def _update_export_state(self):
        has_video = self._video_path is not None
        has_regions = len(self._region_widgets) > 0
        self._export_btn.setEnabled(has_video and has_regions)

    def get_regions(self) -> list[TrimRegion]:
        regions = []
        for w in self._region_widgets:
            r = w.get_region()
            if r.end > r.start:
                regions.append(r)
        return regions

    def _export_trim(self):
        regions = self.get_regions()
        if not regions or not self._video_path:
            return

        # Validate
        for r in regions:
            if r.end <= r.start:
                QMessageBox.warning(self, "Invalid Region", f"{r.label}: end must be after start")
                return
            if r.end - r.start < 0.5:
                QMessageBox.warning(self, "Short Region", f"{r.label} is less than 0.5 seconds")
                return

        is_concat = self._concat_radio.isChecked()
        reencode = self._reencode_check.isChecked()
        source_name = os.path.splitext(os.path.basename(self._video_path))[0]

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not output_dir:
            return

        self._export_btn.setEnabled(False)
        self._progress.setValue(0)

        session_dir = self._temp_manager.create_session()

        from gui.workers import TrimWorker
        self._trim_worker = TrimWorker(
            video_path=self._video_path,
            regions=regions,
            output_dir=output_dir,
            is_concat=is_concat,
            reencode=reencode,
            source_name=source_name,
            runner=self._runner,
            session_dir=str(session_dir),
        )
        self._trim_worker.progress.connect(self._progress.setValue)
        self._trim_worker.finished.connect(self._on_trim_done)
        self._trim_worker.error.connect(self._on_trim_error)
        self._trim_worker.start()

    def _on_trim_done(self, success: bool):
        self._temp_manager.cleanup_session()
        self._export_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "Done", "Trim export complete!")

    def _on_trim_error(self, message: str):
        self._temp_manager.cleanup_session()
        self._export_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", message)
