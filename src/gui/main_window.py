import logging
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu, QStatusBar,
    QMessageBox, QFileDialog,
)
from PyQt6.QtGui import QAction
from paths import get_app_dir
from gui.auto_edit_tab import AutoEditTab
from gui.trim_tab import TrimTab
from config import AppConfig
from ffmpeg.runner import FFmpegRunner
from ffmpeg.filter_graph import FilterGraphBuilder
from ffmpeg.gpu import GPUDetector
from core.temp_manager import TempManager

logger = logging.getLogger("autoeditor.gui")


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self.setWindowTitle("Auto Video Editor")
        self.setMinimumSize(800, 600)

        self._init_menu()
        self._init_tabs()
        self._init_status_bar()

    def _init_menu(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Video...", self)
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)

        # Recent files submenu
        self._recent_menu = QMenu("Recent Files", self)
        file_menu.addMenu(self._recent_menu)
        self._update_recent_menu()

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings menu
        settings_menu = menu_bar.addMenu("Settings")

        ffmpeg_action = QAction("Set FFmpeg Path...", self)
        ffmpeg_action.triggered.connect(self._set_ffmpeg_path)
        settings_menu.addAction(ffmpeg_action)

        self._gpu_action = QAction("Force CPU Encoding", self)
        self._gpu_action.setCheckable(True)
        self._gpu_action.setChecked(self._config.force_cpu)
        self._gpu_action.triggered.connect(self._toggle_gpu)
        settings_menu.addAction(self._gpu_action)

    def _init_tabs(self):
        self._tabs = QTabWidget()
        self._auto_edit_tab = AutoEditTab()
        self._trim_tab = TrimTab()
        self._tabs.addTab(self._auto_edit_tab, "Auto-Edit")
        self._tabs.addTab(self._trim_tab, "Trim")
        self.setCentralWidget(self._tabs)

        # Inject dependencies
        app_dir = get_app_dir()
        runner = FFmpegRunner(self._config.ffmpeg_path or "ffmpeg", self._config.ffprobe_path or "ffprobe")
        filter_builder = FilterGraphBuilder()
        gpu_det = GPUDetector(self._config.ffmpeg_path or "ffmpeg")
        if self._config.gpu_detected and not self._config.force_cpu:
            gpu_det.nvenc_available = self._config.gpu_detected
        temp_mgr = TempManager(app_dir / "temp")

        self._auto_edit_tab.set_dependencies(self._config, runner, filter_builder, gpu_det, temp_mgr)
        self._trim_tab.set_dependencies(self._config, runner, temp_mgr)

    def _init_status_bar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        ffmpeg_status = "FFmpeg: " + (self._config.ffmpeg_path or "Not found")
        gpu_status = "GPU: " + ("Available" if self._config.gpu_detected else "Not available")
        self._status.showMessage(f"{ffmpeg_status} | {gpu_status}")

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", self._config.last_input_folder,
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.ts);;All Files (*)",
        )
        if path:
            import os
            self._config.last_input_folder = os.path.dirname(path)
            self._config.add_recent_file(path)
            self._config.save()
            self._update_recent_menu()
            self._auto_edit_tab._file_list.add_file(path)

    def _update_recent_menu(self):
        self._recent_menu.clear()
        import os
        for filepath in self._config.recent_files:
            if os.path.exists(filepath):
                action = QAction(os.path.basename(filepath), self)
                action.setData(filepath)
                action.triggered.connect(lambda checked, p=filepath: self._open_recent(p))
                self._recent_menu.addAction(action)

    def _open_recent(self, path: str):
        self._auto_edit_tab._file_list.add_file(path)

    def _set_ffmpeg_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate FFmpeg", "",
            "Executables (*.exe);;All Files (*)",
        )
        if path:
            self._config.ffmpeg_path = path
            self._config.save()
            self._status.showMessage(f"FFmpeg: {path}")

    def _toggle_gpu(self, checked: bool):
        self._config.force_cpu = checked
        self._config.save()
