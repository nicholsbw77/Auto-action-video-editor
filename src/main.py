import sys
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from paths import get_app_dir
from config import AppConfig
from core.logger import setup_logging
from core.temp_manager import TempManager
from ffmpeg.detector import FFmpegDetector
from ffmpeg.gpu import GPUDetector
from gui.main_window import MainWindow


def main():
    app_dir = get_app_dir()
    log_dir = app_dir / "logs"
    temp_dir = app_dir / "temp"

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("Starting Auto Video Editor")

    # Load config
    config = AppConfig(app_dir / "config.json")

    # Cleanup stale temp files
    temp_mgr = TempManager(temp_dir)
    temp_mgr.cleanup_stale(max_age_hours=24)

    # Detect FFmpeg
    detector = FFmpegDetector()
    ffmpeg_path = config.ffmpeg_path or detector.find_ffmpeg()
    ffprobe_path = config.ffprobe_path or detector.find_ffprobe()

    if ffmpeg_path and detector.validate_version(ffmpeg_path):
        config.ffmpeg_path = ffmpeg_path
        if ffprobe_path:
            config.ffprobe_path = ffprobe_path
        logger.info("FFmpeg found: %s", ffmpeg_path)
    else:
        logger.warning("FFmpeg not found or version too old")

    # Detect GPU
    if ffmpeg_path:
        gpu_det = GPUDetector(ffmpeg_path)
        config.gpu_detected = gpu_det.detect_nvenc()
    else:
        config.gpu_detected = False

    config.save()

    # Launch GUI
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("Auto Video Editor")

    if not ffmpeg_path:
        reply = QMessageBox.question(
            None, "FFmpeg Not Found",
            "FFmpeg was not found on your system.\n\n"
            "Would you like to download it automatically (~40 MB)?\n\n"
            "Alternatively, download it manually from https://ffmpeg.org/download.html\n"
            "or set the path in Settings > Set FFmpeg Path.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            from PyQt6.QtWidgets import QProgressDialog
            progress = QProgressDialog("Downloading FFmpeg...", None, 0, 100)
            progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress.setMinimumDuration(0)
            progress.show()
            qt_app.processEvents()

            try:
                from ffmpeg.downloader import FFmpegDownloader
                dl = FFmpegDownloader()

                def _update_progress(p: float):
                    progress.setValue(int(p * 100))
                    qt_app.processEvents()

                ffmpeg_path, ffprobe_path = dl.download(progress_callback=_update_progress)
                config.ffmpeg_path = ffmpeg_path
                config.ffprobe_path = ffprobe_path
                logger.info("FFmpeg downloaded to: %s", ffmpeg_path)

                gpu_det = GPUDetector(ffmpeg_path)
                config.gpu_detected = gpu_det.detect_nvenc()
                config.save()
                progress.close()
            except Exception as e:
                progress.close()
                logger.exception("FFmpeg download failed")
                QMessageBox.critical(None, "Download Failed", str(e))

    window = MainWindow(config)
    window.show()

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
