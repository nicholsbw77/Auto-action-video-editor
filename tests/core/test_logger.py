import logging
from pathlib import Path
from core.logger import setup_logging


def test_setup_creates_log_file(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)
    logger = logging.getLogger("autoeditor")
    logger.info("test message")
    log_file = log_dir / "autoeditor.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "test message" in content


def test_setup_returns_logger(tmp_path):
    logger = setup_logging(tmp_path / "logs")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "autoeditor"
