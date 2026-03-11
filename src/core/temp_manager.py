import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("autoeditor.temp")


class TempManager:
    def __init__(self, temp_root: Path):
        self._root = temp_root
        self._session_dir: Path | None = None

    def create_session(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._session_dir = self._root / session_name
        self._session_dir.mkdir()
        logger.info("Created temp session: %s", self._session_dir)
        return self._session_dir

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def cleanup_session(self):
        if self._session_dir and self._session_dir.exists():
            shutil.rmtree(self._session_dir, ignore_errors=True)
            logger.info("Cleaned up temp session: %s", self._session_dir)
            self._session_dir = None

    def cleanup_stale(self, max_age_hours: int = 24):
        if not self._root.exists():
            return
        cutoff = time.time() - (max_age_hours * 3600)
        for entry in self._root.iterdir():
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                logger.info("Removed stale temp dir: %s", entry)
