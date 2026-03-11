import time
import os
from pathlib import Path
from core.temp_manager import TempManager


class TestTempManager:
    def test_create_session_dir(self, tmp_path):
        tm = TempManager(tmp_path / "temp")
        session_dir = tm.create_session()
        assert session_dir.exists()
        assert session_dir.parent == tmp_path / "temp"

    def test_cleanup_session(self, tmp_path):
        tm = TempManager(tmp_path / "temp")
        session_dir = tm.create_session()
        (session_dir / "test.mp4").write_text("data")
        tm.cleanup_session()
        assert not session_dir.exists()

    def test_cleanup_stale_sessions(self, tmp_path):
        temp_root = tmp_path / "temp"
        temp_root.mkdir()
        stale_dir = temp_root / "stale_session"
        stale_dir.mkdir()
        (stale_dir / "file.tmp").write_text("old")
        old_time = time.time() - (25 * 3600)
        os.utime(stale_dir, (old_time, old_time))

        tm = TempManager(temp_root)
        tm.cleanup_stale(max_age_hours=24)
        assert not stale_dir.exists()

    def test_keep_recent_sessions(self, tmp_path):
        temp_root = tmp_path / "temp"
        temp_root.mkdir()
        recent_dir = temp_root / "recent_session"
        recent_dir.mkdir()
        (recent_dir / "file.tmp").write_text("new")

        tm = TempManager(temp_root)
        tm.cleanup_stale(max_age_hours=24)
        assert recent_dir.exists()
