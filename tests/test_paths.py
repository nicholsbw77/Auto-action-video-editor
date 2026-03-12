import sys
from pathlib import Path
from unittest.mock import patch

from paths import get_app_dir


class TestGetAppDir:
    def test_development_mode_returns_path(self):
        """In development, returns a valid Path (the src/ directory)."""
        result = get_app_dir()
        assert isinstance(result, Path)
        assert result.is_dir()

    def test_development_mode_is_src_dir(self):
        """In development, the returned dir contains main.py."""
        result = get_app_dir()
        assert (result / "main.py").exists()

    def test_frozen_mode_returns_exe_parent(self, tmp_path):
        """When frozen, returns the parent of sys.executable."""
        fake_exe = tmp_path / "dist" / "AutoVideoEditor" / "AutoVideoEditor.exe"
        fake_exe.parent.mkdir(parents=True, exist_ok=True)
        fake_exe.touch()

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", str(fake_exe)):
            result = get_app_dir()
            assert result == fake_exe.parent

    def test_frozen_mode_returns_path_object(self, tmp_path):
        """get_app_dir always returns a Path object even when frozen."""
        fake_exe = tmp_path / "app.exe"
        fake_exe.touch()

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "executable", str(fake_exe)):
            result = get_app_dir()
            assert isinstance(result, Path)
