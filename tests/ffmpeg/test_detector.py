import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from ffmpeg.detector import FFmpegDetector


class TestFFmpegDetector:
    def test_find_on_path(self):
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda name: f"/usr/bin/{name}" if name in ("ffmpeg", "ffprobe") else None
            det = FFmpegDetector()
            assert det.find_ffmpeg() == "/usr/bin/ffmpeg"
            assert det.find_ffprobe() == "/usr/bin/ffprobe"

    def test_find_in_common_locations(self, tmp_path):
        ffmpeg_bin = tmp_path / "ffmpeg.exe"
        ffmpeg_bin.write_text("fake")
        with patch("shutil.which", return_value=None):
            det = FFmpegDetector(extra_search_paths=[str(tmp_path)])
            result = det.find_ffmpeg()
            assert result is not None

    def test_not_found(self):
        with patch("shutil.which", return_value=None):
            det = FFmpegDetector(extra_search_paths=[])
            assert det.find_ffmpeg() is None

    def test_validate_version_ok(self):
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 5.1.2 Copyright"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            det = FFmpegDetector()
            assert det.validate_version("/usr/bin/ffmpeg") is True

    def test_validate_version_too_old(self):
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 3.4.1 Copyright"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            det = FFmpegDetector()
            assert det.validate_version("/usr/bin/ffmpeg") is False
