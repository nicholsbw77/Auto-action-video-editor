import re
from unittest.mock import patch, MagicMock
from ffmpeg.runner import FFmpegRunner, parse_progress_time


class TestParseProgressTime:
    def test_parse_time(self):
        assert parse_progress_time("time=00:01:30.50") == 90.5

    def test_parse_time_hours(self):
        assert parse_progress_time("time=01:00:00.00") == 3600.0

    def test_no_match(self):
        assert parse_progress_time("some other output") is None


class TestFFmpegRunner:
    def test_build_basic_command(self):
        runner = FFmpegRunner("ffmpeg")
        cmd = runner.build_command(
            inputs=["input.mp4"],
            output="output.mp4",
            filter_complex=None,
            maps=None,
            extra_args=["-c:v", "libx264"],
        )
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert "input.mp4" in cmd
        assert "output.mp4" == cmd[-1]

    def test_build_command_with_filter(self):
        runner = FFmpegRunner("ffmpeg")
        cmd = runner.build_command(
            inputs=["a.mp4", "b.mp4"],
            output="out.mp4",
            filter_complex="[0:v][1:v]xfade=transition=fade:duration=0.3:offset=2.7[vout]",
            maps=["[vout]"],
            extra_args=[],
        )
        assert "-filter_complex" in cmd
        assert "-map" in cmd

    def test_probe_returns_dict(self):
        mock_result = MagicMock()
        mock_result.stdout = '{"streams": [], "format": {}}'
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            runner = FFmpegRunner("ffmpeg", ffprobe_path="ffprobe")
            info = runner.probe("test.mp4")
            assert "streams" in info
