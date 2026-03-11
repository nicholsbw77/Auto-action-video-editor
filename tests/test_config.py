import json
import os
import pytest
from config import AppConfig


@pytest.fixture
def tmp_config(tmp_path):
    return tmp_path / "config.json"


class TestAppConfig:
    def test_defaults_when_no_file(self, tmp_config):
        cfg = AppConfig(tmp_config)
        assert cfg.ffmpeg_path == ""
        assert cfg.ffprobe_path == ""
        assert cfg.gpu_detected is False
        assert cfg.force_cpu is False
        assert cfg.last_output_folder == ""
        assert cfg.last_input_folder == ""
        assert cfg.recent_files == []
        assert cfg.beat_snap_tolerance_ms == 50
        assert cfg.min_cut_duration == 0.5
        assert cfg.max_cut_duration == 8.0

    def test_save_and_load(self, tmp_config):
        cfg = AppConfig(tmp_config)
        cfg.ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"
        cfg.force_cpu = True
        cfg.save()

        cfg2 = AppConfig(tmp_config)
        assert cfg2.ffmpeg_path == "C:/ffmpeg/bin/ffmpeg.exe"
        assert cfg2.force_cpu is True

    def test_recent_files_max_10(self, tmp_config):
        cfg = AppConfig(tmp_config)
        for i in range(15):
            cfg.add_recent_file(f"video_{i}.mp4")
        assert len(cfg.recent_files) == 10
        assert cfg.recent_files[0] == "video_14.mp4"

    def test_recent_files_no_duplicates(self, tmp_config):
        cfg = AppConfig(tmp_config)
        cfg.add_recent_file("video.mp4")
        cfg.add_recent_file("other.mp4")
        cfg.add_recent_file("video.mp4")
        assert len(cfg.recent_files) == 2
        assert cfg.recent_files[0] == "video.mp4"

    def test_corrupt_file_uses_defaults(self, tmp_config):
        tmp_config.write_text("not valid json {{{")
        cfg = AppConfig(tmp_config)
        assert cfg.ffmpeg_path == ""
        assert cfg.beat_snap_tolerance_ms == 50

    def test_partial_file_fills_missing_keys(self, tmp_config):
        tmp_config.write_text(json.dumps({"ffmpeg_path": "/usr/bin/ffmpeg"}))
        cfg = AppConfig(tmp_config)
        assert cfg.ffmpeg_path == "/usr/bin/ffmpeg"
        assert cfg.beat_snap_tolerance_ms == 50
