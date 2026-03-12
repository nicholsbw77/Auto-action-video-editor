import pytest
from unittest.mock import MagicMock, patch
from core.models import CutPoint, ExportSettings
from core.export_manager import ExportManager


class TestExportManager:
    def test_resolve_settings_single_video(self):
        probe_data = {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080,
                 "r_frame_rate": "30/1", "avg_frame_rate": "30/1"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "60.0"},
        }
        em = ExportManager()
        settings = em.resolve_settings(
            probe_data=[probe_data], output_path="out.mp4",
            video_codec="libx264", video_crf=18,
        )
        assert settings.width == 1920
        assert settings.height == 1080
        assert settings.fps == 30.0
        assert settings.audio_codec == "aac"

    def test_resolution_capped_at_1080p(self):
        probe_data = {
            "streams": [
                {"codec_type": "video", "width": 3840, "height": 2160,
                 "r_frame_rate": "60/1", "avg_frame_rate": "60/1"},
            ],
            "format": {"duration": "60.0"},
        }
        em = ExportManager()
        settings = em.resolve_settings(
            probe_data=[probe_data], output_path="out.mp4",
            video_codec="libx264", video_crf=18,
        )
        assert settings.width == 1920
        assert settings.height == 1080

    def test_fps_capped_at_60(self):
        probe_data = {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080,
                 "r_frame_rate": "120/1", "avg_frame_rate": "120/1"},
            ],
            "format": {"duration": "60.0"},
        }
        em = ExportManager()
        settings = em.resolve_settings(
            probe_data=[probe_data], output_path="out.mp4",
            video_codec="libx264", video_crf=18,
        )
        assert settings.fps == 60.0

    def test_multi_clip_uses_max_resolution(self):
        probes = [
            {"streams": [{"codec_type": "video", "width": 1280, "height": 720,
                          "r_frame_rate": "30/1", "avg_frame_rate": "30/1"}],
             "format": {"duration": "30.0"}},
            {"streams": [{"codec_type": "video", "width": 1920, "height": 1080,
                          "r_frame_rate": "60/1", "avg_frame_rate": "60/1"}],
             "format": {"duration": "30.0"}},
        ]
        em = ExportManager()
        settings = em.resolve_settings(
            probe_data=probes, output_path="out.mp4",
            video_codec="libx264", video_crf=18,
        )
        assert settings.width == 1920
        assert settings.height == 1080
        assert settings.fps == 60.0

    def test_estimate_disk_usage(self):
        em = ExportManager()
        # 60s at 1080p60 ≈ 150 MB/min = 150 MB
        estimate = em.estimate_disk_usage(
            duration_s=60.0, width=1920, height=1080, fps=60.0,
            num_clips=1, has_vfr=False,
        )
        assert estimate > 0

    def test_detect_vfr(self):
        em = ExportManager()
        stream = {"r_frame_rate": "30/1", "avg_frame_rate": "25/1"}
        assert em.is_vfr(stream) is True

    def test_detect_cfr(self):
        em = ExportManager()
        stream = {"r_frame_rate": "30/1", "avg_frame_rate": "29.97/1"}
        assert em.is_vfr(stream) is False

    def test_empty_probe_uses_defaults(self):
        """Critical #5: empty probes should use safe defaults, not 0x0"""
        em = ExportManager()
        settings = em.resolve_settings(
            probe_data=[], output_path="out.mp4",
            video_codec="libx264", video_crf=18,
        )
        assert settings.width > 0
        assert settings.height > 0
        assert settings.fps > 0.0

    def test_no_video_streams_uses_defaults(self):
        """Critical #5: probes with only audio should use safe defaults"""
        probe_data = {
            "streams": [{"codec_type": "audio"}],
            "format": {"duration": "60.0"},
        }
        em = ExportManager()
        settings = em.resolve_settings(
            probe_data=[probe_data], output_path="out.mp4",
            video_codec="libx264", video_crf=18,
        )
        assert settings.width > 0
        assert settings.height > 0

    def test_odd_dimensions_rounded_to_even(self):
        """Critical #6: odd dimensions must be rounded to even for codec compatibility"""
        probe_data = {
            "streams": [
                {"codec_type": "video", "width": 1279, "height": 719,
                 "r_frame_rate": "30/1", "avg_frame_rate": "30/1"},
            ],
            "format": {"duration": "60.0"},
        }
        em = ExportManager()
        settings = em.resolve_settings(
            probe_data=[probe_data], output_path="out.mp4",
            video_codec="libx264", video_crf=18,
        )
        assert settings.width % 2 == 0
        assert settings.height % 2 == 0
        assert settings.width == 1278
        assert settings.height == 718
