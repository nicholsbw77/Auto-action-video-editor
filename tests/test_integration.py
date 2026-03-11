import pytest
from unittest.mock import MagicMock, patch
from core.models import AnalysisResult, CutPoint
from core.cut_generator import CutGenerator
from ffmpeg.filter_graph import FilterGraphBuilder
from core.export_manager import ExportManager
import numpy as np


class TestPipelineIntegration:
    def test_analysis_to_cuts_to_filter_graph(self):
        """End-to-end: analysis result -> cuts -> filter graph string."""
        analysis = AnalysisResult(
            beats=[1.0, 2.0, 3.0, 4.0, 5.0],
            onsets=[1.0, 2.0, 3.0, 4.0, 5.0],
            tempo=120.0,
            energy_envelope=np.random.rand(50),
            energy_times=np.linspace(0, 6, 50),
            spectral_centroids=np.random.rand(50) * 3000,
            onset_env=np.random.rand(50),
            duration=6.0,
        )

        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        assert len(cuts) >= 1

        builder = FilterGraphBuilder()
        groups = builder.group_cuts(cuts)
        assert len(groups) >= 1

        for group in groups:
            mapping = builder.build_source_mapping(group)
            graph = builder.build_xfade_graph(group, mapping)
            assert "trim" in graph

    def test_export_settings_from_probe(self):
        probe = {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080,
                 "r_frame_rate": "60/1", "avg_frame_rate": "60/1"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "30.0"},
        }
        em = ExportManager()
        settings = em.resolve_settings([probe], "out.mp4", "libx264", 18)
        assert settings.width == 1920
        assert settings.fps == 60.0
        assert settings.audio_codec == "aac"
