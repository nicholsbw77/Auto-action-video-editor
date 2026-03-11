import numpy as np
import pytest
from core.models import AnalysisResult, CutPoint, ExportSettings, TrimRegion


class TestAnalysisResult:
    def test_creation(self):
        result = AnalysisResult(
            beats=[1.0, 2.0, 3.0],
            onsets=[0.5, 1.5],
            tempo=120.0,
            energy_envelope=np.array([0.1, 0.5, 0.9]),
            energy_times=np.array([0.0, 1.0, 2.0]),
            spectral_centroids=np.array([1000.0, 2000.0, 3000.0]),
            onset_env=np.array([0.3, 0.7, 0.5]),
            duration=10.0,
        )
        assert result.tempo == 120.0
        assert len(result.beats) == 3
        assert result.duration == 10.0
        assert len(result.onset_env) == 3

    def test_energy_envelope_is_numpy(self):
        result = AnalysisResult(
            beats=[], onsets=[], tempo=0.0,
            energy_envelope=np.array([0.1]),
            energy_times=np.array([0.0]),
            spectral_centroids=np.array([1000.0]),
            onset_env=np.array([0.5]),
            duration=1.0,
        )
        assert isinstance(result.energy_envelope, np.ndarray)


class TestCutPoint:
    def test_hard_cut(self):
        cp = CutPoint(start=0.0, end=3.0, source_index=0,
                       transition_type="hard_cut", transition_duration=0.0)
        assert cp.transition_type == "hard_cut"
        assert cp.transition_duration == 0.0

    def test_crossfade(self):
        cp = CutPoint(start=3.0, end=7.0, source_index=1,
                       transition_type="crossfade", transition_duration=0.3)
        assert cp.transition_duration == 0.3
        assert cp.source_index == 1

    def test_all_transition_types(self):
        valid_types = ["hard_cut", "crossfade", "crossfade_slow",
                       "fade_black", "wipe_left", "wipe_right"]
        for t in valid_types:
            cp = CutPoint(start=0.0, end=1.0, source_index=0,
                           transition_type=t, transition_duration=0.3)
            assert cp.transition_type == t


class TestExportSettings:
    def test_default_export(self):
        es = ExportSettings(
            output_path="output.mp4", width=1920, height=1080, fps=60.0,
            video_codec="libx264", video_crf=18,
            audio_codec="aac", audio_bitrate="192k", container="mp4",
        )
        assert es.width == 1920
        assert es.audio_codec == "aac"

    def test_gpu_export(self):
        es = ExportSettings(
            output_path="output.mp4", width=1920, height=1080, fps=60.0,
            video_codec="h264_nvenc", video_crf=20,
            audio_codec="aac", audio_bitrate="192k", container="mp4",
        )
        assert es.video_codec == "h264_nvenc"


class TestTrimRegion:
    def test_creation(self):
        tr = TrimRegion(start=5.0, end=15.0, label="Region 1")
        assert tr.end - tr.start == 10.0
        assert tr.label == "Region 1"
