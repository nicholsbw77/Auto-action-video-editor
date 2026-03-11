import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from core.audio_analyzer import AudioAnalyzer
from core.models import AnalysisResult


class TestAudioAnalyzer:
    def test_analyze_returns_analysis_result(self):
        # Mock librosa functions
        fake_y = np.random.randn(22050 * 10).astype(np.float32)  # 10s audio
        fake_sr = 22050
        with patch("librosa.load", return_value=(fake_y, fake_sr)), \
             patch("librosa.beat.beat_track", return_value=(120.0, np.array([10, 50, 90]))), \
             patch("librosa.onset.onset_detect", return_value=np.array([5, 45, 85])), \
             patch("librosa.feature.rms", return_value=np.array([[0.1, 0.5, 0.9]])), \
             patch("librosa.times_like", return_value=np.array([0.0, 5.0, 10.0])), \
             patch("librosa.feature.spectral_centroid", return_value=np.array([[1000, 2000, 3000]])), \
             patch("librosa.frames_to_time", return_value=np.array([0.5, 2.3, 4.1])), \
             patch("librosa.onset.onset_strength", return_value=np.array([0.3, 0.7, 0.5])):
            analyzer = AudioAnalyzer()
            result = analyzer.analyze("test_audio.wav")
            assert isinstance(result, AnalysisResult)
            assert result.tempo == 120.0
            assert result.duration > 0
            assert len(result.beats) > 0

    def test_classify_energy(self):
        analyzer = AudioAnalyzer()
        energy = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0])
        levels = analyzer.classify_energy(energy)
        assert "high" in levels
        assert "medium" in levels
        assert "low" in levels

    def test_detect_energy_drops(self):
        analyzer = AudioAnalyzer()
        # Simulate a sudden drop: high energy then sudden low
        energy = np.concatenate([np.ones(50) * 0.8, np.ones(10) * 0.1, np.ones(50) * 0.5])
        times = np.linspace(0, 5.0, len(energy))
        drops = analyzer.detect_energy_drops(energy, times)
        assert len(drops) > 0

    def test_detect_energy_spikes(self):
        analyzer = AudioAnalyzer()
        energy = np.concatenate([np.ones(50) * 0.1, np.ones(10) * 0.9, np.ones(50) * 0.3])
        times = np.linspace(0, 5.0, len(energy))
        spikes = analyzer.detect_energy_spikes(energy, times)
        assert len(spikes) > 0

    def test_classify_beat_strength(self):
        analyzer = AudioAnalyzer()
        beat_times = np.array([1.0, 2.0, 3.0, 4.0])
        onset_env = np.array([0.2, 0.8, 0.3, 0.9])
        strengths = analyzer.classify_beat_strength(beat_times, onset_env)
        assert len(strengths) == 4
        assert all(s in ("strong", "weak") for s in strengths)
