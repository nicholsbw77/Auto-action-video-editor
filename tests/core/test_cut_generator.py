import numpy as np
import pytest
from core.models import AnalysisResult, CutPoint
from core.cut_generator import CutGenerator


def make_analysis(duration=30.0, num_beats=15):
    beat_times = np.linspace(0.5, duration - 0.5, num_beats).tolist()
    return AnalysisResult(
        beats=beat_times,
        onsets=beat_times,
        tempo=120.0,
        energy_envelope=np.random.rand(100),
        energy_times=np.linspace(0, duration, 100),
        spectral_centroids=np.random.rand(100) * 3000,
        onset_env=np.random.rand(100),
        duration=duration,
    )


class TestCutGenerator:
    def test_generates_cut_points(self):
        analysis = make_analysis()
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        assert len(cuts) > 0
        assert all(isinstance(c, CutPoint) for c in cuts)

    def test_cuts_cover_duration(self):
        analysis = make_analysis(duration=20.0)
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        assert cuts[0].start == 0.0
        assert cuts[-1].end <= analysis.duration + 0.1

    def test_min_cut_duration_enforced(self):
        analysis = make_analysis(duration=10.0, num_beats=50)
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        for cut in cuts:
            assert cut.end - cut.start >= 0.4  # allow small tolerance from snap

    def test_max_cut_duration_enforced(self):
        analysis = make_analysis(duration=60.0, num_beats=3)
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        for cut in cuts:
            assert cut.end - cut.start <= 8.5  # small tolerance

    def test_valid_transition_types(self):
        valid = {"hard_cut", "crossfade", "crossfade_slow", "fade_black", "wipe_left", "wipe_right"}
        analysis = make_analysis()
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        for cut in cuts:
            assert cut.transition_type in valid

    def test_multi_clip_round_robin(self):
        analysis = make_analysis(duration=20.0)
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=3)
        sources_used = {c.source_index for c in cuts}
        assert len(sources_used) > 1  # should use multiple sources

    def test_wipe_every_third(self):
        # With high energy throughout and varying onset strength so some beats are "strong"
        # onset_env must vary so median splits strong/weak; alternating high/low
        # Beats every 0.5s so midpoints align within < 0.5s threshold
        # Onset envelope increases linearly so later beats are "strong" (> median)
        # Energy is all high so _get_energy_at_time returns "high" for later cuts
        beats = np.arange(0.5, 30.0, 0.5).tolist()
        # onset_env increases: first half weak, second half strong
        onset_env = np.linspace(0.0, 1.0, 200)
        # energy: first quarter low, rest linearly increasing — ensures varied > p75
        energy = np.concatenate([np.ones(50) * 0.1, np.linspace(0.5, 1.0, 150)])
        analysis = AnalysisResult(
            beats=beats,
            onsets=beats,
            tempo=120.0,
            energy_envelope=energy,
            energy_times=np.linspace(0, 30, 200),
            spectral_centroids=np.ones(200) * 2000,
            onset_env=onset_env,
            duration=30.0,
        )
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        wipes = [c for c in cuts if c.transition_type in ("wipe_left", "wipe_right")]
        # Should have at least one wipe among many high-energy strong-beat cuts
        assert len(wipes) >= 1

    def test_no_beats_fallback(self):
        analysis = AnalysisResult(
            beats=[], onsets=[], tempo=0.0,
            energy_envelope=np.zeros(100),
            energy_times=np.linspace(0, 15, 100),
            spectral_centroids=np.zeros(100),
            onset_env=np.zeros(100),
            duration=15.0,
        )
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        assert len(cuts) > 0
        # Fallback: fixed 3s intervals with hard cuts
        for c in cuts:
            assert c.transition_type == "hard_cut"

    def test_very_short_audio(self):
        analysis = AnalysisResult(
            beats=[0.5], onsets=[], tempo=60.0,
            energy_envelope=np.array([0.5]),
            energy_times=np.array([0.0]),
            spectral_centroids=np.array([1000]),
            onset_env=np.array([0.5]),
            duration=1.5,
        )
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        assert len(cuts) == 1
        assert cuts[0].start == 0.0
        assert cuts[0].end == 1.5
