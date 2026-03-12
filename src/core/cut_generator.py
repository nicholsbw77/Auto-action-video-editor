import logging

import numpy as np

from core.models import AnalysisResult, CutPoint
from core.audio_analyzer import detect_energy_drops as _detect_energy_drops
from core.audio_analyzer import detect_energy_spikes as _detect_energy_spikes

logger = logging.getLogger("autoeditor.cuts")

TRANSITION_DURATIONS = {
    "hard_cut": 0.0,
    "crossfade": 0.3,
    "crossfade_slow": 0.8,
    "fade_black": 0.5,
    "wipe_left": 0.2,
    "wipe_right": 0.2,
}

ALL_TRANSITIONS = list(TRANSITION_DURATIONS.keys())

AGGRESSIVENESS_TABLE = {
    1:  {"min_cut": 6.0, "max_cut": 15.0, "filter": "strong_beats_only"},
    2:  {"min_cut": 5.0, "max_cut": 12.0, "filter": "strong_beats_only"},
    3:  {"min_cut": 4.0, "max_cut": 10.0, "filter": "strong_beats_and_onsets"},
    4:  {"min_cut": 3.0, "max_cut": 10.0, "filter": "strong_beats_and_onsets"},
    5:  {"min_cut": 2.0, "max_cut": 8.0,  "filter": "all_beats_strong_onsets"},
    6:  {"min_cut": 1.5, "max_cut": 6.0,  "filter": "all_beats_strong_onsets"},
    7:  {"min_cut": 1.2, "max_cut": 5.0,  "filter": "all"},
    8:  {"min_cut": 1.0, "max_cut": 4.0,  "filter": "all"},
    9:  {"min_cut": 0.7, "max_cut": 3.0,  "filter": "all"},
    10: {"min_cut": 0.5, "max_cut": 2.0,  "filter": "all"},
}

TRANSITION_FALLBACKS = {
    "hard_cut": ["crossfade", "wipe_left", "wipe_right", "fade_black", "crossfade_slow"],
    "crossfade": ["crossfade_slow", "hard_cut", "fade_black"],
    "crossfade_slow": ["crossfade", "fade_black", "hard_cut"],
    "fade_black": ["crossfade_slow", "crossfade", "hard_cut"],
    "wipe_left": ["wipe_right", "hard_cut", "crossfade"],
    "wipe_right": ["wipe_left", "hard_cut", "crossfade"],
}


class CutGenerator:
    def __init__(
        self,
        aggressiveness: int = 5,
        min_cut: float | None = None,
        max_cut: float | None = None,
        beat_snap_ms: int = 50,
        allowed_transitions: list[str] | None = None,
    ):
        self.aggressiveness = max(1, min(10, aggressiveness))
        table = AGGRESSIVENESS_TABLE[self.aggressiveness]
        self.min_cut = min_cut if min_cut is not None else table["min_cut"]
        self.max_cut = max_cut if max_cut is not None else table["max_cut"]
        self._candidate_filter = table["filter"]
        self.beat_snap_s = beat_snap_ms / 1000.0
        self.allowed_transitions = set(allowed_transitions or ALL_TRANSITIONS)
        if not self.allowed_transitions:
            self.allowed_transitions = {"hard_cut"}

    def generate(self, analysis: AnalysisResult, num_sources: int = 1) -> list[CutPoint]:
        duration = analysis.duration

        # Very short audio: single segment
        if duration < 2.0:
            return [CutPoint(0.0, duration, 0, "hard_cut", 0.0)]

        # No beats: fixed-interval fallback
        if len(analysis.beats) == 0:
            return self._fallback_cuts(duration, num_sources)

        # Build candidate cut times from beats and energy events
        candidates = self._build_candidates(analysis)

        # Enforce min duration: drop candidates too close to previous
        filtered = [candidates[0]] if candidates else [0.0]
        for t in candidates[1:]:
            if t - filtered[-1] >= self.min_cut:
                filtered.append(t)

        # Beat-snap remaining candidates
        beat_arr = np.array(analysis.beats) if analysis.beats else np.array([0.0])
        snapped = []
        for t in filtered:
            diffs = np.abs(beat_arr - t)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= self.beat_snap_s:
                snapped.append(float(beat_arr[min_idx]))
            else:
                snapped.append(t)

        # Enforce max duration: insert cuts where gaps are too large
        final_times = [0.0]
        for t in snapped:
            if t <= final_times[-1]:
                continue
            while t - final_times[-1] > self.max_cut:
                final_times.append(final_times[-1] + self.max_cut)
            final_times.append(t)
        # Ensure we reach the end
        if duration - final_times[-1] > 0.1:
            if duration - final_times[-1] > self.max_cut:
                while duration - final_times[-1] > self.max_cut:
                    final_times.append(final_times[-1] + self.max_cut)
            final_times.append(duration)
        else:
            final_times[-1] = duration

        # Build CutPoints with transition types
        energy_levels = self._classify_energy(analysis)
        beat_strengths = self._classify_beat_strength(analysis)
        drops = self._detect_drops(analysis)
        spikes = self._detect_spikes(analysis)

        cuts = []
        source_idx = 0
        high_energy_counter = 0
        wipe_counter = 0

        for i in range(len(final_times) - 1):
            start = final_times[i]
            end = final_times[i + 1]
            mid = (start + end) / 2.0

            transition = self._decide_transition(
                mid, energy_levels, analysis, beat_strengths,
                drops, spikes, high_energy_counter, wipe_counter,
            )
            t_type, high_energy_counter, wipe_counter = transition
            t_type = self._resolve_transition(t_type)

            t_dur = TRANSITION_DURATIONS[t_type]
            # Clamp transition to half the shorter adjacent segment
            seg_dur = end - start
            if i > 0:
                prev_dur = cuts[-1].end - cuts[-1].start
                max_t = min(seg_dur, prev_dur) / 2.0
                t_dur = min(t_dur, max_t)

            cuts.append(CutPoint(start, end, source_idx, t_type, t_dur))

            if num_sources > 1:
                source_idx = (source_idx + 1) % num_sources

        return cuts

    def _fallback_cuts(self, duration: float, num_sources: int) -> list[CutPoint]:
        cuts = []
        t = 0.0
        source_idx = 0
        while t < duration:
            end = min(t + 3.0, duration)
            cuts.append(CutPoint(t, end, source_idx, "hard_cut", 0.0))
            t = end
            if num_sources > 1:
                source_idx = (source_idx + 1) % num_sources
        return cuts

    def _build_candidates(self, analysis: AnalysisResult) -> list[float]:
        beat_strengths = self._classify_beat_strength(analysis)
        onset_strengths = self._classify_onset_strength(analysis)

        if self._candidate_filter == "strong_beats_only":
            candidates = [b for b in analysis.beats if beat_strengths.get(b) == "strong"]
        elif self._candidate_filter == "strong_beats_and_onsets":
            strong_beats = [b for b in analysis.beats if beat_strengths.get(b) == "strong"]
            strong_onsets = [o for o in analysis.onsets if onset_strengths.get(o, "weak") == "strong"]
            candidates = sorted(set(strong_beats + strong_onsets))
        elif self._candidate_filter == "all_beats_strong_onsets":
            strong_onsets = [o for o in analysis.onsets if onset_strengths.get(o, "weak") == "strong"]
            candidates = sorted(set(analysis.beats + strong_onsets))
        else:  # "all"
            candidates = sorted(set(analysis.beats + analysis.onsets))

        return [t for t in candidates if 0 < t < analysis.duration]

    def _classify_energy(self, analysis: AnalysisResult) -> dict:
        e = analysis.energy_envelope
        return {"p25": float(np.percentile(e, 25)), "p75": float(np.percentile(e, 75))}

    def _get_energy_at_time(self, t: float, analysis: AnalysisResult) -> str:
        idx = np.argmin(np.abs(analysis.energy_times - t))
        val = analysis.energy_envelope[idx]
        levels = self._classify_energy(analysis)
        if val > levels["p75"]:
            return "high"
        elif val >= levels["p25"]:
            return "medium"
        return "low"

    def _classify_beat_strength(self, analysis: AnalysisResult) -> dict[float, str]:
        if not analysis.beats or len(analysis.onset_env) == 0:
            return {}
        # Sample onset strength envelope at each beat position (per spec)
        onset_env = analysis.onset_env
        beat_arr = np.array(analysis.beats)
        # Map beat times to onset_env frame indices
        frame_indices = np.clip(
            (beat_arr / analysis.duration * len(onset_env)).astype(int),
            0, len(onset_env) - 1,
        )
        strengths_at_beats = onset_env[frame_indices]
        median_strength = float(np.median(strengths_at_beats))
        return {
            bt: "strong" if strengths_at_beats[i] > median_strength else "weak"
            for i, bt in enumerate(analysis.beats)
        }

    def _detect_drops(self, analysis: AnalysisResult) -> list[float]:
        return _detect_energy_drops(analysis.energy_envelope, analysis.energy_times)

    def _detect_spikes(self, analysis: AnalysisResult) -> list[float]:
        return _detect_energy_spikes(analysis.energy_envelope, analysis.energy_times)

    def _classify_onset_strength(self, analysis: AnalysisResult) -> dict[float, str]:
        if not analysis.onsets or len(analysis.onset_env) == 0:
            return {}
        onset_arr = np.array(analysis.onsets)
        frame_indices = np.clip(
            (onset_arr / analysis.duration * len(analysis.onset_env)).astype(int),
            0, len(analysis.onset_env) - 1,
        )
        strengths = analysis.onset_env[frame_indices]
        median_strength = float(np.median(strengths))
        return {
            t: "strong" if strengths[i] > median_strength else "weak"
            for i, t in enumerate(analysis.onsets)
        }

    def _resolve_transition(self, preferred: str) -> str:
        if preferred in self.allowed_transitions:
            return preferred
        for fallback in TRANSITION_FALLBACKS.get(preferred, []):
            if fallback in self.allowed_transitions:
                return fallback
        return next(iter(self.allowed_transitions))

    def _decide_transition(
        self, t: float, energy_levels: dict, analysis: AnalysisResult,
        beat_strengths: dict[float, str], drops: list[float], spikes: list[float],
        high_counter: int, wipe_counter: int,
    ) -> tuple[str, int, int]:
        # Check for energy drop nearby
        for d in drops:
            if abs(t - d) < 1.0:
                return "fade_black", high_counter, wipe_counter

        # Check for energy spike nearby
        for s in spikes:
            if abs(t - s) < 0.5:
                return "hard_cut", high_counter, wipe_counter

        energy = self._get_energy_at_time(t, analysis)

        # Find nearest beat and its strength
        nearest_beat = None
        if analysis.beats:
            diffs = [abs(b - t) for b in analysis.beats]
            min_diff = min(diffs)
            if min_diff < 0.5:
                nearest_beat = analysis.beats[diffs.index(min_diff)]

        beat_strength = "weak"
        if nearest_beat and nearest_beat in beat_strengths:
            beat_strength = beat_strengths[nearest_beat]

        # Decision table
        if beat_strength == "strong" and energy == "high":
            high_counter += 1
            if high_counter % 3 == 0:
                wipe_counter += 1
                direction = "wipe_left" if wipe_counter % 2 == 1 else "wipe_right"
                return direction, high_counter, wipe_counter
            return "hard_cut", high_counter, wipe_counter

        if beat_strength == "strong" and energy == "medium":
            return "hard_cut", high_counter, wipe_counter

        if energy == "low":
            # Check if sustained low
            idx = np.argmin(np.abs(analysis.energy_times - t))
            window = min(20, len(analysis.energy_envelope) - idx)
            if window > 5:
                upcoming = analysis.energy_envelope[idx:idx + window]
                p25 = energy_levels["p25"]
                if np.all(upcoming < p25):
                    return "crossfade_slow", high_counter, wipe_counter
            return "crossfade", high_counter, wipe_counter

        return "hard_cut", high_counter, wipe_counter
