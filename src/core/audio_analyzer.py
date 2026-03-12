import logging

import librosa
import numpy as np

from core.models import AnalysisResult

logger = logging.getLogger("autoeditor.audio")

MAX_AUDIO_DURATION = 30 * 60  # 30 minutes in seconds
SAMPLE_RATE = 22050


def detect_energy_drops(energy, times, window_ms=200, ref_ms=500, threshold=0.5):
    """Module-level function for energy drop detection, shared with cut_generator."""
    sr_frames = len(energy) / (times[-1] - times[0]) if len(times) > 1 else 1
    window = max(1, int(window_ms / 1000 * sr_frames))
    ref_window = max(1, int(ref_ms / 1000 * sr_frames))
    drops = []
    for i in range(ref_window, len(energy) - window):
        ref_avg = np.mean(energy[i - ref_window:i])
        if ref_avg == 0:
            continue
        current = np.mean(energy[i:i + window])
        if current < ref_avg * (1 - threshold):
            drops.append(float(times[i]))
    filtered = []
    for t in drops:
        if not filtered or t - filtered[-1] > 0.5:
            filtered.append(t)
    return filtered


def detect_energy_spikes(energy, times, window_ms=200, ref_ms=500, threshold=1.0):
    """Module-level function for energy spike detection, shared with cut_generator."""
    sr_frames = len(energy) / (times[-1] - times[0]) if len(times) > 1 else 1
    window = max(1, int(window_ms / 1000 * sr_frames))
    ref_window = max(1, int(ref_ms / 1000 * sr_frames))
    spikes = []
    for i in range(ref_window, len(energy) - window):
        ref_avg = np.mean(energy[i - ref_window:i])
        if ref_avg == 0:
            continue
        current = np.mean(energy[i:i + window])
        if current > ref_avg * (1 + threshold):
            spikes.append(float(times[i]))
    filtered = []
    for t in spikes:
        if not filtered or t - filtered[-1] > 0.5:
            filtered.append(t)
    return filtered


class AudioAnalyzer:
    def analyze(self, audio_path: str, cancel_check=None) -> AnalysisResult:
        logger.info("Loading audio: %s", audio_path)
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            raise RuntimeError(f"Audio analysis failed: {e}") from e

        duration = librosa.get_duration(y=y, sr=sr)
        if duration > MAX_AUDIO_DURATION:
            raise ValueError(
                f"Audio is {duration/60:.0f} minutes — max is 30 minutes. "
                "Please trim the audio first."
            )

        if cancel_check and cancel_check():
            raise RuntimeError("Analysis cancelled")

        logger.info("Detecting beats and tempo...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        if cancel_check and cancel_check():
            raise RuntimeError("Analysis cancelled")

        logger.info("Detecting onsets...")
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

        if cancel_check and cancel_check():
            raise RuntimeError("Analysis cancelled")

        logger.info("Computing energy envelope...")
        rms = librosa.feature.rms(y=y)[0]
        rms_times = librosa.times_like(rms, sr=sr)

        logger.info("Computing spectral centroids...")
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        return AnalysisResult(
            beats=beat_times,
            onsets=onset_times,
            tempo=tempo_val,
            energy_envelope=rms,
            energy_times=rms_times,
            spectral_centroids=centroids,
            onset_env=onset_env,
            duration=duration,
        )

    def classify_energy(self, energy: np.ndarray) -> list[str]:
        p25 = np.percentile(energy, 25)
        p75 = np.percentile(energy, 75)
        levels = []
        for val in energy:
            if val > p75:
                levels.append("high")
            elif val >= p25:
                levels.append("medium")
            else:
                levels.append("low")
        return levels

    def detect_energy_drops(self, energy, times, **kwargs):
        return detect_energy_drops(energy, times, **kwargs)

    def detect_energy_spikes(self, energy, times, **kwargs):
        return detect_energy_spikes(energy, times, **kwargs)

    def classify_beat_strength(
        self, beat_times: np.ndarray, onset_env: np.ndarray,
        duration: float | None = None,
    ) -> list[str]:
        if len(beat_times) == 0:
            return []
        # Use duration if provided, otherwise estimate from last beat
        dur = duration if duration else float(beat_times[-1])
        if dur <= 0:
            return ["weak"] * len(beat_times)
        # Map beat times to onset_env frame indices
        # Uses same formula as CutGenerator._classify_beat_strength
        indices = np.clip(
            (beat_times / dur * len(onset_env)).astype(int),
            0, len(onset_env) - 1,
        )
        strengths_at_beats = onset_env[indices]
        median_strength = np.median(strengths_at_beats)
        return [
            "strong" if s > median_strength else "weak"
            for s in strengths_at_beats
        ]
