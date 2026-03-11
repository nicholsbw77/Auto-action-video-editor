from dataclasses import dataclass

import numpy as np


@dataclass
class AnalysisResult:
    """Result of audio analysis containing beats, energy, and spectral data."""
    beats: list[float]
    onsets: list[float]
    tempo: float
    energy_envelope: np.ndarray
    energy_times: np.ndarray
    spectral_centroids: np.ndarray
    onset_env: np.ndarray
    duration: float


@dataclass
class CutPoint:
    """A single cut in the edit sequence.
    transition_type/transition_duration describe the transition FROM this segment to the next."""
    start: float
    end: float
    source_index: int
    transition_type: str
    transition_duration: float


@dataclass
class ExportSettings:
    """Encoding and output settings for the final render."""
    output_path: str
    width: int
    height: int
    fps: float
    video_codec: str
    video_crf: int
    audio_codec: str
    audio_bitrate: str
    container: str


@dataclass
class TrimRegion:
    """A region to keep when trimming a video."""
    start: float
    end: float
    label: str
