# Auto Video Editor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python desktop app that auto-edits video based on audio analysis, with PyQt6 GUI, FFmpeg backend, NVIDIA GPU support, and an integrated trim tool.

**Architecture:** 4-layer architecture: GUI (PyQt6 tabs), Core Engine (audio analysis, cut generation, export), FFmpeg Interface (filter graphs, GPU detection, subprocess management), System Layer (config, file I/O). All video processing via FFmpeg subprocess calls. Audio analysis via librosa.

**Tech Stack:** Python 3.10+, PyQt6 >= 6.5, librosa >= 0.10, numpy >= 1.23, soundfile >= 0.12, FFmpeg >= 4.3 (external binary)

**Spec:** `docs/superpowers/specs/2026-03-11-auto-video-editor-design.md`

---

## Chunk 1: Foundation

### Task 1: Project scaffolding and requirements

**Files:**
- Create: `requirements.txt` (project root)
- Create: `pyproject.toml` (project root, for pytest config)
- Create: `.gitignore`
- Create: `src/core/__init__.py`
- Create: `src/gui/__init__.py`
- Create: `src/ffmpeg/__init__.py`
- Create: `src/main.py` (placeholder)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/core src/gui src/ffmpeg tests/core tests/gui tests/ffmpeg
```

- [ ] **Step 2: Create requirements.txt at project root**

Create `requirements.txt`:
```
PyQt6>=6.5,<7.0
librosa>=0.10,<1.0
numpy>=1.23,<2.0
soundfile>=0.12,<1.0
```

- [ ] **Step 3: Create pyproject.toml for pytest path config**

Create `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Create .gitignore**

Create `.gitignore`:
```
__pycache__/
*.pyc
logs/
temp/
config.json
*.egg-info/
dist/
build/
.pytest_cache/
```

- [ ] **Step 5: Create empty __init__.py files and main.py placeholder**

Create empty `src/core/__init__.py`, `src/gui/__init__.py`, `src/ffmpeg/__init__.py`.
Create `src/main.py` with content:
```python
"""Auto Video Editor - Entry Point."""
```

- [ ] **Step 6: Commit scaffolding**

```bash
git add requirements.txt pyproject.toml .gitignore src/ tests/
git commit -m "chore: scaffold project structure with requirements and pytest config"
```

---

### Task 2: Data models

**Files:**
- Create: `src/core/models.py`
- Create: `tests/core/test_models.py`

- [ ] **Step 1: Write tests for data models**

Create `tests/core/test_models.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/core/test_models.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'core.models'`

- [ ] **Step 3: Implement data models**

Create `src/core/models.py`:
```python
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
    onset_env: np.ndarray  # Onset strength envelope for beat strength classification
    duration: float


@dataclass
class CutPoint:
    """A single cut in the edit sequence.
    transition_type/transition_duration describe the transition FROM this segment to the next."""
    start: float
    end: float
    source_index: int
    transition_type: str  # hard_cut, crossfade, crossfade_slow, fade_black, wipe_left, wipe_right
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/core/test_models.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/models.py tests/core/test_models.py
git commit -m "feat: add data model dataclasses"
```

---

### Task 3: Configuration system

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write tests for config**

Create `tests/test_config.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement config module**

Create `src/config.py`:
```python
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULTS = {
    "ffmpeg_path": "",
    "ffprobe_path": "",
    "gpu_detected": False,
    "force_cpu": False,
    "last_output_folder": "",
    "last_input_folder": "",
    "recent_files": [],
    "beat_snap_tolerance_ms": 50,
    "min_cut_duration": 0.5,
    "max_cut_duration": 8.0,
}


class AppConfig:
    def __init__(self, path: Path | str | None = None):
        if path is None:
            path = Path(__file__).parent / "config.json"
        self._path = Path(path)
        self._data: dict = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Config file corrupt or unreadable, using defaults: %s", e)
                self._data = {}
        else:
            self._data = {}
        # Fill missing keys with defaults
        for key, default in DEFAULTS.items():
            if key not in self._data:
                self._data[key] = default

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2), encoding="utf-8"
        )

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in DEFAULTS:
            return self._data.get(name, DEFAULTS[name])
        raise AttributeError(f"No config key: {name}")

    def __setattr__(self, name: str, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name in DEFAULTS:
            self._data[name] = value
        else:
            super().__setattr__(name, value)

    def add_recent_file(self, filepath: str):
        files = self._data.get("recent_files", [])
        if filepath in files:
            files.remove(filepath)
        files.insert(0, filepath)
        self._data["recent_files"] = files[:10]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/test_config.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add JSON config persistence with defaults"
```

---

### Task 4: Logging setup

**Files:**
- Create: `src/core/logger.py`
- Create: `tests/core/test_logger.py`

- [ ] **Step 1: Write test for logging setup**

Create `tests/core/test_logger.py`:
```python
import logging
from pathlib import Path
from core.logger import setup_logging


def test_setup_creates_log_file(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)
    logger = logging.getLogger("autoeditor")
    logger.info("test message")
    log_file = log_dir / "autoeditor.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "test message" in content


def test_setup_returns_logger(tmp_path):
    logger = setup_logging(tmp_path / "logs")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "autoeditor"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/core/test_logger.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement logger**

Create `src/core/logger.py`:
```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("autoeditor")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = RotatingFileHandler(
            log_dir / "autoeditor.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(file_handler)

    return logger
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/core/test_logger.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/logger.py tests/core/test_logger.py
git commit -m "feat: add rotating file logger"
```

---

### Task 5: Temp file manager

**Files:**
- Create: `src/core/temp_manager.py`
- Create: `tests/core/test_temp_manager.py`

- [ ] **Step 1: Write tests for temp manager**

Create `tests/core/test_temp_manager.py`:
```python
import time
from pathlib import Path
from core.temp_manager import TempManager


class TestTempManager:
    def test_create_session_dir(self, tmp_path):
        tm = TempManager(tmp_path / "temp")
        session_dir = tm.create_session()
        assert session_dir.exists()
        assert session_dir.parent == tmp_path / "temp"

    def test_cleanup_session(self, tmp_path):
        tm = TempManager(tmp_path / "temp")
        session_dir = tm.create_session()
        # Create a temp file inside
        (session_dir / "test.mp4").write_text("data")
        tm.cleanup_session()
        assert not session_dir.exists()

    def test_cleanup_stale_sessions(self, tmp_path):
        temp_root = tmp_path / "temp"
        temp_root.mkdir()
        # Create a "stale" directory with old mtime
        stale_dir = temp_root / "stale_session"
        stale_dir.mkdir()
        (stale_dir / "file.tmp").write_text("old")
        # Set mtime to 25 hours ago
        old_time = time.time() - (25 * 3600)
        import os
        os.utime(stale_dir, (old_time, old_time))

        tm = TempManager(temp_root)
        tm.cleanup_stale(max_age_hours=24)
        assert not stale_dir.exists()

    def test_keep_recent_sessions(self, tmp_path):
        temp_root = tmp_path / "temp"
        temp_root.mkdir()
        recent_dir = temp_root / "recent_session"
        recent_dir.mkdir()
        (recent_dir / "file.tmp").write_text("new")

        tm = TempManager(temp_root)
        tm.cleanup_stale(max_age_hours=24)
        assert recent_dir.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/core/test_temp_manager.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement temp manager**

Create `src/core/temp_manager.py`:
```python
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("autoeditor.temp")


class TempManager:
    def __init__(self, temp_root: Path):
        self._root = temp_root
        self._session_dir: Path | None = None

    def create_session(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._session_dir = self._root / session_name
        self._session_dir.mkdir()
        logger.info("Created temp session: %s", self._session_dir)
        return self._session_dir

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    def cleanup_session(self):
        if self._session_dir and self._session_dir.exists():
            shutil.rmtree(self._session_dir, ignore_errors=True)
            logger.info("Cleaned up temp session: %s", self._session_dir)
            self._session_dir = None

    def cleanup_stale(self, max_age_hours: int = 24):
        if not self._root.exists():
            return
        cutoff = time.time() - (max_age_hours * 3600)
        for entry in self._root.iterdir():
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                logger.info("Removed stale temp dir: %s", entry)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/core/test_temp_manager.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/temp_manager.py tests/core/test_temp_manager.py
git commit -m "feat: add session-based temp file manager with stale cleanup"
```

---

## Chunk 2: FFmpeg Layer

### Task 6: FFmpeg binary detector

**Files:**
- Create: `src/ffmpeg/detector.py`
- Create: `tests/ffmpeg/test_detector.py`

- [ ] **Step 1: Write tests for FFmpeg detector**

Create `tests/ffmpeg/test_detector.py`:
```python
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from ffmpeg.detector import FFmpegDetector


class TestFFmpegDetector:
    def test_find_on_path(self):
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda name: f"/usr/bin/{name}" if name in ("ffmpeg", "ffprobe") else None
            det = FFmpegDetector()
            assert det.find_ffmpeg() == "/usr/bin/ffmpeg"
            assert det.find_ffprobe() == "/usr/bin/ffprobe"

    def test_find_in_common_locations(self, tmp_path):
        ffmpeg_bin = tmp_path / "ffmpeg.exe"
        ffmpeg_bin.write_text("fake")
        with patch("shutil.which", return_value=None):
            det = FFmpegDetector(extra_search_paths=[str(tmp_path)])
            result = det.find_ffmpeg()
            assert result is not None

    def test_not_found(self):
        with patch("shutil.which", return_value=None):
            det = FFmpegDetector(extra_search_paths=[])
            assert det.find_ffmpeg() is None

    def test_validate_version_ok(self):
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 5.1.2 Copyright"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            det = FFmpegDetector()
            assert det.validate_version("/usr/bin/ffmpeg") is True

    def test_validate_version_too_old(self):
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 3.4.1 Copyright"
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            det = FFmpegDetector()
            assert det.validate_version("/usr/bin/ffmpeg") is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/ffmpeg/test_detector.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement detector**

Create `src/ffmpeg/detector.py`:
```python
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("autoeditor.ffmpeg")

COMMON_PATHS_WIN = [
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
]

MIN_VERSION = (4, 3, 0)


class FFmpegDetector:
    def __init__(self, extra_search_paths: list[str] | None = None):
        self._extra_paths = extra_search_paths if extra_search_paths is not None else COMMON_PATHS_WIN

    def find_ffmpeg(self) -> str | None:
        return self._find_binary("ffmpeg")

    def find_ffprobe(self) -> str | None:
        return self._find_binary("ffprobe")

    def _find_binary(self, name: str) -> str | None:
        # Check PATH first
        result = shutil.which(name)
        if result:
            logger.info("Found %s on PATH: %s", name, result)
            return result
        # Check common locations
        for search_path in self._extra_paths:
            for ext in ("", ".exe"):
                candidate = Path(search_path) / f"{name}{ext}"
                if candidate.is_file():
                    logger.info("Found %s at: %s", name, candidate)
                    return str(candidate)
        logger.warning("%s not found", name)
        return None

    def validate_version(self, ffmpeg_path: str) -> bool:
        try:
            result = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True, text=True, timeout=10,
            )
            match = re.search(r"ffmpeg version (\d+)\.(\d+)\.?(\d*)", result.stdout)
            if not match:
                logger.warning("Could not parse FFmpeg version from: %s", result.stdout[:100])
                return False
            major, minor = int(match.group(1)), int(match.group(2))
            patch_v = int(match.group(3)) if match.group(3) else 0
            version = (major, minor, patch_v)
            if version >= MIN_VERSION:
                logger.info("FFmpeg version %s meets minimum %s", version, MIN_VERSION)
                return True
            logger.warning("FFmpeg version %s below minimum %s", version, MIN_VERSION)
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.error("Failed to check FFmpeg version: %s", e)
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/ffmpeg/test_detector.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ffmpeg/detector.py tests/ffmpeg/test_detector.py
git commit -m "feat: add FFmpeg/ffprobe binary detection and version validation"
```

---

### Task 7: GPU detection

**Files:**
- Create: `src/ffmpeg/gpu.py`
- Create: `tests/ffmpeg/test_gpu.py`

- [ ] **Step 1: Write tests for GPU detection**

Create `tests/ffmpeg/test_gpu.py`:
```python
from unittest.mock import patch, MagicMock
import subprocess
from ffmpeg.gpu import GPUDetector


class TestGPUDetector:
    def test_nvidia_gpu_present(self):
        smi_result = MagicMock(returncode=0)
        enc_result = MagicMock(returncode=0, stdout="h264_nvenc")
        with patch("subprocess.run", side_effect=[smi_result, enc_result]):
            det = GPUDetector(ffmpeg_path="ffmpeg")
            assert det.detect_nvenc() is True

    def test_no_nvidia_gpu(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            det = GPUDetector(ffmpeg_path="ffmpeg")
            assert det.detect_nvenc() is False

    def test_nvidia_gpu_no_nvenc(self):
        smi_result = MagicMock(returncode=0)
        enc_result = MagicMock(returncode=0, stdout="libx264")
        with patch("subprocess.run", side_effect=[smi_result, enc_result]):
            det = GPUDetector(ffmpeg_path="ffmpeg")
            assert det.detect_nvenc() is False

    def test_get_encoder_gpu(self):
        det = GPUDetector(ffmpeg_path="ffmpeg")
        det._nvenc_available = True
        assert det.get_video_encoder(force_cpu=False) == "h264_nvenc"
        assert det.get_crf(force_cpu=False) == 20

    def test_get_encoder_cpu(self):
        det = GPUDetector(ffmpeg_path="ffmpeg")
        det._nvenc_available = False
        assert det.get_video_encoder(force_cpu=False) == "libx264"
        assert det.get_crf(force_cpu=False) == 18

    def test_force_cpu_overrides_gpu(self):
        det = GPUDetector(ffmpeg_path="ffmpeg")
        det._nvenc_available = True
        assert det.get_video_encoder(force_cpu=True) == "libx264"
        assert det.get_crf(force_cpu=True) == 18
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/ffmpeg/test_gpu.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement GPU detector**

Create `src/ffmpeg/gpu.py`:
```python
import logging
import subprocess

logger = logging.getLogger("autoeditor.ffmpeg")


class GPUDetector:
    def __init__(self, ffmpeg_path: str):
        self._ffmpeg_path = ffmpeg_path
        self._nvenc_available = False

    def detect_nvenc(self) -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                logger.info("nvidia-smi returned non-zero — no NVIDIA GPU")
                self._nvenc_available = False
                return False
        except (FileNotFoundError, OSError):
            logger.info("nvidia-smi not found — no NVIDIA GPU")
            self._nvenc_available = False
            return False

        try:
            result = subprocess.run(
                [self._ffmpeg_path, "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            if "h264_nvenc" in result.stdout:
                logger.info("NVENC encoder available")
                self._nvenc_available = True
                return True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as e:
            logger.warning("Failed to check NVENC support: %s", e)

        logger.info("NVENC not available, will use CPU encoding")
        self._nvenc_available = False
        return False

    def get_video_encoder(self, force_cpu: bool = False) -> str:
        if force_cpu or not self._nvenc_available:
            return "libx264"
        return "h264_nvenc"

    def get_crf(self, force_cpu: bool = False) -> int:
        if force_cpu or not self._nvenc_available:
            return 18  # libx264 CRF
        return 20  # h264_nvenc CQ

    def get_encoder_preset(self, force_cpu: bool = False) -> str:
        if force_cpu or not self._nvenc_available:
            return "medium"
        return "p4"

    @property
    def nvenc_available(self) -> bool:
        return self._nvenc_available
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/ffmpeg/test_gpu.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ffmpeg/gpu.py tests/ffmpeg/test_gpu.py
git commit -m "feat: add NVIDIA GPU and NVENC detection"
```

---

### Task 8: FFmpeg subprocess runner

**Files:**
- Create: `src/ffmpeg/runner.py`
- Create: `tests/ffmpeg/test_runner.py`

- [ ] **Step 1: Write tests for runner**

Create `tests/ffmpeg/test_runner.py`:
```python
import re
from unittest.mock import patch, MagicMock, PropertyMock
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/ffmpeg/test_runner.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement runner**

Create `src/ffmpeg/runner.py`:
```python
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Callable

logger = logging.getLogger("autoeditor.ffmpeg")

TIME_PATTERN = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})")


def parse_progress_time(line: str) -> float | None:
    match = TIME_PATTERN.search(line)
    if not match:
        return None
    h, m, s, cs = int(match[1]), int(match[2]), int(match[3]), int(match[4])
    return h * 3600 + m * 60 + s + cs / 100.0


class FFmpegRunner:
    def __init__(self, ffmpeg_path: str, ffprobe_path: str | None = None):
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path or "ffprobe"

    def build_command(
        self,
        inputs: list[str],
        output: str,
        filter_complex: str | None = None,
        maps: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = [self._ffmpeg, "-y"]
        for inp in inputs:
            cmd.extend(["-i", inp])
        if filter_complex:
            cmd.extend(["-filter_complex", filter_complex])
        if maps:
            for m in maps:
                cmd.extend(["-map", m])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(output)
        return cmd

    def run(
        self,
        cmd: list[str],
        total_duration: float | None = None,
        progress_callback: Callable[[float], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> subprocess.CompletedProcess:
        logger.info("Running: %s", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr_lines = []
        try:
            for line in iter(process.stderr.readline, ""):
                stderr_lines.append(line)
                if total_duration and progress_callback:
                    t = parse_progress_time(line)
                    if t is not None:
                        progress_callback(min(t / total_duration, 1.0))
                if cancel_check and cancel_check():
                    process.stdin.write("q\n")
                    process.stdin.flush()
                    process.wait(timeout=10)
                    raise RuntimeError("FFmpeg cancelled by user")
        except BrokenPipeError:
            pass
        process.wait()
        stderr_text = "".join(stderr_lines)
        if process.returncode != 0:
            logger.error("FFmpeg failed (rc=%d): %s", process.returncode, stderr_text[-500:])
        return subprocess.CompletedProcess(
            args=cmd, returncode=process.returncode,
            stdout="", stderr=stderr_text,
        )

    def probe(self, filepath: str) -> dict:
        cmd = [
            self._ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            filepath,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {filepath}: {result.stderr[:200]}")
        return json.loads(result.stdout)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/ffmpeg/test_runner.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ffmpeg/runner.py tests/ffmpeg/test_runner.py
git commit -m "feat: add FFmpeg subprocess runner with progress parsing and cancellation"
```

---

## Chunk 3: Audio Analysis & Cut Generation

### Task 9: Audio analyzer

**Files:**
- Create: `src/core/audio_analyzer.py`
- Create: `tests/core/test_audio_analyzer.py`

- [ ] **Step 1: Write tests for audio analyzer**

Create `tests/core/test_audio_analyzer.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/core/test_audio_analyzer.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement audio analyzer**

Create `src/core/audio_analyzer.py`:
```python
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
    ) -> list[str]:
        if len(beat_times) == 0:
            return []
        # Sample onset strength at each beat position
        indices = np.clip(
            (beat_times * len(onset_env) / beat_times[-1]).astype(int),
            0, len(onset_env) - 1,
        ) if beat_times[-1] > 0 else np.zeros(len(beat_times), dtype=int)
        strengths_at_beats = onset_env[indices]
        median_strength = np.median(strengths_at_beats)
        return [
            "strong" if s > median_strength else "weak"
            for s in strengths_at_beats
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/core/test_audio_analyzer.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/audio_analyzer.py tests/core/test_audio_analyzer.py
git commit -m "feat: add librosa-based audio analyzer with energy/beat classification"
```

---

### Task 10: Cut generator

**Files:**
- Create: `src/core/cut_generator.py`
- Create: `tests/core/test_cut_generator.py`

- [ ] **Step 1: Write tests for cut generator**

Create `tests/core/test_cut_generator.py`:
```python
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
        # With high energy throughout, we should get some wipes
        analysis = AnalysisResult(
            beats=np.linspace(0.5, 29.5, 30).tolist(),
            onsets=np.linspace(0.5, 29.5, 30).tolist(),
            tempo=120.0,
            energy_envelope=np.ones(100) * 0.95,  # all high energy
            energy_times=np.linspace(0, 30, 100),
            spectral_centroids=np.ones(100) * 2000,
            onset_env=np.ones(100) * 0.8,
            duration=30.0,
        )
        gen = CutGenerator(min_cut=0.5, max_cut=8.0, beat_snap_ms=50)
        cuts = gen.generate(analysis, num_sources=1)
        wipes = [c for c in cuts if c.transition_type in ("wipe_left", "wipe_right")]
        # Should have at least one wipe in 30 cuts
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/core/test_cut_generator.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement cut generator**

Create `src/core/cut_generator.py`:
```python
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


class CutGenerator:
    def __init__(self, min_cut: float = 0.5, max_cut: float = 8.0, beat_snap_ms: int = 50):
        self.min_cut = min_cut
        self.max_cut = max_cut
        self.beat_snap_s = beat_snap_ms / 1000.0

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
        all_times = sorted(set(analysis.beats + analysis.onsets))
        return [t for t in all_times if 0 < t < analysis.duration]

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/core/test_cut_generator.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/cut_generator.py tests/core/test_cut_generator.py
git commit -m "feat: add audio-driven cut generator with transition logic"
```

---

## Chunk 4: Filter Graph & Export

### Task 11: Filter graph builder

**Files:**
- Create: `src/ffmpeg/filter_graph.py`
- Create: `tests/ffmpeg/test_filter_graph.py`

- [ ] **Step 1: Write tests for filter graph builder**

Create `tests/ffmpeg/test_filter_graph.py`:
```python
import pytest
from core.models import CutPoint
from ffmpeg.filter_graph import FilterGraphBuilder


def make_cuts():
    return [
        CutPoint(2.0, 5.0, 0, "hard_cut", 0.0),
        CutPoint(12.0, 16.0, 0, "crossfade", 0.3),
        CutPoint(22.0, 25.0, 0, "wipe_left", 0.2),
    ]


class TestFilterGraphBuilder:
    def test_group_by_transitions(self):
        cuts = make_cuts()
        builder = FilterGraphBuilder()
        groups = builder.group_cuts(cuts)
        # First cut has hard_cut transition -> boundary
        # Cuts 1-2 have non-hard transitions -> one group
        assert len(groups) >= 1

    def test_build_xfade_graph_single_source(self):
        cuts = [
            CutPoint(2.0, 5.0, 0, "crossfade", 0.3),
            CutPoint(12.0, 16.0, 0, "crossfade", 0.3),
            CutPoint(22.0, 25.0, 0, "crossfade", 0.3),
        ]
        builder = FilterGraphBuilder()
        graph = builder.build_xfade_graph(cuts, source_mapping={0: 0})
        assert "xfade" in graph
        assert "trim" in graph
        assert "setpts" in graph

    def test_offset_calculation(self):
        builder = FilterGraphBuilder()
        # Segment 0: 3s, transition 0: 0.3s => offset_0 = 2.7
        # Segment 1: 4s, transition 1: 0.2s => offset_1 = 2.7 + 4.0 - 0.2 = 6.5
        cuts = [
            CutPoint(2.0, 5.0, 0, "crossfade", 0.3),
            CutPoint(12.0, 16.0, 0, "wipe_left", 0.2),
            CutPoint(22.0, 25.0, 0, "hard_cut", 0.0),  # last, no xfade after
        ]
        offsets = builder.calculate_offsets(cuts)
        assert abs(offsets[0] - 2.7) < 0.001
        assert abs(offsets[1] - 6.5) < 0.001

    def test_transition_mapping(self):
        builder = FilterGraphBuilder()
        assert builder.xfade_name("crossfade") == "fade"
        assert builder.xfade_name("crossfade_slow") == "fade"
        assert builder.xfade_name("fade_black") == "fadeblack"
        assert builder.xfade_name("wipe_left") == "wipeleft"
        assert builder.xfade_name("wipe_right") == "wiperight"

    def test_batch_splitting(self):
        # 25 cuts with non-hard transitions -> should split at 20
        cuts = [
            CutPoint(float(i), float(i + 1), 0, "crossfade", 0.3)
            for i in range(25)
        ]
        builder = FilterGraphBuilder()
        batches = builder.split_batches(cuts, max_per_batch=20)
        assert len(batches) == 2
        assert len(batches[0]) == 20
        assert len(batches[1]) == 5

    def test_multi_source_mapping(self):
        cuts = [
            CutPoint(5.0, 8.0, 0, "crossfade", 0.3),
            CutPoint(2.0, 6.0, 2, "crossfade", 0.3),
            CutPoint(15.0, 18.0, 0, "hard_cut", 0.0),
        ]
        builder = FilterGraphBuilder()
        mapping = builder.build_source_mapping(cuts)
        # Should map original indices 0, 2 to batch-local 0, 1
        assert 0 in mapping
        assert 2 in mapping
        assert mapping[0] == 0
        assert mapping[2] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/ffmpeg/test_filter_graph.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement filter graph builder**

Create `src/ffmpeg/filter_graph.py`:
```python
import logging
from core.models import CutPoint

logger = logging.getLogger("autoeditor.ffmpeg")

XFADE_MAP = {
    "crossfade": "fade",
    "crossfade_slow": "fade",
    "fade_black": "fadeblack",
    "wipe_left": "wipeleft",
    "wipe_right": "wiperight",
}


class FilterGraphBuilder:
    def xfade_name(self, transition_type: str) -> str:
        return XFADE_MAP.get(transition_type, "fade")

    def group_cuts(self, cuts: list[CutPoint]) -> list[list[CutPoint]]:
        """Group cuts by transition boundaries.
        Convention: CutPoint.transition_type = transition FROM this segment to the next.
        A hard_cut on cuts[i] means a hard cut AFTER segment i, so segment i+1 starts a new group."""
        if not cuts:
            return []
        groups = []
        current_group = [cuts[0]]
        for i in range(1, len(cuts)):
            # Check if the PREVIOUS segment has a hard_cut (transition after it)
            if cuts[i - 1].transition_type == "hard_cut":
                groups.append(current_group)
                current_group = [cuts[i]]
            else:
                current_group.append(cuts[i])
        groups.append(current_group)
        return groups

    def split_batches(self, cuts: list[CutPoint], max_per_batch: int = 20) -> list[list[CutPoint]]:
        if len(cuts) <= max_per_batch:
            return [cuts]
        batches = []
        for i in range(0, len(cuts), max_per_batch):
            batches.append(cuts[i:i + max_per_batch])
        return batches

    def build_source_mapping(self, cuts: list[CutPoint]) -> dict[int, int]:
        unique_sources = sorted(set(c.source_index for c in cuts))
        return {orig: local for local, orig in enumerate(unique_sources)}

    def calculate_offsets(self, cuts: list[CutPoint]) -> list[float]:
        if len(cuts) < 2:
            return []
        offsets = []
        seg0_dur = cuts[0].end - cuts[0].start
        offset = seg0_dur - cuts[0].transition_duration
        offsets.append(offset)
        for i in range(1, len(cuts) - 1):
            seg_dur = cuts[i].end - cuts[i].start
            offset = offset + seg_dur - cuts[i].transition_duration
            offsets.append(offset)
        return offsets

    def build_xfade_graph(self, cuts: list[CutPoint], source_mapping: dict[int, int]) -> str:
        if len(cuts) < 2:
            c = cuts[0]
            local_idx = source_mapping[c.source_index]
            return f"[{local_idx}:v]trim=start={c.start}:end={c.end},setpts=PTS-STARTPTS[vout]"

        lines = []
        # Trim each segment
        for i, c in enumerate(cuts):
            local_idx = source_mapping[c.source_index]
            lines.append(f"[{local_idx}:v]trim=start={c.start}:end={c.end},setpts=PTS-STARTPTS[v{i}]")

        # Chain xfade transitions
        offsets = self.calculate_offsets(cuts)
        prev_label = "[v0]"
        for i in range(len(cuts) - 1):
            xfade = self.xfade_name(cuts[i].transition_type)
            dur = cuts[i].transition_duration
            offset = offsets[i]
            if i == len(cuts) - 2:
                out_label = "[vout]"
            else:
                out_label = f"[vt{i + 1}]"
            lines.append(
                f"{prev_label}[v{i + 1}]xfade=transition={xfade}:duration={dur}:offset={offset}{out_label}"
            )
            prev_label = out_label

        return ";\n".join(lines)

    def build_audio_graph(
        self, cuts: list[CutPoint], source_mapping: dict[int, int], use_separate_audio: bool,
    ) -> str | None:
        if use_separate_audio:
            return None  # Audio is laid down as-is from the separate file

        if len(cuts) < 2:
            c = cuts[0]
            local_idx = source_mapping[c.source_index]
            return f"[{local_idx}:a]atrim=start={c.start}:end={c.end},asetpts=PTS-STARTPTS[aout]"

        lines = []
        for i, c in enumerate(cuts):
            local_idx = source_mapping[c.source_index]
            # Compute overlap extensions per spec:
            # Outgoing: extend atrim_end by outgoing transition duration
            # Incoming: start atrim earlier by incoming transition duration
            out_t = c.transition_duration if i < len(cuts) - 1 else 0.0
            in_t = cuts[i - 1].transition_duration if i > 0 else 0.0
            # For hard cuts, use 30ms overlap
            if out_t == 0.0 and i < len(cuts) - 1:
                out_t = 0.03
            if in_t == 0.0 and i > 0:
                in_t = 0.03
            atrim_start = max(0, c.start - in_t)
            atrim_end = c.end + out_t
            lines.append(
                f"[{local_idx}:a]atrim=start={atrim_start}:end={atrim_end},asetpts=PTS-STARTPTS[a{i}]"
            )

        # Chain acrossfade
        prev_label = "[a0]"
        for i in range(len(cuts) - 1):
            dur = cuts[i].transition_duration
            if dur == 0.0:
                dur = 0.03  # 30ms for hard cuts
            if i == len(cuts) - 2:
                out_label = "[aout]"
            else:
                out_label = f"[at{i + 1}]"
            lines.append(f"{prev_label}[a{i + 1}]acrossfade=d={dur}{out_label}")
            prev_label = out_label

        return ";\n".join(lines)

    def build_concat_file(self, file_paths: list[str]) -> str:
        lines = []
        for path in file_paths:
            escaped = path.replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/ffmpeg/test_filter_graph.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ffmpeg/filter_graph.py tests/ffmpeg/test_filter_graph.py
git commit -m "feat: add filter graph builder with xfade, batching, and audio graphs"
```

---

### Task 12: Export manager

**Files:**
- Create: `src/core/export_manager.py`
- Create: `tests/core/test_export_manager.py`

- [ ] **Step 1: Write tests for export manager**

Create `tests/core/test_export_manager.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/core/test_export_manager.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement export manager**

Create `src/core/export_manager.py`:
```python
import logging
import shutil
from fractions import Fraction
from pathlib import Path

from core.models import ExportSettings

logger = logging.getLogger("autoeditor.export")

MAX_WIDTH = 1920
MAX_HEIGHT = 1080
MAX_FPS = 60.0
BITRATE_MBPS_1080P60 = 20  # Mbps for estimation


class ExportManager:
    def resolve_settings(
        self,
        probe_data: list[dict],
        output_path: str,
        video_codec: str,
        video_crf: int,
    ) -> ExportSettings:
        max_w, max_h, max_fps = 0, 0, 0.0

        for probe in probe_data:
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "video":
                    w = int(stream.get("width", 0))
                    h = int(stream.get("height", 0))
                    fps = self._parse_framerate(stream.get("r_frame_rate", "30/1"))
                    max_w = max(max_w, w)
                    max_h = max(max_h, h)
                    max_fps = max(max_fps, fps)

        # Cap at 1080p60
        if max_w > MAX_WIDTH or max_h > MAX_HEIGHT:
            scale = min(MAX_WIDTH / max_w, MAX_HEIGHT / max_h)
            max_w = int(max_w * scale)
            max_h = int(max_h * scale)
            # Ensure even dimensions
            max_w = max_w - (max_w % 2)
            max_h = max_h - (max_h % 2)
        max_fps = min(max_fps, MAX_FPS)

        return ExportSettings(
            output_path=output_path,
            width=max_w, height=max_h, fps=max_fps,
            video_codec=video_codec, video_crf=video_crf,
            audio_codec="aac", audio_bitrate="192k", container="mp4",
        )

    def _parse_framerate(self, rate_str: str) -> float:
        try:
            frac = Fraction(rate_str)
            return float(frac)
        except (ValueError, ZeroDivisionError):
            return 30.0

    def is_vfr(self, video_stream: dict) -> bool:
        r_fps = self._parse_framerate(video_stream.get("r_frame_rate", "30/1"))
        avg_fps = self._parse_framerate(video_stream.get("avg_frame_rate", "30/1"))
        if r_fps == 0:
            return False
        diff_pct = abs(r_fps - avg_fps) / r_fps
        return diff_pct > 0.05  # 5% threshold

    def estimate_disk_usage(
        self,
        duration_s: float,
        width: int, height: int, fps: float,
        num_clips: int = 1,
        has_vfr: bool = False,
    ) -> int:
        # Scale bitrate from 1080p60 baseline
        pixel_ratio = (width * height) / (1920 * 1080)
        fps_ratio = fps / 60.0
        bitrate = BITRATE_MBPS_1080P60 * pixel_ratio * fps_ratio  # Mbps
        output_bytes = int(bitrate * 1_000_000 / 8 * duration_s)

        # Intermediates: normalization + batch files ≈ 2x output for multi-clip
        intermediate_factor = 2.0 if num_clips > 1 else 1.0
        if has_vfr:
            intermediate_factor += 1.0

        total = int(output_bytes * (1 + intermediate_factor))
        return total

    def check_disk_space(self, output_path: str, required_bytes: int) -> bool:
        drive = Path(output_path).resolve().anchor
        usage = shutil.disk_usage(drive)
        if usage.free < required_bytes:
            logger.warning(
                "Low disk space: need %d MB, have %d MB",
                required_bytes // (1024 * 1024),
                usage.free // (1024 * 1024),
            )
            return False
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/core/test_export_manager.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/export_manager.py tests/core/test_export_manager.py
git commit -m "feat: add export manager with settings resolution, VFR detection, disk estimation"
```

---

## Chunk 5: GUI

### Task 13: Shared widgets

**Files:**
- Create: `src/gui/widgets.py`
- Create: `tests/gui/test_widgets.py`

- [ ] **Step 1: Write tests for widgets**

Create `tests/gui/test_widgets.py`:
```python
import pytest
from gui.widgets import parse_timecode, format_timecode


class TestTimecode:
    def test_parse_timecode(self):
        assert parse_timecode("00:01:30.50") == 90.5

    def test_parse_timecode_hours(self):
        assert parse_timecode("01:00:00.00") == 3600.0

    def test_parse_timecode_zero(self):
        assert parse_timecode("00:00:00.00") == 0.0

    def test_format_timecode(self):
        assert format_timecode(90.5) == "00:01:30.50"

    def test_format_timecode_hours(self):
        assert format_timecode(3661.25) == "01:01:01.25"

    def test_roundtrip(self):
        for val in [0.0, 1.5, 90.5, 3600.0, 7261.99]:
            assert abs(parse_timecode(format_timecode(val)) - val) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/gui/test_widgets.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement shared widgets**

Create `src/gui/widgets.py`:
```python
import re
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QProgressBar, QListWidget, QListWidgetItem,
)
from PyQt6.QtCore import pyqtSignal

TIMECODE_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{2})")


def parse_timecode(tc: str) -> float:
    m = TIMECODE_RE.match(tc)
    if not m:
        raise ValueError(f"Invalid timecode: {tc}")
    h, mn, s, cs = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return h * 3600 + mn * 60 + s + cs / 100.0


def format_timecode(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    seconds %= 60
    s = int(seconds)
    cs = int(round((seconds - s) * 100))
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


class TimecodeInput(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if label:
            layout.addWidget(QLabel(label))
        self._edit = QLineEdit("00:00:00.00")
        self._edit.setInputMask("99:99:99.99")
        self._edit.textChanged.connect(self._on_changed)
        layout.addWidget(self._edit)

    def _on_changed(self, text: str):
        try:
            val = parse_timecode(text)
            self.value_changed.emit(val)
        except ValueError:
            pass

    def get_value(self) -> float:
        try:
            return parse_timecode(self._edit.text())
        except ValueError:
            return 0.0

    def set_value(self, seconds: float):
        self._edit.setText(format_timecode(seconds))


class FileListWidget(QListWidget):
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ts"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._paths: list[str] = []

    def add_file(self, path: str) -> bool:
        import os
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.VIDEO_EXTENSIONS:
            return False
        if path in self._paths:
            return False
        self._paths.append(path)
        self.addItem(QListWidgetItem(os.path.basename(path)))
        return True

    def get_paths(self) -> list[str]:
        return list(self._paths)

    def clear_all(self):
        self._paths.clear()
        self.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        import os
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and not os.path.isdir(path):
                self.add_file(path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/gui/test_widgets.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gui/widgets.py tests/gui/test_widgets.py
git commit -m "feat: add shared GUI widgets (timecode, file list with drag-drop)"
```

---

### Task 14: QThread workers

**Files:**
- Create: `src/gui/workers.py`
- Create: `tests/gui/test_workers.py`

- [ ] **Step 1: Write tests for workers**

Create `tests/gui/test_workers.py`:
```python
import pytest
from unittest.mock import MagicMock
from gui.workers import AnalysisWorker, ExportWorker


class TestAnalysisWorker:
    def test_creates_with_audio_path(self):
        worker = AnalysisWorker("test.wav")
        assert worker._audio_path == "test.wav"
        assert worker._cancelled is False

    def test_cancel_sets_flag(self):
        worker = AnalysisWorker("test.wav")
        worker.cancel()
        assert worker._cancelled is True


class TestExportWorker:
    def test_creates_with_params(self):
        worker = ExportWorker(
            cuts=[], ffmpeg_runner=MagicMock(),
            filter_builder=MagicMock(), settings=MagicMock(),
            source_paths=["a.mp4"], audio_path="audio.wav",
            temp_dir="/tmp/session", use_separate_audio=True,
        )
        assert worker._use_separate_audio is True

    def test_cancel_sets_flag(self):
        worker = ExportWorker(
            cuts=[], ffmpeg_runner=MagicMock(),
            filter_builder=MagicMock(), settings=MagicMock(),
            source_paths=[], audio_path=None,
            temp_dir="/tmp", use_separate_audio=False,
        )
        worker.cancel()
        assert worker._cancelled is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest ../tests/gui/test_workers.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement workers**

Create `src/gui/workers.py`:
```python
import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from core.audio_analyzer import AudioAnalyzer
from core.models import AnalysisResult, CutPoint, ExportSettings
from ffmpeg.filter_graph import FilterGraphBuilder
from ffmpeg.runner import FFmpegRunner

logger = logging.getLogger("autoeditor.workers")


class AnalysisWorker(QThread):
    progress = pyqtSignal(str)  # status message
    finished = pyqtSignal(object)  # AnalysisResult or None
    error = pyqtSignal(str)

    def __init__(self, audio_path: str, parent=None):
        super().__init__(parent)
        self._audio_path = audio_path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit("Analyzing audio...")
            analyzer = AudioAnalyzer()
            result = analyzer.analyze(
                self._audio_path,
                cancel_check=lambda: self._cancelled,
            )
            if self._cancelled:
                self.finished.emit(None)
                return
            self.progress.emit("Analysis complete")
            self.finished.emit(result)
        except Exception as e:
            logger.exception("Analysis failed")
            self.error.emit(str(e))


class ExportWorker(QThread):
    progress = pyqtSignal(float)  # 0.0 to 1.0
    stage = pyqtSignal(str)  # stage description
    finished = pyqtSignal(bool)  # success
    error = pyqtSignal(str)

    def __init__(
        self,
        cuts: list[CutPoint],
        ffmpeg_runner: FFmpegRunner,
        filter_builder: FilterGraphBuilder,
        settings: ExportSettings,
        source_paths: list[str],
        audio_path: str | None,
        temp_dir: str,
        use_separate_audio: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._cuts = cuts
        self._runner = ffmpeg_runner
        self._filter_builder = filter_builder
        self._settings = settings
        self._source_paths = source_paths
        self._audio_path = audio_path
        self._temp_dir = temp_dir
        self._use_separate_audio = use_separate_audio
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.stage.emit("Grouping segments...")
            groups = self._filter_builder.group_cuts(self._cuts)
            intermediate_files = []

            total_groups = len(groups)
            for gi, group in enumerate(groups):
                if self._cancelled:
                    self.finished.emit(False)
                    return

                self.stage.emit(f"Encoding group {gi + 1}/{total_groups}...")
                batches = self._filter_builder.split_batches(group)

                for bi, batch in enumerate(batches):
                    if self._cancelled:
                        self.finished.emit(False)
                        return

                    source_mapping = self._filter_builder.build_source_mapping(batch)
                    video_graph = self._filter_builder.build_xfade_graph(batch, source_mapping)
                    audio_graph = self._filter_builder.build_audio_graph(
                        batch, source_mapping, self._use_separate_audio,
                    )

                    # Build combined filter
                    combined = video_graph
                    maps = ["[vout]"]
                    if audio_graph:
                        combined += ";\n" + audio_graph
                        maps.append("[aout]")

                    # Determine input files for this batch
                    unique_sources = sorted(set(c.source_index for c in batch))
                    inputs = [self._source_paths[s] for s in unique_sources]

                    out_path = str(Path(self._temp_dir) / f"batch_{gi}_{bi}.mp4")

                    extra_args = [
                        "-c:v", self._settings.video_codec,
                        "-pix_fmt", "yuv420p",
                    ]
                    if self._settings.video_codec == "libx264":
                        extra_args += ["-crf", str(self._settings.video_crf), "-preset", "medium"]
                    else:
                        extra_args += ["-cq", str(self._settings.video_crf), "-preset", "p4"]

                    extra_args += ["-c:a", "aac", "-b:a", "192k"]

                    cmd = self._runner.build_command(
                        inputs=inputs, output=out_path,
                        filter_complex=combined, maps=maps,
                        extra_args=extra_args,
                    )

                    total_dur = sum(c.end - c.start for c in batch)
                    result = self._runner.run(
                        cmd, total_duration=total_dur,
                        progress_callback=lambda p: self.progress.emit(
                            (gi + p) / total_groups
                        ),
                        cancel_check=lambda: self._cancelled,
                    )

                    if result.returncode != 0:
                        # Try CPU fallback if GPU failed
                        if self._settings.video_codec == "h264_nvenc":
                            self.stage.emit("GPU failed, retrying with CPU...")
                            extra_args_cpu = [
                                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                                "-pix_fmt", "yuv420p",
                                "-c:a", "aac", "-b:a", "192k",
                            ]
                            cmd = self._runner.build_command(
                                inputs=inputs, output=out_path,
                                filter_complex=combined, maps=maps,
                                extra_args=extra_args_cpu,
                            )
                            result = self._runner.run(cmd, total_duration=total_dur)
                            if result.returncode != 0:
                                self.error.emit(f"Encoding failed: {result.stderr[-300:]}")
                                self.finished.emit(False)
                                return
                        else:
                            self.error.emit(f"Encoding failed: {result.stderr[-300:]}")
                            self.finished.emit(False)
                            return

                    intermediate_files.append(out_path)

            # Concatenate intermediates if needed
            if len(intermediate_files) == 1:
                import shutil
                shutil.move(intermediate_files[0], self._settings.output_path)
            else:
                self.stage.emit("Concatenating segments...")
                concat_file = str(Path(self._temp_dir) / "concat.txt")
                with open(concat_file, "w") as f:
                    f.write(self._filter_builder.build_concat_file(intermediate_files))

                cmd = self._runner.build_command(
                    inputs=[],
                    output=self._settings.output_path,
                    extra_args=[
                        "-f", "concat", "-safe", "0", "-i", concat_file,
                        "-c:v", self._settings.video_codec,
                        "-crf" if self._settings.video_codec == "libx264" else "-cq",
                        str(self._settings.video_crf),
                        "-c:a", "aac", "-b:a", "192k",
                    ],
                )
                result = self._runner.run(cmd)
                if result.returncode != 0:
                    self.error.emit(f"Concat failed: {result.stderr[-300:]}")
                    self.finished.emit(False)
                    return

            # Add separate audio if needed
            if self._use_separate_audio and self._audio_path:
                self.stage.emit("Adding audio track...")
                final_with_audio = self._settings.output_path
                temp_video = str(Path(self._temp_dir) / "video_only.mp4")
                import shutil
                shutil.move(final_with_audio, temp_video)

                cmd = self._runner.build_command(
                    inputs=[temp_video, self._audio_path],
                    output=final_with_audio,
                    extra_args=[
                        "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k",
                        "-map", "0:v", "-map", "1:a",
                        "-shortest",
                    ],
                )
                result = self._runner.run(cmd)
                if result.returncode != 0:
                    self.error.emit(f"Audio mux failed: {result.stderr[-300:]}")
                    self.finished.emit(False)
                    return

            self.stage.emit("Export complete!")
            self.finished.emit(True)

        except Exception as e:
            logger.exception("Export failed")
            self.error.emit(str(e))
            self.finished.emit(False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest ../tests/gui/test_workers.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gui/workers.py tests/gui/test_workers.py
git commit -m "feat: add QThread workers for analysis and export pipelines"
```

---

### Task 15: Auto-Edit tab

**Files:**
- Create: `src/gui/auto_edit_tab.py`

- [ ] **Step 1: Implement Auto-Edit tab**

Create `src/gui/auto_edit_tab.py`:
```python
import os
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QProgressBar, QTextEdit, QLineEdit, QGroupBox, QMessageBox,
)
from PyQt6.QtCore import Qt
from gui.widgets import FileListWidget

logger = logging.getLogger("autoeditor.gui")


class AutoEditTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio_path: str | None = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        # Video files
        video_header = QHBoxLayout()
        video_header.addWidget(QLabel("Video Files:"))
        self._add_video_btn = QPushButton("Add Video(s)")
        self._add_video_btn.clicked.connect(self._add_videos)
        video_header.addWidget(self._add_video_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_videos)
        video_header.addWidget(self._clear_btn)
        video_header.addStretch()
        input_layout.addLayout(video_header)

        self._file_list = FileListWidget()
        self._file_list.setMaximumHeight(120)
        input_layout.addWidget(self._file_list)

        # Mode toggle
        mode_layout = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._single_radio = QRadioButton("Single Video")
        self._multi_radio = QRadioButton("Multi-Clip Assembly")
        self._single_radio.setChecked(True)
        self._mode_group.addButton(self._single_radio)
        self._mode_group.addButton(self._multi_radio)
        self._single_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._single_radio)
        mode_layout.addWidget(self._multi_radio)
        mode_layout.addStretch()
        input_layout.addLayout(mode_layout)

        # Audio source
        audio_layout = QHBoxLayout()
        self._load_audio_btn = QPushButton("Load Audio")
        self._load_audio_btn.clicked.connect(self._load_audio)
        audio_layout.addWidget(self._load_audio_btn)
        self._audio_label = QLabel("No audio loaded")
        audio_layout.addWidget(self._audio_label)
        self._use_video_audio = QCheckBox("Use video's own audio")
        self._use_video_audio.setToolTip("Use the audio track from the video file for analysis")
        audio_layout.addWidget(self._use_video_audio)
        audio_layout.addStretch()
        input_layout.addLayout(audio_layout)

        layout.addWidget(input_group)

        # Export settings
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Folder:"))
        self._output_folder = QLineEdit()
        self._output_folder.setReadOnly(True)
        out_layout.addWidget(self._output_folder)
        self._browse_output = QPushButton("Browse...")
        self._browse_output.clicked.connect(self._browse_output_folder)
        out_layout.addWidget(self._browse_output)
        export_layout.addLayout(out_layout)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Filename:"))
        self._output_name = QLineEdit("output_edited.mp4")  # Updated dynamically when videos are added
        name_layout.addWidget(self._output_name)
        export_layout.addLayout(name_layout)

        self._info_label = QLabel("Resolution: -- | FPS: -- | Codec: --")
        export_layout.addWidget(self._info_label)

        layout.addWidget(export_group)

        # Go button and progress
        action_layout = QHBoxLayout()
        self._go_btn = QPushButton("Go")
        self._go_btn.setEnabled(False)
        self._go_btn.setMinimumHeight(40)
        self._go_btn.setToolTip("Add at least one video to start")
        action_layout.addWidget(self._go_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        action_layout.addWidget(self._cancel_btn)
        layout.addLayout(action_layout)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._stage_label = QLabel("")
        layout.addWidget(self._stage_label)

        # Log area
        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMaximumHeight(150)
        layout.addWidget(self._log_area)

    def _add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.ts);;All Files (*)",
        )
        for f in files:
            self._file_list.add_file(f)
        self._update_go_state()

    def _clear_videos(self):
        self._file_list.clear_all()
        self._update_go_state()

    def _load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.mp3 *.wav *.flac *.aac *.ogg);;All Files (*)",
        )
        if path:
            self._audio_path = path
            self._audio_label.setText(os.path.basename(path))
            self._update_go_state()

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_folder.setText(folder)

    def _on_mode_changed(self):
        is_multi = self._multi_radio.isChecked()
        self._use_video_audio.setEnabled(not is_multi)
        if is_multi:
            self._use_video_audio.setChecked(False)
            self._use_video_audio.setToolTip("Not available in multi-clip mode")
        else:
            self._use_video_audio.setToolTip("Use the audio track from the video file")
        self._update_go_state()

    def _update_go_state(self):
        has_videos = len(self._file_list.get_paths()) > 0
        is_multi = self._multi_radio.isChecked()
        has_audio = self._audio_path is not None
        use_own = self._use_video_audio.isChecked()

        if is_multi:
            ready = has_videos and len(self._file_list.get_paths()) >= 2 and has_audio
            if not ready:
                self._go_btn.setToolTip("Need 2+ videos and an audio file for multi-clip mode")
        else:
            ready = has_videos and (has_audio or use_own)
            if not ready:
                self._go_btn.setToolTip("Add a video and load audio (or check 'Use video's own audio')")

        self._go_btn.setEnabled(ready)

    def log(self, message: str):
        self._log_area.append(message)

    def set_progress(self, value: float):
        self._progress.setValue(int(value * 100))

    def set_stage(self, text: str):
        self._stage_label.setText(text)
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/auto_edit_tab.py
git commit -m "feat: add Auto-Edit tab with file input, mode toggle, and export settings"
```

---

### Task 16: Trim tab

**Files:**
- Create: `src/gui/trim_tab.py`

- [ ] **Step 1: Implement Trim tab**

Create `src/gui/trim_tab.py`:
```python
import os
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QProgressBar, QGroupBox, QScrollArea, QFrame, QMessageBox,
)
from PyQt6.QtCore import Qt
from gui.widgets import TimecodeInput
from core.models import TrimRegion

logger = logging.getLogger("autoeditor.gui")


class TrimRegionWidget(QFrame):
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        layout = QHBoxLayout(self)

        self._label = QLabel(f"Region {index + 1}")
        layout.addWidget(self._label)

        self._start = TimecodeInput("Start:")
        layout.addWidget(self._start)

        self._end = TimecodeInput("End:")
        layout.addWidget(self._end)

        self._remove_btn = QPushButton("Remove")
        layout.addWidget(self._remove_btn)

    def get_region(self) -> TrimRegion:
        return TrimRegion(
            start=self._start.get_value(),
            end=self._end.get_value(),
            label=self._label.text(),
        )

    def set_max_time(self, duration: float):
        pass  # Validation done on export


class TrimTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: str | None = None
        self._video_duration: float = 0.0
        self._region_widgets: list[TrimRegionWidget] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Video loader
        file_group = QGroupBox("Video File")
        file_layout = QHBoxLayout(file_group)
        self._load_btn = QPushButton("Load Video")
        self._load_btn.clicked.connect(self._load_video)
        file_layout.addWidget(self._load_btn)
        self._file_label = QLabel("No video loaded")
        file_layout.addWidget(self._file_label)
        self._duration_label = QLabel("")
        file_layout.addWidget(self._duration_label)
        file_layout.addStretch()
        layout.addWidget(file_group)

        # Regions
        regions_group = QGroupBox("Trim Regions")
        regions_layout = QVBoxLayout(regions_group)

        self._regions_container = QVBoxLayout()
        regions_layout.addLayout(self._regions_container)

        self._add_region_btn = QPushButton("Add Region")
        self._add_region_btn.clicked.connect(self._add_region)
        regions_layout.addWidget(self._add_region_btn)

        layout.addWidget(regions_group)

        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout(export_group)

        self._mode_group = QButtonGroup(self)
        self._separate_radio = QRadioButton("Export as separate files")
        self._concat_radio = QRadioButton("Concatenate into one file")
        self._separate_radio.setChecked(True)
        self._mode_group.addButton(self._separate_radio)
        self._mode_group.addButton(self._concat_radio)
        self._concat_radio.toggled.connect(self._on_concat_toggled)
        export_layout.addWidget(self._separate_radio)
        export_layout.addWidget(self._concat_radio)

        self._reencode_check = QCheckBox("Force re-encode for frame accuracy")
        self._reencode_check.setToolTip(
            "When unchecked, uses stream copy (fast but cuts only on keyframes). "
            "When checked, re-encodes for frame-accurate cuts."
        )
        export_layout.addWidget(self._reencode_check)

        layout.addWidget(export_group)

        # Export button and progress
        self._export_btn = QPushButton("Export Trim")
        self._export_btn.setEnabled(False)
        self._export_btn.setMinimumHeight(40)
        layout.addWidget(self._export_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        layout.addStretch()

        # Add initial region
        self._add_region()

    def _load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.ts);;All Files (*)",
        )
        if path:
            self._video_path = path
            self._file_label.setText(os.path.basename(path))
            self._update_export_state()

    def _add_region(self):
        idx = len(self._region_widgets)
        widget = TrimRegionWidget(idx)
        widget._remove_btn.clicked.connect(lambda: self._remove_region(widget))
        self._region_widgets.append(widget)
        self._regions_container.addWidget(widget)
        self._update_export_state()

    def _remove_region(self, widget: TrimRegionWidget):
        if len(self._region_widgets) <= 1:
            return  # Keep at least one
        self._region_widgets.remove(widget)
        self._regions_container.removeWidget(widget)
        widget.deleteLater()
        # Renumber
        for i, w in enumerate(self._region_widgets):
            w._label.setText(f"Region {i + 1}")
        self._update_export_state()

    def _on_concat_toggled(self, checked: bool):
        if checked:
            self._reencode_check.setChecked(True)
            self._reencode_check.setEnabled(False)
            self._reencode_check.setToolTip(
                "Re-encode is required when concatenating to avoid corrupted frames"
            )
        else:
            self._reencode_check.setEnabled(True)
            self._reencode_check.setToolTip(
                "When unchecked, uses stream copy (fast but cuts only on keyframes)"
            )

    def _update_export_state(self):
        has_video = self._video_path is not None
        has_regions = len(self._region_widgets) > 0
        self._export_btn.setEnabled(has_video and has_regions)

    def get_regions(self) -> list[TrimRegion]:
        regions = []
        for w in self._region_widgets:
            r = w.get_region()
            if r.end > r.start:
                regions.append(r)
        return regions
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/trim_tab.py
git commit -m "feat: add Trim tab with multi-region support and re-encode toggle"
```

---

### Task 17: Main window

**Files:**
- Create: `src/gui/main_window.py`

- [ ] **Step 1: Implement main window**

Create `src/gui/main_window.py`:
```python
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu, QStatusBar,
    QMessageBox, QFileDialog,
)
from PyQt6.QtGui import QAction
from gui.auto_edit_tab import AutoEditTab
from gui.trim_tab import TrimTab
from config import AppConfig

logger = logging.getLogger("autoeditor.gui")


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self.setWindowTitle("Auto Video Editor")
        self.setMinimumSize(800, 600)

        self._init_menu()
        self._init_tabs()
        self._init_status_bar()

    def _init_menu(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open Video...", self)
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)

        # Recent files submenu
        self._recent_menu = QMenu("Recent Files", self)
        file_menu.addMenu(self._recent_menu)
        self._update_recent_menu()

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings menu
        settings_menu = menu_bar.addMenu("Settings")

        ffmpeg_action = QAction("Set FFmpeg Path...", self)
        ffmpeg_action.triggered.connect(self._set_ffmpeg_path)
        settings_menu.addAction(ffmpeg_action)

        self._gpu_action = QAction("Force CPU Encoding", self)
        self._gpu_action.setCheckable(True)
        self._gpu_action.setChecked(self._config.force_cpu)
        self._gpu_action.triggered.connect(self._toggle_gpu)
        settings_menu.addAction(self._gpu_action)

    def _init_tabs(self):
        self._tabs = QTabWidget()
        self._auto_edit_tab = AutoEditTab()
        self._trim_tab = TrimTab()
        self._tabs.addTab(self._auto_edit_tab, "Auto-Edit")
        self._tabs.addTab(self._trim_tab, "Trim")
        self.setCentralWidget(self._tabs)

    def _init_status_bar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        ffmpeg_status = "FFmpeg: " + (self._config.ffmpeg_path or "Not found")
        gpu_status = "GPU: " + ("Available" if self._config.gpu_detected else "Not available")
        self._status.showMessage(f"{ffmpeg_status} | {gpu_status}")

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", self._config.last_input_folder,
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.ts);;All Files (*)",
        )
        if path:
            import os
            self._config.last_input_folder = os.path.dirname(path)
            self._config.add_recent_file(path)
            self._config.save()
            self._update_recent_menu()
            self._auto_edit_tab._file_list.add_file(path)

    def _update_recent_menu(self):
        self._recent_menu.clear()
        import os
        for filepath in self._config.recent_files:
            if os.path.exists(filepath):
                action = QAction(os.path.basename(filepath), self)
                action.setData(filepath)
                action.triggered.connect(lambda checked, p=filepath: self._open_recent(p))
                self._recent_menu.addAction(action)

    def _open_recent(self, path: str):
        self._auto_edit_tab._file_list.add_file(path)

    def _set_ffmpeg_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate FFmpeg", "",
            "Executables (*.exe);;All Files (*)",
        )
        if path:
            self._config.ffmpeg_path = path
            self._config.save()
            self._status.showMessage(f"FFmpeg: {path}")

    def _toggle_gpu(self, checked: bool):
        self._config.force_cpu = checked
        self._config.save()
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: add main window with menu bar, tabs, and status bar"
```

---

## Chunk 6: Integration & Main Entry Point

### Task 18: Main entry point

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Implement main.py**

Create `src/main.py`:
```python
import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from config import AppConfig
from core.logger import setup_logging
from core.temp_manager import TempManager
from ffmpeg.detector import FFmpegDetector
from ffmpeg.gpu import GPUDetector
from gui.main_window import MainWindow


def main():
    app_dir = Path(__file__).parent
    log_dir = app_dir / "logs"
    temp_dir = app_dir / "temp"

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("Starting Auto Video Editor")

    # Load config
    config = AppConfig(app_dir / "config.json")

    # Cleanup stale temp files
    temp_mgr = TempManager(temp_dir)
    temp_mgr.cleanup_stale(max_age_hours=24)

    # Detect FFmpeg
    detector = FFmpegDetector()
    ffmpeg_path = config.ffmpeg_path or detector.find_ffmpeg()
    ffprobe_path = config.ffprobe_path or detector.find_ffprobe()

    if ffmpeg_path and detector.validate_version(ffmpeg_path):
        config.ffmpeg_path = ffmpeg_path
        if ffprobe_path:
            config.ffprobe_path = ffprobe_path
        logger.info("FFmpeg found: %s", ffmpeg_path)
    else:
        logger.warning("FFmpeg not found or version too old")

    # Detect GPU
    if ffmpeg_path:
        gpu_det = GPUDetector(ffmpeg_path)
        config.gpu_detected = gpu_det.detect_nvenc()
    else:
        config.gpu_detected = False

    config.save()

    # Launch GUI
    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("Auto Video Editor")

    if not ffmpeg_path:
        QMessageBox.warning(
            None, "FFmpeg Not Found",
            "FFmpeg was not found on your system.\n\n"
            "Please download it from https://ffmpeg.org/download.html\n"
            "or set the path in Settings > Set FFmpeg Path.",
        )

    window = MainWindow(config)
    window.show()

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: add main entry point with startup checks"
```

---

### Task 19: Wire up Auto-Edit pipeline in GUI

**Files:**
- Modify: `src/gui/auto_edit_tab.py`

This task connects the Go button to the full pipeline: audio extraction → analysis → cut generation → export.

- [ ] **Step 1: Add pipeline orchestration to AutoEditTab**

Add these methods to `src/gui/auto_edit_tab.py`:

```python
# Add to __init__ method:
self._go_btn.clicked.connect(self._start_pipeline)
self._cancel_btn.clicked.connect(self._cancel_pipeline)
self._analysis_worker = None
self._export_worker = None

# Add new methods:
def set_dependencies(self, config, ffmpeg_runner, filter_builder, gpu_detector, temp_manager):
    """Called by MainWindow after construction to inject dependencies."""
    self._config = config
    self._runner = ffmpeg_runner
    self._filter_builder = filter_builder
    self._gpu_detector = gpu_detector
    self._temp_manager = temp_manager

def _start_pipeline(self):
    paths = self._file_list.get_paths()
    if not paths:
        return

    self._go_btn.setEnabled(False)
    self._cancel_btn.setEnabled(True)
    self._progress.setValue(0)

    # Always create a temp session
    self._temp_manager.create_session()

    # Update default filename from first video
    import os
    first_name = os.path.splitext(os.path.basename(paths[0]))[0]
    if self._output_name.text() == "output_edited.mp4":
        self._output_name.setText(f"{first_name}_edited.mp4")

    use_own_audio = self._use_video_audio.isChecked()

    if use_own_audio:
        # Extract audio from first video
        audio_path = self._extract_audio(paths[0])
    else:
        audio_path = self._audio_path

    if not audio_path:
        self.log("Error: No audio source available")
        self._go_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        return

    from gui.workers import AnalysisWorker
    self._analysis_worker = AnalysisWorker(audio_path)
    self._analysis_worker.progress.connect(self.log)
    self._analysis_worker.finished.connect(
        lambda result: self._on_analysis_done(result, paths, audio_path, use_own_audio)
    )
    self._analysis_worker.error.connect(self._on_error)
    self._analysis_worker.start()

def _extract_audio(self, video_path: str) -> str | None:
    session_dir = self._temp_manager.create_session()
    audio_out = str(session_dir / "extracted_audio.wav")
    cmd = self._runner.build_command(
        inputs=[video_path], output=audio_out,
        extra_args=["-vn", "-ac", "1", "-ar", "22050"],
    )
    result = self._runner.run(cmd)
    if result.returncode != 0:
        self.log(f"Audio extraction failed: {result.stderr[-200:]}")
        return None
    return audio_out

def _on_analysis_done(self, result, paths, audio_path, use_separate_audio):
    if result is None:
        self.log("Analysis cancelled")
        self._reset_ui()
        return

    self.log(f"Analysis complete: {result.tempo:.0f} BPM, {len(result.beats)} beats")

    from core.cut_generator import CutGenerator
    gen = CutGenerator(
        min_cut=self._config.min_cut_duration,
        max_cut=self._config.max_cut_duration,
        beat_snap_ms=self._config.beat_snap_tolerance_ms,
    )
    cuts = gen.generate(result, num_sources=len(paths))
    self.log(f"Generated {len(cuts)} cut points")

    # Resolve export settings
    from core.export_manager import ExportManager
    em = ExportManager()
    probes = []
    for p in paths:
        try:
            probes.append(self._runner.probe(p))
        except Exception as e:
            self.log(f"Warning: Failed to probe {p}: {e}")

    output_folder = self._output_folder.text() or str(Path(paths[0]).parent)
    output_name = self._output_name.text() or "output_edited.mp4"
    output_path = str(Path(output_folder) / output_name)

    force_cpu = self._config.force_cpu
    codec = self._gpu_detector.get_video_encoder(force_cpu)
    crf = self._gpu_detector.get_crf(force_cpu)

    settings = em.resolve_settings(probes, output_path, codec, crf)
    self.log(f"Export: {settings.width}x{settings.height} @ {settings.fps}fps, {settings.video_codec}")

    from gui.workers import ExportWorker
    self._export_worker = ExportWorker(
        cuts=cuts, ffmpeg_runner=self._runner,
        filter_builder=self._filter_builder, settings=settings,
        source_paths=paths,
        audio_path=audio_path if use_separate_audio else None,
        temp_dir=str(self._temp_manager.session_dir),
        use_separate_audio=use_separate_audio,
    )
    self._export_worker.progress.connect(self.set_progress)
    self._export_worker.stage.connect(self.set_stage)
    self._export_worker.finished.connect(self._on_export_done)
    self._export_worker.error.connect(self._on_error)
    self._export_worker.start()

def _on_export_done(self, success: bool):
    if success:
        self.log("Export completed successfully!")
        self.set_stage("Done!")
        QMessageBox.information(self, "Success", "Video export complete!")
    self._temp_manager.cleanup_session()
    self._reset_ui()

def _on_error(self, message: str):
    self.log(f"Error: {message}")
    QMessageBox.critical(self, "Error", message)
    self._reset_ui()

def _cancel_pipeline(self):
    if self._analysis_worker and self._analysis_worker.isRunning():
        self._analysis_worker.cancel()
    if self._export_worker and self._export_worker.isRunning():
        self._export_worker.cancel()

def _reset_ui(self):
    self._go_btn.setEnabled(True)
    self._cancel_btn.setEnabled(False)
    self._update_go_state()
```

- [ ] **Step 2: Update MainWindow to inject dependencies**

Add to `src/gui/main_window.py` `_init_tabs` method, after creating tabs:

```python
from ffmpeg.runner import FFmpegRunner
from ffmpeg.filter_graph import FilterGraphBuilder
from ffmpeg.gpu import GPUDetector
from core.temp_manager import TempManager
from pathlib import Path

app_dir = Path(__file__).parent.parent
runner = FFmpegRunner(config.ffmpeg_path or "ffmpeg", config.ffprobe_path or "ffprobe")
filter_builder = FilterGraphBuilder()
gpu_det = GPUDetector(config.ffmpeg_path or "ffmpeg")
if config.gpu_detected and not config.force_cpu:
    gpu_det._nvenc_available = config.gpu_detected
temp_mgr = TempManager(app_dir / "temp")

self._auto_edit_tab.set_dependencies(config, runner, filter_builder, gpu_det, temp_mgr)
```

- [ ] **Step 3: Commit**

```bash
git add src/gui/auto_edit_tab.py src/gui/main_window.py
git commit -m "feat: wire up full auto-edit pipeline (analysis → cuts → export)"
```

---

### Task 20: Wire up Trim tab export

**Files:**
- Modify: `src/gui/trim_tab.py`

- [ ] **Step 1: Add trim export logic**

Add to `src/gui/trim_tab.py`:

```python
# Add to __init__:
self._export_btn.clicked.connect(self._export_trim)
self._runner = None

def set_dependencies(self, config, ffmpeg_runner, temp_manager):
    self._config = config
    self._runner = ffmpeg_runner
    self._temp_manager = temp_manager

def _export_trim(self):
    regions = self.get_regions()
    if not regions or not self._video_path:
        return

    # Validate
    for r in regions:
        if r.end <= r.start:
            QMessageBox.warning(self, "Invalid Region", f"{r.label}: end must be after start")
            return
        if r.end - r.start < 0.5:
            QMessageBox.warning(self, "Short Region", f"{r.label} is less than 0.5 seconds")
            return

    is_concat = self._concat_radio.isChecked()
    reencode = self._reencode_check.isChecked()
    source_name = os.path.splitext(os.path.basename(self._video_path))[0]

    output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
    if not output_dir:
        return

    self._export_btn.setEnabled(False)
    self._progress.setValue(0)

    session_dir = self._temp_manager.create_session()
    segment_files = []

    for i, region in enumerate(regions):
        out_name = f"{source_name}_trim_{i + 1}.mp4"
        out_path = os.path.join(output_dir if not is_concat else str(session_dir), out_name)

        if reencode:
            force_cpu = self._config.force_cpu if hasattr(self, '_config') else True
            codec = "libx264"
            extra = ["-c:v", codec, "-crf", "18", "-preset", "medium",
                     "-c:a", "aac", "-b:a", "192k"]
        else:
            extra = ["-c", "copy"]

        cmd = self._runner.build_command(
            inputs=[self._video_path], output=out_path,
            extra_args=["-ss", str(region.start), "-to", str(region.end)] + extra,
        )
        result = self._runner.run(cmd)
        if result.returncode != 0:
            QMessageBox.critical(self, "Error", f"Failed to export {region.label}")
            self._export_btn.setEnabled(True)
            return

        segment_files.append(out_path)
        self._progress.setValue(int((i + 1) / len(regions) * (100 if not is_concat else 80)))

    if is_concat and len(segment_files) > 1:
        concat_file = os.path.join(str(session_dir), "concat.txt")
        with open(concat_file, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        final_output = os.path.join(output_dir, f"{source_name}_trimmed.mp4")
        cmd = self._runner.build_command(
            inputs=[], output=final_output,
            extra_args=["-f", "concat", "-safe", "0", "-i", concat_file,
                        "-c:v", "libx264", "-crf", "18",
                        "-c:a", "aac", "-b:a", "192k"],
        )
        result = self._runner.run(cmd)
        if result.returncode != 0:
            QMessageBox.critical(self, "Error", "Failed to concatenate regions")
            self._export_btn.setEnabled(True)
            return

    self._progress.setValue(100)
    self._temp_manager.cleanup_session()
    self._export_btn.setEnabled(True)
    QMessageBox.information(self, "Done", "Trim export complete!")
```

- [ ] **Step 2: Update MainWindow to inject trim dependencies**

Add to `_init_tabs` in `main_window.py`:
```python
self._trim_tab.set_dependencies(config, runner, temp_mgr)
```

- [ ] **Step 3: Commit**

```bash
git add src/gui/trim_tab.py src/gui/main_window.py
git commit -m "feat: wire up Trim tab export with concat and re-encode support"
```

---

### Task 21: Integration tests

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration smoke tests**

Create `tests/test_integration.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from core.models import AnalysisResult, CutPoint
from core.cut_generator import CutGenerator
from ffmpeg.filter_graph import FilterGraphBuilder
from core.export_manager import ExportManager
import numpy as np


class TestPipelineIntegration:
    def test_analysis_to_cuts_to_filter_graph(self):
        """End-to-end: analysis result → cuts → filter graph string."""
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
```

- [ ] **Step 2: Run tests**

```bash
cd src && python -m pytest ../tests/test_integration.py -v
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration smoke tests for full pipeline"
```

---

### Task 22: Run all tests and verify

- [ ] **Step 1: Run full test suite**

```bash
cd src && python -m pytest ../tests/ -v --tb=short
```
Expected: All tests PASS

- [ ] **Step 2: Test app launch**

```bash
cd src && python main.py
```
Expected: GUI window opens with Auto-Edit and Trim tabs. FFmpeg warning if not installed.

- [ ] **Step 3: Final commit**

```bash
git add src/ tests/
git commit -m "chore: finalize project structure"
```
