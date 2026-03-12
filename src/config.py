import copy
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
    "aggressiveness": 5,
    "allowed_transitions": [
        "hard_cut", "crossfade", "crossfade_slow",
        "fade_black", "wipe_left", "wipe_right",
    ],
}


class AppConfig:
    def __init__(self, path: Path | str | None = None):
        if path is None:
            from paths import get_app_dir
            path = get_app_dir() / "config.json"
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
        for key, default in DEFAULTS.items():
            if key not in self._data:
                self._data[key] = copy.deepcopy(default)

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

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
