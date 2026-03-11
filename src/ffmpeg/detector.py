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
        result = shutil.which(name)
        if result:
            logger.info("Found %s on PATH: %s", name, result)
            return result
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
                return False
            major, minor = int(match.group(1)), int(match.group(2))
            patch_v = int(match.group(3)) if match.group(3) else 0
            return (major, minor, patch_v) >= MIN_VERSION
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
