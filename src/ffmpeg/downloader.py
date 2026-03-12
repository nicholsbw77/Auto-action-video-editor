import io
import logging
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request
from typing import Callable

from paths import get_app_dir

logger = logging.getLogger("autoeditor.ffmpeg")

FFMPEG_DOWNLOAD_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
CHUNK_SIZE = 65536


class FFmpegDownloader:
    def __init__(self, target_dir: Path | None = None):
        self.target_dir = target_dir or (get_app_dir() / "tools")

    def download(self, progress_callback: Callable[[float], None] | None = None) -> tuple[str, str]:
        """Download and extract ffmpeg + ffprobe. Returns (ffmpeg_path, ffprobe_path)."""
        self.target_dir.mkdir(parents=True, exist_ok=True)

        req = Request(FFMPEG_DOWNLOAD_URL, headers={"User-Agent": "AutoVideoEditor/1.0"})
        response = urlopen(req, timeout=120)
        total = int(response.headers.get("Content-Length", 0))
        data = io.BytesIO()
        downloaded = 0

        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            data.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total > 0:
                progress_callback(downloaded / total * 0.8)  # 80% for download

        data.seek(0)
        with zipfile.ZipFile(data) as zf:
            for member in zf.namelist():
                basename = Path(member).name.lower()
                if basename in ("ffmpeg.exe", "ffprobe.exe"):
                    target = self.target_dir / Path(member).name
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    logger.info("Extracted %s to %s", basename, target)

        if progress_callback:
            progress_callback(1.0)

        ffmpeg = str(self.target_dir / "ffmpeg.exe")
        ffprobe = str(self.target_dir / "ffprobe.exe")

        if not Path(ffmpeg).exists() or not Path(ffprobe).exists():
            raise RuntimeError("Failed to extract ffmpeg binaries from download")

        return ffmpeg, ffprobe
