import logging
import subprocess

logger = logging.getLogger("autoeditor.ffmpeg")


class GPUDetector:
    def __init__(self, ffmpeg_path: str):
        self._ffmpeg_path = ffmpeg_path
        self._nvenc_available = False

    def detect_nvenc(self) -> bool:
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
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
                self._nvenc_available = True
                return True
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass

        self._nvenc_available = False
        return False

    def get_video_encoder(self, force_cpu: bool = False) -> str:
        if force_cpu or not self._nvenc_available:
            return "libx264"
        return "h264_nvenc"

    def get_crf(self, force_cpu: bool = False) -> int:
        if force_cpu or not self._nvenc_available:
            return 18
        return 20

    def get_encoder_preset(self, force_cpu: bool = False) -> str:
        if force_cpu or not self._nvenc_available:
            return "medium"
        return "p4"

    @property
    def nvenc_available(self) -> bool:
        return self._nvenc_available

    @nvenc_available.setter
    def nvenc_available(self, value: bool):
        self._nvenc_available = value
