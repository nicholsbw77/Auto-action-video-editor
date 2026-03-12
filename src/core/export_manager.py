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

        # Validate: if no video streams found, use safe defaults
        if max_w == 0 or max_h == 0 or max_fps == 0.0:
            logger.warning("No valid video streams found in probes, using 1920x1080@30fps defaults")
            max_w = max_w or 1920
            max_h = max_h or 1080
            max_fps = max_fps or 30.0

        # Cap at 1080p60
        if max_w > MAX_WIDTH or max_h > MAX_HEIGHT:
            scale = min(MAX_WIDTH / max_w, MAX_HEIGHT / max_h)
            max_w = int(max_w * scale)
            max_h = int(max_h * scale)
        max_fps = min(max_fps, MAX_FPS)

        # Ensure even dimensions (required by libx264 and h264_nvenc)
        max_w = max_w - (max_w % 2)
        max_h = max_h - (max_h % 2)

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
