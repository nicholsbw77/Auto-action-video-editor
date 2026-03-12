import json
import logging
import re
import subprocess
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
                    try:
                        process.stdin.write("q\n")
                        process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        process.kill()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    raise RuntimeError("FFmpeg cancelled by user")
        except BrokenPipeError:
            pass
        except RuntimeError:
            raise  # Re-raise cancellation
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg did not exit in 60s, killing process")
            process.kill()
            process.wait(timeout=5)
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
