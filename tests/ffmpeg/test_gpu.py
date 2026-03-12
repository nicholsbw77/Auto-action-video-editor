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
        det.nvenc_available = True
        assert det.get_video_encoder(force_cpu=False) == "h264_nvenc"
        assert det.get_crf(force_cpu=False) == 20

    def test_get_encoder_cpu(self):
        det = GPUDetector(ffmpeg_path="ffmpeg")
        det.nvenc_available = False
        assert det.get_video_encoder(force_cpu=False) == "libx264"
        assert det.get_crf(force_cpu=False) == 18

    def test_force_cpu_overrides_gpu(self):
        det = GPUDetector(ffmpeg_path="ffmpeg")
        det.nvenc_available = True
        assert det.get_video_encoder(force_cpu=True) == "libx264"
        assert det.get_crf(force_cpu=True) == 18

    def test_nvenc_available_setter(self):
        det = GPUDetector(ffmpeg_path="ffmpeg")
        assert det.nvenc_available is False
        det.nvenc_available = True
        assert det.nvenc_available is True
        det.nvenc_available = False
        assert det.nvenc_available is False
