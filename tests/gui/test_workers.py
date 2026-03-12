import pytest
from unittest.mock import MagicMock
from gui.workers import AnalysisWorker, ExportWorker, AudioExtractWorker, TrimWorker


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


class TestAudioExtractWorker:
    def test_creates_with_params(self):
        runner = MagicMock()
        worker = AudioExtractWorker("video.mp4", "audio.wav", runner)
        assert worker._video_path == "video.mp4"
        assert worker._output_path == "audio.wav"
        assert worker._runner is runner


class TestTrimWorker:
    def test_creates_with_params(self):
        runner = MagicMock()
        worker = TrimWorker(
            video_path="video.mp4", regions=[], output_dir="/out",
            is_concat=False, reencode=True, source_name="video",
            runner=runner, session_dir="/tmp/session",
        )
        assert worker._video_path == "video.mp4"
        assert worker._is_concat is False
        assert worker._reencode is True

    def test_cancel_sets_flag(self):
        worker = TrimWorker(
            video_path="video.mp4", regions=[], output_dir="/out",
            is_concat=False, reencode=True, source_name="video",
            runner=MagicMock(), session_dir="/tmp/session",
        )
        worker.cancel()
        assert worker._cancelled is True
