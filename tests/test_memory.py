"""tests for memory management and model lifecycle"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from meetcap.services.summarization import SummarizationService
from meetcap.services.transcription import (
    FasterWhisperService,
    MlxWhisperService,
    VoskTranscriptionService,
)
from meetcap.utils.memory import (
    MEMORY_HEADROOM_MB,
    MemoryMonitor,
    check_memory_for_model,
    check_memory_pressure,
    estimate_model_memory,
    get_available_memory_mb,
    get_fallback_model,
    get_memory_usage,
    get_total_memory_mb,
    preflight_memory_check,
    safe_model_loading,
)


class TestMemoryUtils:
    """test memory utility functions."""

    def test_get_memory_usage(self):
        """test memory usage retrieval."""
        usage = get_memory_usage()
        assert isinstance(usage, dict)
        assert "rss_mb" in usage
        assert "vms_mb" in usage
        assert "percent" in usage
        assert usage["rss_mb"] >= 0
        assert usage["vms_mb"] >= 0
        assert 0 <= usage["percent"] <= 100

    @patch("psutil.virtual_memory")
    def test_check_memory_pressure(self, mock_virtual_memory):
        """test memory pressure detection."""
        # mock low memory usage
        mock_memory = Mock()
        mock_memory.percent = 50
        mock_virtual_memory.return_value = mock_memory

        assert check_memory_pressure(threshold_percent=80) is False

        # mock high memory usage
        mock_memory.percent = 90
        mock_virtual_memory.return_value = mock_memory
        assert check_memory_pressure(threshold_percent=80) is True

    def test_estimate_model_memory(self):
        """test model memory estimation."""
        # STT models
        assert estimate_model_memory("stt", "whisper-large-v3") == 1500
        assert estimate_model_memory("stt", "whisper-small") == 500
        assert estimate_model_memory("stt", "mlx-community/whisper-large-v3-turbo") == 1500
        assert estimate_model_memory("stt", "vosk-standard") == 1800

        # parakeet models
        assert estimate_model_memory("stt", "mlx-community/parakeet-tdt-0.6b-v3") == 800
        assert estimate_model_memory("stt", "parakeet-tdt-0.6b") == 800

        # LLM models
        assert estimate_model_memory("llm", "qwen2.5-3b") == 3000
        assert estimate_model_memory("llm", "qwen2.5-7b") == 7000
        assert estimate_model_memory("llm", "mlx-community/qwen3.5-4b-mlx-4bit") == 3200
        assert estimate_model_memory("llm", "mlx-community/qwen3.5-9b-mlx-4bit") == 6000

        # diarization
        assert estimate_model_memory("diarization", "sherpa-diarization") == 250

        # unknown models
        assert estimate_model_memory("stt", "unknown-model") == 1500
        assert estimate_model_memory("llm", "unknown-model") == 4000
        assert estimate_model_memory("diarization", "unknown") == 250

    def test_get_available_memory(self):
        """test available memory retrieval."""
        available = get_available_memory_mb()
        assert isinstance(available, float)
        assert available >= 0

    def test_get_total_memory(self):
        """test total memory retrieval."""
        total = get_total_memory_mb()
        assert isinstance(total, float)
        assert total >= 0

    def test_get_fallback_model(self):
        """test fallback model selection."""
        # should suggest fallback when memory is low
        fallback = get_fallback_model("whisper-large-v3", available_memory_mb=1000)
        assert fallback == "whisper-small"

        fallback = get_fallback_model(
            "mlx-community/whisper-large-v3-turbo", available_memory_mb=1000
        )
        assert fallback == "whisper-small"

        # no fallback for unknown models
        fallback = get_fallback_model("unknown-model", available_memory_mb=1000)
        assert fallback is None

        # no fallback when sufficient memory
        fallback = get_fallback_model("whisper-large-v3", available_memory_mb=5000)
        assert fallback is None


class TestMemoryMonitor:
    """test memory monitoring functionality."""

    def test_checkpoint_creation(self):
        """test creating memory checkpoints."""
        monitor = MemoryMonitor()

        # create checkpoint
        usage = monitor.checkpoint("test", verbose=False)
        assert isinstance(usage, dict)
        assert "test" in monitor.checkpoints
        assert monitor.checkpoints["test"] == usage

    def test_checkpoint_delta(self):
        """test calculating delta between checkpoints."""
        monitor = MemoryMonitor()

        # create two checkpoints
        monitor.checkpoint("start", verbose=False)
        monitor.checkpoint("end", verbose=False)

        # calculate delta
        delta = monitor.get_delta("start", "end")
        assert isinstance(delta, float)

        # delta for non-existent checkpoints
        delta = monitor.get_delta("missing1", "missing2")
        assert delta == 0.0

    def test_memory_report(self, capsys):
        """test memory usage report generation."""
        monitor = MemoryMonitor()

        # create checkpoints
        monitor.checkpoint("start", verbose=False)
        monitor.checkpoint("middle", verbose=False)
        monitor.checkpoint("end", verbose=False)

        # generate report
        monitor.report(detailed=True)

        captured = capsys.readouterr()
        assert "Memory Usage Report" in captured.out
        assert "start:" in captured.out
        assert "middle:" in captured.out
        assert "end:" in captured.out
        assert "Peak memory:" in captured.out


class TestSTTMemoryManagement:
    """test memory management for STT services."""

    @patch("meetcap.services.transcription.FasterWhisperService._load_model")
    def test_faster_whisper_unload(self, mock_load):
        """test FasterWhisperService model unloading."""
        service = FasterWhisperService(model_name="small", auto_download=False)

        # simulate loaded model
        service.model = Mock()

        # test unloading
        service.unload_model()

        assert service.model is None
        # gc.collect() is actually called in unload_model, not an issue

    @patch("meetcap.services.transcription.MlxWhisperService._load_model")
    def test_mlx_whisper_unload(self, mock_load):
        """test MlxWhisperService model unloading."""
        service = MlxWhisperService(model_name="test", auto_download=False)

        # simulate loaded model
        service.model = "loaded"
        service.model_source = "test_source"

        # test unloading
        service.unload_model()

        assert service.model is None
        assert service.model_source is None

    @patch("meetcap.services.transcription.VoskTranscriptionService._load_model")
    def test_vosk_unload(self, mock_load):
        """test VoskTranscriptionService model unloading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            service = VoskTranscriptionService(model_path=str(model_path), enable_diarization=False)

            # simulate loaded models
            service.model = Mock()
            service.spk_model = Mock()

            # test unloading
            service.unload_model()

            assert service.model is None
            assert service.spk_model is None

    def test_is_loaded_status(self):
        """test model loaded status checking."""
        service = FasterWhisperService(model_name="small", auto_download=False)

        # initially not loaded
        assert service.is_loaded() is False

        # simulate loaded model
        service.model = Mock()
        assert service.is_loaded() is True

        # after unloading
        service.unload_model()
        assert service.is_loaded() is False


class TestLLMMemoryManagement:
    """test memory management for LLM service."""

    @patch("meetcap.services.summarization.SummarizationService._load_model")
    def test_llm_unload(self, mock_load, temp_dir):
        """test SummarizationService model unloading."""
        service = SummarizationService(model_name="test-model")

        # simulate loaded model
        service.model = Mock()
        service.processor = Mock()

        # test unloading
        service.unload_model()

        assert service.model is None

    def test_llm_is_loaded(self, temp_dir):
        """test LLM loaded status checking."""
        service = SummarizationService(model_name="test-model")

        # initially not loaded
        assert service.is_loaded() is False

        # simulate loaded model
        service.model = Mock()
        assert service.is_loaded() is True

        # after unloading
        service.unload_model()
        assert service.is_loaded() is False


class TestMemoryLifecycle:
    """test complete memory lifecycle during processing."""

    @patch("meetcap.cli.preflight_memory_check", return_value=(True, ""))
    @patch("meetcap.cli.RecordingOrchestrator._process_audio_to_transcript")
    @patch("meetcap.cli.RecordingOrchestrator._process_transcript_to_summary")
    def test_orchestrator_memory_management(
        self, mock_summary, mock_transcript, mock_preflight, config
    ):
        """test orchestrator properly manages memory between STT and LLM."""
        from meetcap.cli import RecordingOrchestrator

        # enable memory monitoring
        config.config["memory"]["enable_monitoring"] = True
        orchestrator = RecordingOrchestrator(config)

        # create test audio file
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "test.wav"
            audio_file.write_bytes(b"dummy audio")

            # mock successful processing
            transcript_path = Path(tmpdir) / "test.transcript.txt"
            json_path = Path(tmpdir) / "test.transcript.json"
            summary_path = Path(tmpdir) / "test.summary.md"

            transcript_path.write_text("test transcript")
            json_path.write_text('{"segments": []}')
            summary_path.write_text("## Meeting Title\nTest")

            mock_transcript.return_value = (transcript_path, json_path)
            mock_summary.return_value = summary_path

            # run processing
            orchestrator._process_recording(
                audio_path=audio_file, stt_engine="fwhisper", llm_model=None, seed=None
            )

            # verify memory monitoring was enabled
            assert orchestrator.memory_monitor is not None
            assert "start" in orchestrator.memory_monitor.checkpoints
            assert "before_stt" in orchestrator.memory_monitor.checkpoints
            assert "after_stt" in orchestrator.memory_monitor.checkpoints
            assert "before_llm" in orchestrator.memory_monitor.checkpoints
            assert "after_llm" in orchestrator.memory_monitor.checkpoints

    @patch("meetcap.cli.check_memory_for_model", return_value=(True, 8000.0, 1500.0, ""))
    @patch("meetcap.services.transcription.FasterWhisperService.transcribe")
    @patch("meetcap.services.transcription.FasterWhisperService._load_model")
    def test_stt_explicit_lifecycle(self, mock_load, mock_transcribe, mock_mem, config):
        """test STT service explicit load/unload lifecycle."""
        from meetcap.cli import RecordingOrchestrator
        from meetcap.services.transcription import TranscriptResult

        orchestrator = RecordingOrchestrator(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_file = Path(tmpdir) / "test.wav"
            audio_file.write_bytes(b"dummy audio")
            base_path = Path(tmpdir) / "test"

            # mock transcription result
            mock_result = TranscriptResult(
                audio_path=str(audio_file),
                sample_rate=48000,
                language="en",
                segments=[],
                duration=0.0,
                stt={"engine": "faster-whisper"},
            )
            mock_transcribe.return_value = mock_result

            # process audio
            result = orchestrator._process_audio_to_transcript(audio_file, base_path, "fwhisper")

            assert result is not None
            # verify model was loaded (through _load_model call)
            mock_load.assert_called()


class TestMemoryConfiguration:
    """test memory configuration options."""

    def test_default_memory_config(self, config):
        """test default memory configuration values."""
        assert config.get("memory", "aggressive_gc") is True
        assert config.get("memory", "enable_monitoring") is False
        assert config.get("memory", "memory_report") is False
        assert config.get("memory", "warning_threshold") == 80
        assert config.get("memory", "critical_threshold") == 90
        assert config.get("memory", "auto_fallback") is True
        assert config.get("memory", "explicit_lifecycle") is True

    def test_memory_env_overrides(self, monkeypatch):
        """test memory configuration via environment variables."""
        from meetcap.utils.config import Config

        # set environment variables
        monkeypatch.setenv("MEETCAP_MEMORY_AGGRESSIVE_GC", "false")
        monkeypatch.setenv("MEETCAP_MEMORY_MONITORING", "true")
        monkeypatch.setenv("MEETCAP_MEMORY_REPORT", "true")
        monkeypatch.setenv("MEETCAP_MEMORY_WARNING_THRESHOLD", "75")
        monkeypatch.setenv("MEETCAP_MEMORY_AUTO_FALLBACK", "false")

        config = Config()

        assert config.get("memory", "aggressive_gc") is False
        assert config.get("memory", "enable_monitoring") is True
        assert config.get("memory", "memory_report") is True
        assert config.get("memory", "warning_threshold") == 75
        assert config.get("memory", "auto_fallback") is False


class TestMemoryPreflightCheck:
    """test pre-flight memory checks."""

    @patch("meetcap.utils.memory.get_available_memory_mb")
    @patch("meetcap.utils.memory.get_total_memory_mb")
    def test_check_memory_for_model_sufficient(self, mock_total, mock_avail):
        """test memory check passes when sufficient memory available."""
        mock_avail.return_value = 8000.0
        mock_total.return_value = 16000.0

        sufficient, avail, needed, msg = check_memory_for_model(
            "stt", "mlx-community/parakeet-tdt-0.6b-v3"
        )
        assert sufficient is True
        assert msg == ""
        assert avail == 8000.0

    @patch("meetcap.utils.memory.get_available_memory_mb")
    @patch("meetcap.utils.memory.get_total_memory_mb")
    def test_check_memory_for_model_insufficient(self, mock_total, mock_avail):
        """test memory check fails when insufficient memory available."""
        mock_avail.return_value = 500.0
        mock_total.return_value = 8000.0

        sufficient, avail, needed, msg = check_memory_for_model(
            "llm", "mlx-community/Qwen3.5-4B-MLX-4bit"
        )
        assert sufficient is False
        assert "insufficient memory" in msg
        assert avail == 500.0
        assert needed == 3200 + MEMORY_HEADROOM_MB

    @patch("meetcap.utils.memory.get_available_memory_mb")
    def test_check_memory_psutil_unavailable(self, mock_avail):
        """test memory check passes gracefully when psutil unavailable."""
        mock_avail.return_value = 0.0

        sufficient, avail, needed, msg = check_memory_for_model("stt", "parakeet")
        assert sufficient is True
        assert msg == ""

    @patch("meetcap.utils.memory.get_available_memory_mb")
    @patch("meetcap.utils.memory.get_total_memory_mb")
    def test_preflight_check_sufficient(self, mock_total, mock_avail):
        """test preflight check passes with adequate memory."""
        mock_avail.return_value = 8000.0
        mock_total.return_value = 16000.0

        can_proceed, warning = preflight_memory_check(
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        )
        assert can_proceed is True
        assert warning == ""

    @patch("meetcap.utils.memory.get_available_memory_mb")
    @patch("meetcap.utils.memory.get_total_memory_mb")
    def test_preflight_check_low_memory_warning(self, mock_total, mock_avail):
        """test preflight check warns when memory is tight."""
        # enough for the smaller STT model but tight for LLM
        mock_avail.return_value = 3000.0
        mock_total.return_value = 8000.0

        can_proceed, warning = preflight_memory_check(
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        )
        assert "low memory warning" in warning
        # still enough for at least 50% of peak → can proceed
        assert can_proceed is True

    @patch("meetcap.utils.memory.get_available_memory_mb")
    @patch("meetcap.utils.memory.get_total_memory_mb")
    def test_preflight_check_critical_memory(self, mock_total, mock_avail):
        """test preflight check blocks when memory is critically low."""
        mock_avail.return_value = 500.0
        mock_total.return_value = 8000.0

        can_proceed, warning = preflight_memory_check(
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        )
        assert can_proceed is False
        assert "low memory warning" in warning

    @patch("meetcap.utils.memory.get_available_memory_mb")
    def test_preflight_check_psutil_unavailable(self, mock_avail):
        """test preflight check passes when psutil unavailable."""
        mock_avail.return_value = 0.0

        can_proceed, warning = preflight_memory_check(
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        )
        assert can_proceed is True
        assert warning == ""

    @patch("meetcap.utils.memory.get_available_memory_mb")
    @patch("meetcap.utils.memory.get_total_memory_mb")
    def test_preflight_uses_sequential_peak(self, mock_total, mock_avail):
        """test that preflight checks peak (max single model), not sum."""
        # parakeet ~800 MB, qwen3.5-4b ~3200 MB
        # sequential peak = 3200 + 512 headroom = 3712
        # if it summed both, would need ~4000+
        mock_avail.return_value = 4000.0
        mock_total.return_value = 16000.0

        can_proceed, warning = preflight_memory_check(
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        )
        assert can_proceed is True
        assert warning == ""


class TestProcessScreenReprocessMode:
    """test ProcessScreen summary-only reprocessing."""

    def test_read_existing_transcript(self, temp_dir):
        """test reading existing transcript for summary-only mode."""
        from meetcap.tui.screens.process import ProcessScreen

        # create a fake audio file and transcript
        audio_file = temp_dir / "recording.opus"
        audio_file.write_bytes(b"fake audio")
        transcript_file = temp_dir / "recording.transcript.txt"
        transcript_file.write_text("hello world this is a test transcript")

        screen = ProcessScreen(audio_path=audio_file, mode="summary")
        text = screen._read_existing_transcript(audio_file)
        assert text == "hello world this is a test transcript"

    def test_read_existing_transcript_not_found(self, temp_dir):
        """test reading transcript when none exists."""
        from meetcap.tui.screens.process import ProcessScreen

        audio_file = temp_dir / "recording.opus"
        audio_file.write_bytes(b"fake audio")

        screen = ProcessScreen(audio_path=audio_file, mode="summary")
        text = screen._read_existing_transcript(audio_file)
        assert text is None

    def test_process_screen_mode_default(self):
        """test ProcessScreen defaults to full pipeline mode."""
        from meetcap.tui.screens.process import ProcessScreen

        screen = ProcessScreen()
        assert screen._mode == "stt"

    def test_process_screen_mode_summary(self, temp_dir):
        """test ProcessScreen accepts summary-only mode."""
        from meetcap.tui.screens.process import ProcessScreen

        audio_file = temp_dir / "recording.opus"
        audio_file.write_bytes(b"fake audio")

        screen = ProcessScreen(audio_path=audio_file, mode="summary")
        assert screen._mode == "summary"
        assert screen._audio_path == audio_file

    def test_get_stt_model_name(self):
        """test _get_stt_model_name helper for various engines."""
        from meetcap.tui.screens.process import ProcessScreen

        config = Mock()
        config.get.return_value = "test-model"

        name = ProcessScreen._get_stt_model_name(config, "parakeet")
        assert name == "test-model"

        name = ProcessScreen._get_stt_model_name(config, "mlx-whisper")
        assert name == "test-model"

        name = ProcessScreen._get_stt_model_name(config, "faster-whisper")
        assert name == "test-model"

        name = ProcessScreen._get_stt_model_name(config, "whisper.cpp")
        assert name == "whisper.cpp"


class TestSafeModelLoading:
    """test safe_model_loading wrapper."""

    def test_successful_load(self):
        """test safe_model_loading with successful load function."""
        load_func = Mock()
        safe_model_loading(load_func, "test-model")
        load_func.assert_called_once()

    def test_memory_error_raises(self):
        """test safe_model_loading catches memory-related errors."""
        import pytest

        from meetcap.utils.memory import MemoryError

        load_func = Mock(side_effect=Exception("memory allocation failed"))
        with pytest.raises(MemoryError, match="insufficient memory"):
            safe_model_loading(load_func, "test-model")

    def test_non_memory_error_reraises(self):
        """test safe_model_loading re-raises non-memory errors."""
        import pytest

        load_func = Mock(side_effect=ValueError("bad value"))
        with pytest.raises(ValueError, match="bad value"):
            safe_model_loading(load_func, "test-model")


class TestMemoryReportEdgeCases:
    """test memory report edge cases for coverage."""

    def test_empty_report(self):
        """test report with no checkpoints does nothing."""
        monitor = MemoryMonitor()
        # should not raise
        monitor.report()
        monitor.report(detailed=True)
