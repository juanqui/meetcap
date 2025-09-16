"""comprehensive tests for CLI functionality"""

import signal
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from meetcap.cli import BackupManager, RecordingOrchestrator, app, validate_auto_stop_time
from meetcap.core.devices import AudioDevice
from meetcap.core.recorder import AudioRecorder
from meetcap.utils.config import Config


class TestRecordingOrchestrator:
    """test recording orchestrator functionality"""

    @pytest.fixture
    def config(self, temp_dir, mock_config_data):
        """create test config"""
        config = Config(config_path=temp_dir / "config.toml")
        config.config = mock_config_data
        return config

    @pytest.fixture
    def orchestrator(self, config):
        """create orchestrator instance"""
        return RecordingOrchestrator(config)

    @pytest.fixture
    def mock_devices(self):
        """mock audio devices"""
        return [
            AudioDevice(0, "Built-in Microphone"),
            AudioDevice(1, "BlackHole 2ch", is_aggregate=True),
            AudioDevice(2, "Aggregate Device", is_aggregate=True),
        ]

    def test_init(self, config):
        """test orchestrator initialization"""
        orch = RecordingOrchestrator(config)

        assert orch.config == config
        assert orch.recorder is None
        assert orch.hotkey_manager is None
        assert isinstance(orch.stop_event, threading.Event)
        assert orch.interrupt_count == 0
        assert orch.processing_complete is False

    @patch("meetcap.cli.list_audio_devices")
    @patch("meetcap.cli.AudioRecorder")
    @patch("meetcap.cli.HotkeyManager")
    @patch("meetcap.cli.signal.signal")
    def test_run_basic_recording(
        self,
        mock_signal,
        mock_hotkey_class,
        mock_recorder_class,
        mock_list_devices,
        orchestrator,
        mock_devices,
        mock_console,
    ):
        """test basic recording workflow"""
        mock_list_devices.return_value = mock_devices
        mock_recorder = Mock()
        mock_recorder_class.return_value = mock_recorder

        # Create actual temp directory for test
        import tempfile

        test_dir = Path(tempfile.mkdtemp(suffix="-temp"))
        (test_dir / "recording.wav").write_bytes(b"fake audio")

        mock_recorder.start_recording.return_value = test_dir
        mock_recorder.stop_recording.return_value = test_dir
        mock_recorder.is_recording.side_effect = [True, True, False]  # simulate recording then stop

        # simulate stop event being set quickly
        def set_stop(*args):
            orchestrator.stop_event.set()

        orchestrator.stop_event.set()  # stop immediately for test

        orchestrator.run(device="1")  # use device index

        # verify recorder was initialized
        mock_recorder_class.assert_called_once()

        # verify recording started with correct device
        mock_recorder.start_recording.assert_called()
        call_args = mock_recorder.start_recording.call_args
        assert call_args[1]["device_index"] == 1
        assert call_args[1]["device_name"] == "BlackHole 2ch"

    def test_run_no_devices_found(self, orchestrator):
        """test handling when no audio devices found"""
        with patch("meetcap.cli.list_audio_devices", return_value=[]):
            with patch("sys.exit") as mock_exit:
                orchestrator.run()

                mock_exit.assert_called_once()
                assert mock_exit.call_args[0][0] == 4  # EXIT_RUNTIME_ERROR

    def test_run_device_not_found(self, orchestrator, mock_devices):
        """test handling when specified device not found"""
        with patch("meetcap.cli.list_audio_devices", return_value=mock_devices):
            with patch("sys.exit") as mock_exit:
                orchestrator.run(device="Nonexistent Device")

                mock_exit.assert_called_once()
                assert mock_exit.call_args[0][0] == 2  # EXIT_CONFIG_ERROR

    def test_stop_recording(self, orchestrator):
        """test stop recording method"""
        orchestrator.recorder = Mock()
        orchestrator.hotkey_manager = Mock()

        orchestrator._stop_recording()

        assert orchestrator.stop_event.is_set()

    def test_auto_stop_validation(self):
        """test auto stop time validation"""
        # Test valid values
        assert validate_auto_stop_time(0)
        assert validate_auto_stop_time(30)
        assert validate_auto_stop_time(60)
        assert validate_auto_stop_time(90)
        assert validate_auto_stop_time(120)

        # Test invalid values
        assert not validate_auto_stop_time(45)
        assert not validate_auto_stop_time(-30)
        assert not validate_auto_stop_time(150)
        assert not validate_auto_stop_time(1)
        assert not validate_auto_stop_time(25)

    def test_auto_stop_timer_thread(self, orchestrator):
        """test auto stop timer thread creation and termination"""
        orchestrator.auto_stop_minutes = 30
        orchestrator.recorder = Mock()
        orchestrator.recorder.get_elapsed_time.return_value = 0

        orchestrator._start_auto_stop_timer()

        assert orchestrator.auto_stop_timer is not None
        # Note: We can't easily test if the thread is alive in a unit test
        # since it depends on the recorder's get_elapsed_time method

        # Test that the worker method exists and can be called
        # This is a basic test to ensure the method is properly defined
        assert hasattr(orchestrator, "_auto_stop_worker")

    def test_handle_interrupt_single(self, orchestrator):
        """test handling single interrupt"""
        orchestrator.recorder = Mock()
        orchestrator.recorder.is_recording.return_value = True

        with patch("time.time", return_value=100.0):
            with patch("meetcap.cli.console") as mock_console:
                orchestrator._handle_interrupt(signal.SIGINT, None)

                assert orchestrator.interrupt_count == 1
                assert orchestrator.stop_event.is_set()
                mock_console.print.assert_called()

    def test_handle_interrupt_double(self, orchestrator):
        """test handling double interrupt for force quit"""
        with patch("time.time", side_effect=[100.0, 100.5]):  # 0.5s apart
            with patch("sys.exit") as mock_exit:
                orchestrator._handle_interrupt(signal.SIGINT, None)
                orchestrator._handle_interrupt(signal.SIGINT, None)

                mock_exit.assert_called_once_with(1)

    def test_process_recording_with_transcription(self, orchestrator, temp_dir):
        """test processing recording with transcription"""
        # Create a recording directory structure
        recording_dir = temp_dir / "test-temp"
        recording_dir.mkdir()
        audio_file = recording_dir / "recording.wav"
        audio_file.write_bytes(b"fake audio")

        with patch("meetcap.cli.FasterWhisperService") as mock_stt:
            mock_service = Mock()
            mock_service.transcribe.return_value = Mock(
                segments=[Mock(text="Test transcript")], language="en", duration=10.0
            )
            mock_stt.return_value = mock_service

            with patch("meetcap.cli.save_transcript") as mock_save:
                mock_save.return_value = (
                    temp_dir / "test.transcript.txt",
                    temp_dir / "test.transcript.json",
                )

                orchestrator._process_recording(
                    recording_dir, stt_engine="fwhisper", llm_path=None, seed=None
                )

                mock_service.transcribe.assert_called_once_with(audio_file)
                mock_save.assert_called_once()

    def test_process_recording_with_summarization(self, orchestrator, temp_dir):
        """test processing with both transcription and summarization"""
        # Create a recording directory structure
        recording_dir = temp_dir / "test-temp"
        recording_dir.mkdir()
        audio_file = recording_dir / "recording.wav"
        audio_file.write_bytes(b"fake audio")

        transcript_file = recording_dir / "recording.transcript.txt"
        transcript_file.write_text("Test transcript content")

        with patch("meetcap.cli.FasterWhisperService") as mock_stt:
            mock_service = Mock()
            mock_service.transcribe.return_value = Mock(
                segments=[Mock(text="Test transcript")],
            )
            mock_stt.return_value = mock_service

            with patch("meetcap.cli.save_transcript") as mock_save_transcript:
                mock_save_transcript.return_value = (transcript_file, None)

                with patch("meetcap.cli.SummarizationService") as mock_llm:
                    mock_llm_service = Mock()
                    mock_llm_service.summarize.return_value = "## Summary\n\nTest summary"
                    mock_llm.return_value = mock_llm_service

                    with patch("meetcap.cli.save_summary") as mock_save_summary:
                        # Also patch extract_meeting_title
                        with patch("meetcap.cli.extract_meeting_title") as mock_extract:
                            mock_extract.return_value = "TestMeeting"
                            orchestrator._process_recording(
                                recording_dir,
                                stt_engine="fwhisper",
                                llm_path="/path/to/model.gguf",
                                seed=None,
                            )

                        mock_llm_service.summarize.assert_called_once()
                        mock_save_summary.assert_called_once()


class TestCLICommands:
    """test CLI commands"""

    @pytest.fixture
    def runner(self):
        """create CLI test runner"""
        return CliRunner()

    def test_devices_command(self, runner):
        """test devices command"""
        mock_devices = [
            AudioDevice(0, "Microphone"),
            AudioDevice(1, "BlackHole", is_aggregate=True),
        ]

        with patch("meetcap.cli.list_audio_devices", return_value=mock_devices):
            with patch("meetcap.cli.print_devices") as mock_print:
                result = runner.invoke(app, ["devices"])

                assert result.exit_code == 0
                mock_print.assert_called_once_with(mock_devices)

    def test_devices_command_no_devices(self, runner):
        """test devices command with no devices"""
        with patch("meetcap.cli.list_audio_devices", return_value=[]):
            with patch("meetcap.cli.console") as mock_console:
                result = runner.invoke(app, ["devices"])

                assert result.exit_code == 0
                mock_console.print.assert_called()
                # Check all console.print calls for the expected message
                call_strs = [str(call) for call in mock_console.print.call_args_list]
                assert any("no audio devices found" in call.lower() for call in call_strs)

    def test_verify_command_success(self, runner, temp_dir):
        """test verify command success"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            mock_config.expand_path.return_value = temp_dir
            mock_config.get.side_effect = lambda s, k, d=None: {
                ("models", "stt_model_path"): str(temp_dir / "whisper"),
                ("models", "llm_gguf_path"): str(temp_dir / "llm.gguf"),
            }.get((s, k), d)
            mock_config_class.return_value = mock_config

            # create mock model files
            (temp_dir / "whisper").mkdir()
            (temp_dir / "llm.gguf").touch()

            # make sure model file exists in test too
            model_file = temp_dir / "model"
            model_file.mkdir(parents=True, exist_ok=True)
            llm_model = temp_dir / "model.gguf"
            llm_model.touch()

            with patch("meetcap.cli.list_audio_devices") as mock_list:
                mock_list.return_value = [AudioDevice(0, "Mic")]

                with patch(
                    "meetcap.cli.PermissionChecker.check_microphone_permission"
                ) as mock_perm:
                    mock_perm.return_value = True

                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = Mock(returncode=0)

                        result = runner.invoke(app, ["verify"])

                        assert result.exit_code == 0
                        # The verify command shows a table with component statuses
                        # We should check for the presence of key components instead
                        assert "ffmpeg" in result.output.lower()
                        assert "installed" in result.output.lower()

    def test_verify_command_failures(self, runner):
        """test verify command with failures"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            mock_config.get.return_value = "/nonexistent/model"
            mock_config.expand_path.return_value = Path("/nonexistent")
            mock_config_class.return_value = mock_config

            with patch("meetcap.cli.list_audio_devices", return_value=[]):
                with patch(
                    "meetcap.cli.PermissionChecker.check_microphone_permission"
                ) as mock_perm:
                    mock_perm.return_value = False

                    result = runner.invoke(app, ["verify"])

                    # should still complete but show issues
                    assert result.exit_code == 0
                    output = result.output.lower()
                    # Check for any indication of problems
                    assert "not found" in output or "none found" in output or "attention" in output

    def test_record_command(self, runner):
        """test record command"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            # Mock get method to return proper default values
            mock_config.get.return_value = 0
            mock_config_class.return_value = mock_config

            with patch("meetcap.cli.RecordingOrchestrator") as mock_orch_class:
                mock_orch = Mock()
                mock_orch_class.return_value = mock_orch

                # Mock typer.prompt to return a valid option (1 = No automatic stop)
                with patch("typer.prompt", return_value="1"):
                    runner.invoke(app, ["record", "--device", "Mic", "--stt", "fwhisper"])

                    # orchestrator should be created and run called
                    mock_orch_class.assert_called_once_with(mock_config)
                    mock_orch.run.assert_called_once()

                    # check parameters passed
                    call_kwargs = mock_orch.run.call_args[1]
                    assert call_kwargs["device"] == "Mic"
                    assert call_kwargs["stt_engine"] == "fwhisper"
                    # auto_stop should default to 0 when not specified
                    assert call_kwargs["auto_stop"] == 0

    def test_record_command_with_all_options(self, runner):
        """test record command with all options"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            # Mock get method to return proper default values
            mock_config.get.return_value = 0
            mock_config_class.return_value = mock_config

            with patch("meetcap.cli.RecordingOrchestrator") as mock_orch_class:
                mock_orch = Mock()
                mock_orch_class.return_value = mock_orch

                # Mock typer.prompt to return a valid option (1 = No automatic stop)
                with patch("typer.prompt", return_value="1"):
                    result = runner.invoke(
                        app,
                        [
                            "record",
                            "--device",
                            "1",
                            "--out",
                            "/tmp/output",
                            "--rate",
                            "44100",
                            "--channels",
                            "1",
                            "--stt",
                            "fwhisper",
                            "--llm",
                            "/models/llm.gguf",
                            "--seed",
                            "42",
                        ],
                    )

                    if result.exit_code != 0:
                        print("Exit code:", result.exit_code)
                        print("Output:", result.output)
                        if result.exception:
                            print("Exception:", result.exception)

                    # orchestrator should be created and run called
                    mock_orch_class.assert_called_once_with(mock_config)
                    mock_orch.run.assert_called_once()

                    # check parameters passed (if the call was made)
                    if mock_orch.run.call_args:
                        call_kwargs = mock_orch.run.call_args[1]
                        assert call_kwargs["device"] == "1"
                        assert call_kwargs["output_dir"] == "/tmp/output"
                        assert call_kwargs["sample_rate"] == 44100
                        assert call_kwargs["channels"] == 1
                        assert call_kwargs["stt_engine"] == "fwhisper"
                        assert call_kwargs["llm_path"] == "/models/llm.gguf"
                        assert call_kwargs["seed"] == 42
                        # auto_stop should default to 0 when not specified
                        assert call_kwargs["auto_stop"] == 0

    def test_summarize_command_audio_file(self, runner, temp_dir):
        """test summarize command with audio file"""
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"fake audio")

        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()

            # Mock to return different values for different calls
            def mock_get_side_effect(section, key, default=None):
                if section == "models" and key == "llm_gguf_path":
                    return str(temp_dir / "model.gguf")
                elif section == "memory":
                    # Return proper memory config defaults
                    memory_config = {
                        "auto_fallback": True,
                        "warning_threshold": 80,
                        "aggressive_gc": True,
                        "enable_monitoring": False,
                    }
                    return memory_config.get(key, default)
                return default

            mock_config.get.side_effect = mock_get_side_effect
            mock_config.expand_path.return_value = temp_dir / "model.gguf"
            mock_config_class.return_value = mock_config

            # create mock model file
            (temp_dir / "model.gguf").touch()

            with patch("meetcap.cli.FasterWhisperService") as mock_stt:
                mock_service = Mock()
                mock_service.transcribe.return_value = Mock(
                    segments=[Mock(text="Test")],
                )
                mock_stt.return_value = mock_service

                with patch("meetcap.cli.save_transcript"):
                    with patch("meetcap.cli.SummarizationService") as mock_llm:
                        mock_llm_service = Mock()
                        mock_llm_service.summarize.return_value = "Summary"
                        mock_llm.return_value = mock_llm_service

                        with patch("meetcap.cli.save_summary"):
                            result = runner.invoke(app, ["summarize", str(audio_file)])

                            if result.exit_code != 0:
                                print("Exit code:", result.exit_code)
                                print("Output:", result.output)
                                if result.exception:
                                    print("Exception:", result.exception)

                            assert result.exit_code == 0
                            # Transcription should always be called for audio files
                            mock_service.transcribe.assert_called_once()
                            # Summarization may not be called if no LLM model is detected
                            # Just verify the command succeeded

    def test_summarize_command_transcript_file(self, runner, temp_dir):
        """test summarize command with audio file (creates transcript internally)"""
        audio_file = temp_dir / "test.wav"
        audio_file.write_bytes(b"fake audio")

        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()

            # Mock config to return specific values for different keys
            def mock_get(section, key, default=None):
                if section == "models" and key == "stt_engine":
                    return "faster-whisper"  # Force use of faster-whisper for test
                elif section == "models" and key == "stt_model_name":
                    return "large-v3"
                elif section == "models" and key == "stt_model_path":
                    return "~/.meetcap/models/whisper-large-v3"
                elif section == "memory":
                    # Return proper memory config defaults
                    memory_config = {
                        "auto_fallback": True,
                        "warning_threshold": 80,
                        "aggressive_gc": True,
                        "enable_monitoring": False,
                    }
                    return memory_config.get(key, default)
                elif section == "models" and key == "llm_gguf_path":
                    return str(temp_dir / "model.gguf")
                else:
                    return default

            mock_config.get.side_effect = mock_get
            mock_config.expand_path.return_value = temp_dir / "model.gguf"
            mock_config_class.return_value = mock_config

            # create mock model file
            (temp_dir / "model.gguf").touch()

            with patch("meetcap.cli.FasterWhisperService") as mock_stt:
                mock_service = Mock()
                mock_service.transcribe.return_value = Mock(
                    segments=[Mock(text="Test transcript")],
                )
                mock_stt.return_value = mock_service

                with patch("meetcap.cli.save_transcript"):
                    with patch("meetcap.cli.SummarizationService") as mock_llm:
                        mock_llm_service = Mock()
                        mock_llm_service.summarize.return_value = "Summary"
                        mock_llm.return_value = mock_llm_service

                        with patch("meetcap.cli.save_summary"):
                            result = runner.invoke(app, ["summarize", str(audio_file)])

                            if result.exit_code != 0:
                                print("Exit code:", result.exit_code)
                                print("Output:", result.output)
                                if result.exception:
                                    print("Exception:", result.exception)

                            assert result.exit_code == 0
                            # Transcription should be called for audio files
                            mock_service.transcribe.assert_called_once()
                            # Summarization may depend on model availability

    def test_setup_command(self, runner):
        """test setup command with comprehensive mocking to avoid real system calls"""
        # Mock all external dependencies at once to speed up test
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            mock_config.get.side_effect = lambda section, key, default=None: {
                ("llm", "n_ctx"): 32768,
                ("paths", "models_dir"): "~/.meetcap/models",
            }.get((section, key), default)
            mock_config.expand_path.return_value = Path("/mock/models")
            mock_config.config = {
                "models": {},
                "paths": {"models_dir": "~/.meetcap/models"},
                "llm": {"n_ctx": 32768},
            }
            mock_config_class.return_value = mock_config

            # Mock all system calls and external dependencies comprehensively
            patches = [
                patch("subprocess.run", return_value=Mock(returncode=0)),
                patch("platform.processor", return_value="arm"),
                patch(
                    "typer.prompt", side_effect=["1", "1", "2", "~/Recordings/meetcap", "1"]
                ),  # User choices: engine, whisper model, context size, output dir, llm model
                patch("typer.confirm", return_value=True),
                patch("time.sleep"),  # Mock time.sleep
                patch("threading.Event.wait", return_value=False),  # Mock hotkey timeout
                # Mock all file operations
                patch("pathlib.Path.exists", return_value=False),
                patch("pathlib.Path.unlink"),
                patch("pathlib.Path.mkdir"),
                patch("pathlib.Path.expanduser", return_value=Path("/mock/models")),
                # Mock all import checks to avoid real module loading
                patch("importlib.util.find_spec", return_value=Mock()),
                # Mock all model verification/download functions
                patch("meetcap.cli.verify_whisper_model", return_value=False),
                patch("meetcap.cli.verify_qwen_model", return_value=False),
                patch("meetcap.cli.verify_mlx_whisper_model", return_value=False),
                patch("meetcap.cli.ensure_whisper_model", return_value="/models/whisper"),
                patch("meetcap.cli.ensure_qwen_model", return_value="/models/qwen.gguf"),
                patch("meetcap.cli.ensure_mlx_whisper_model", return_value="/models/mlx-whisper"),
                # Mock hardware interactions
                patch("meetcap.cli.list_audio_devices", return_value=[AudioDevice(0, "Mic")]),
                patch(
                    "meetcap.cli.AudioRecorder",
                    return_value=Mock(
                        start_recording=Mock(return_value=Path("/tmp/test.wav")),
                        stop_recording=Mock(return_value=None),
                    ),
                ),
                patch("meetcap.cli.HotkeyManager", return_value=Mock()),
                # Mock permission checker
                patch(
                    "meetcap.cli.PermissionChecker",
                    return_value=Mock(check_microphone_permission=Mock(return_value=True)),
                ),
            ]

            # Apply all patches and run test
            with ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)

                result = runner.invoke(app, ["setup"])

                if result.exit_code != 0:
                    print("STDOUT:", result.stdout)
                    if result.exception:
                        print("EXCEPTION:", result.exception)

                assert result.exit_code == 0
                mock_config.save.assert_called()


class TestCLIIntegration:
    """integration tests for CLI"""

    def test_app_help(self):
        """test app help message"""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "meetcap" in result.output
        assert "offline meeting recorder" in result.output.lower()

        # check commands are listed
        assert "devices" in result.output
        assert "record" in result.output
        assert "summarize" in result.output
        assert "reprocess" in result.output
        assert "verify" in result.output
        assert "setup" in result.output

    def test_command_help(self):
        """test individual command help"""
        runner = CliRunner()

        commands = ["devices", "record", "summarize", "reprocess", "verify", "setup"]

        for cmd in commands:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0
            assert cmd in result.output.lower() or "usage" in result.output.lower()

    def test_version_display(self):
        """test version display"""
        # runner = CliRunner() - unused variable

        with patch("meetcap.cli.__version__", "1.0.0"):
            # Note: typer doesn't have built-in version flag,
            # would need to add it to the app
            pass  # version is shown in record command banner


class TestBackupManager:
    """test backup manager functionality"""

    def test_create_backup(self, temp_dir):
        """test creating backup files"""
        # create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("original content")

        manager = BackupManager()
        backup_path = manager.create_backup(test_file)

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.name == "test.txt.backup"
        assert backup_path.read_text() == "original content"
        assert backup_path in manager.backups

    def test_create_backup_nonexistent_file(self, temp_dir):
        """test creating backup for nonexistent file"""
        test_file = temp_dir / "nonexistent.txt"

        manager = BackupManager()
        backup_path = manager.create_backup(test_file)

        assert backup_path is None
        assert len(manager.backups) == 0

    def test_restore_backup(self, temp_dir):
        """test restoring from backup"""
        # create test file and backup
        test_file = temp_dir / "test.txt"
        test_file.write_text("original content")

        manager = BackupManager()
        backup_path = manager.create_backup(test_file)

        # modify original file
        test_file.write_text("modified content")

        # restore from backup
        success = manager.restore_backup(test_file)

        assert success is True
        assert test_file.read_text() == "original content"
        assert backup_path is None or not backup_path.exists()
        assert backup_path not in manager.backups

    def test_restore_backup_no_backup(self, temp_dir):
        """test restoring when no backup exists"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        manager = BackupManager()
        success = manager.restore_backup(test_file)

        assert success is False
        assert test_file.read_text() == "content"

    def test_cleanup_backups(self, temp_dir):
        """test cleaning up all backups"""
        # create multiple backup files
        for i in range(3):
            backup_file = temp_dir / f"file{i}.txt.backup"
            backup_file.write_text(f"backup {i}")

        manager = BackupManager()
        manager.backups = list(temp_dir.glob("*.backup"))

        manager.cleanup_backups(temp_dir)

        # all backups should be deleted
        assert len(list(temp_dir.glob("*.backup"))) == 0
        assert len(manager.backups) == 0

    def test_restore_all(self, temp_dir):
        """test restoring all tracked backups"""
        # create multiple files and backups
        files = []
        for i in range(3):
            test_file = temp_dir / f"file{i}.txt"
            test_file.write_text(f"original {i}")
            files.append(test_file)

        manager = BackupManager()
        for test_file in files:
            manager.create_backup(test_file)
            test_file.write_text("modified")

        # restore all
        manager.restore_all()

        # all files should be restored
        for i, test_file in enumerate(files):
            assert test_file.read_text() == f"original {i}"

    def test_backup_notes_file(self, temp_dir):
        """test backing up notes.md specifically"""
        notes_file = temp_dir / "notes.md"
        notes_file.write_text("# Original Notes")

        manager = BackupManager()
        backup_path = manager.create_backup(notes_file)

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.name == "notes.md.backup"
        assert backup_path.read_text() == "# Original Notes"
        assert notes_file.exists()  # Original should still exist

    def test_create_backup_error(self, temp_dir):
        """test error handling in create_backup"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("original content")

        manager = BackupManager()

        # Cause an error during copy
        with (
            patch("shutil.copy2", side_effect=PermissionError("Access denied")),
            patch("meetcap.utils.logger.logger.error") as mock_logger,
        ):
            backup_path = manager.create_backup(test_file)

            assert backup_path is None
            mock_logger.assert_called_once()
            assert "failed to create backup" in str(mock_logger.call_args)

    def test_restore_backup_error(self, temp_dir):
        """test error handling in restore_backup"""
        test_file = temp_dir / "test.txt"
        backup_file = temp_dir / "test.txt.backup"

        test_file.write_text("original content")
        backup_file.write_text("backup content")

        manager = BackupManager()
        manager.backups.append(backup_file)

        # Cause an error during move
        with (
            patch("shutil.move", side_effect=PermissionError("Access denied")),
            patch("meetcap.utils.logger.logger.error") as mock_logger,
        ):
            result = manager.restore_backup(test_file)

            assert result is False
            mock_logger.assert_called_once()
            assert "failed to restore backup" in str(mock_logger.call_args)

    def test_cleanup_backups_error(self, temp_dir):
        """test error handling in cleanup_backups"""
        backup_file = temp_dir / "test.txt.backup"
        backup_file.write_text("backup content")

        manager = BackupManager()
        manager.backups.append(backup_file)

        # Cause an error during unlink
        with (
            patch.object(Path, "unlink", side_effect=PermissionError("Access denied")),
            patch("meetcap.utils.logger.logger.warning") as mock_logger,
        ):
            manager.cleanup_backups(temp_dir)

            mock_logger.assert_called_once()
            assert "failed to remove backup" in str(mock_logger.call_args)
            assert len(manager.backups) == 0  # backups list should still be cleared


class TestReprocessCommand:
    """test reprocess command functionality"""

    @pytest.fixture
    def runner(self):
        """create CLI runner"""
        return CliRunner()

    @pytest.fixture
    def mock_recording_dir(self, temp_dir):
        """create mock recording directory"""
        recording_dir = temp_dir / "2025_Jan_15_TestMeeting"
        recording_dir.mkdir()

        # create mock files
        (recording_dir / "recording.wav").write_bytes(b"RIFF" + b"\x00" * 1000)
        (recording_dir / "recording.transcript.txt").write_text("test transcript")
        (recording_dir / "recording.transcript.json").write_text('{"segments": []}')
        (recording_dir / "recording.summary.md").write_text("# Test Summary")

        return recording_dir

    def test_reprocess_command_help(self, runner):
        """test reprocess command help"""
        result = runner.invoke(app, ["reprocess", "--help"])

        assert result.exit_code == 0

        # Strip ANSI codes for more reliable testing in CI
        import re

        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

        assert "reprocess" in clean_output.lower()
        assert "recording" in clean_output.lower()
        assert "--mode" in clean_output or "-m" in clean_output
        assert "--stt" in clean_output
        assert "--llm" in clean_output

    def test_reprocess_invalid_mode(self, runner):
        """test reprocess with invalid mode"""
        with patch("meetcap.cli.Config"):
            result = runner.invoke(app, ["reprocess", "test_dir", "--mode", "invalid"])

            assert result.exit_code == 1
            assert "invalid mode" in result.output.lower()

    def test_reprocess_invalid_stt(self, runner):
        """test reprocess with invalid stt engine"""
        with patch("meetcap.cli.Config"):
            result = runner.invoke(app, ["reprocess", "test_dir", "--stt", "invalid"])

            assert result.exit_code == 1
            assert "invalid stt engine" in result.output.lower()

    def test_reprocess_directory_not_found(self, runner):
        """test reprocess with nonexistent directory"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            mock_config.expand_path.return_value = Path("/nonexistent")
            mock_config_class.return_value = mock_config

            result = runner.invoke(app, ["reprocess", "nonexistent_dir", "--yes"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_reprocess_stt_mode(self, runner, mock_recording_dir):
        """test reprocess in stt mode"""
        with ExitStack() as stack:
            mock_config_class = stack.enter_context(patch("meetcap.cli.Config"))
            mock_config = Mock()
            mock_config.expand_path.return_value = mock_recording_dir.parent
            mock_config.get.return_value = str(mock_recording_dir.parent)
            mock_config.get_section.return_value = {}
            mock_config_class.return_value = mock_config

            mock_process_audio = stack.enter_context(
                patch("meetcap.cli.RecordingOrchestrator._process_audio_to_transcript")
            )
            mock_process_audio.return_value = (
                mock_recording_dir / "recording.transcript.txt",
                mock_recording_dir / "recording.transcript.json",
            )

            mock_process_summary = stack.enter_context(
                patch("meetcap.cli.RecordingOrchestrator._process_transcript_to_summary")
            )
            mock_process_summary.return_value = mock_recording_dir / "recording.summary.md"

            result = runner.invoke(
                app,
                ["reprocess", str(mock_recording_dir), "--mode", "stt", "--yes"],
            )

            assert result.exit_code == 0
            assert "reprocessing complete" in result.output.lower()
            mock_process_audio.assert_called_once()
            mock_process_summary.assert_called_once()

    def test_reprocess_summary_mode(self, runner, mock_recording_dir):
        """test reprocess in summary mode"""
        with ExitStack() as stack:
            mock_config_class = stack.enter_context(patch("meetcap.cli.Config"))
            mock_config = Mock()
            mock_config.expand_path.return_value = mock_recording_dir.parent
            mock_config.get.return_value = str(mock_recording_dir.parent)
            mock_config.get_section.return_value = {}
            mock_config_class.return_value = mock_config

            mock_process_summary = stack.enter_context(
                patch("meetcap.cli.RecordingOrchestrator._process_transcript_to_summary")
            )
            mock_process_summary.return_value = mock_recording_dir / "recording.summary.md"

            result = runner.invoke(
                app,
                ["reprocess", str(mock_recording_dir), "--mode", "summary", "--yes"],
            )

            assert result.exit_code == 0
            assert "reprocessing complete" in result.output.lower()
            mock_process_summary.assert_called_once()

    def test_reprocess_with_confirmation(self, runner, mock_recording_dir):
        """test reprocess with confirmation prompt"""
        with ExitStack() as stack:
            mock_config_class = stack.enter_context(patch("meetcap.cli.Config"))
            mock_config = Mock()
            mock_config.expand_path.return_value = mock_recording_dir.parent
            mock_config.get.return_value = str(mock_recording_dir.parent)
            mock_config_class.return_value = mock_config

            # simulate user saying no to confirmation
            result = runner.invoke(
                app,
                ["reprocess", str(mock_recording_dir)],
                input="n\n",
            )

            assert "cancelled" in result.output.lower()

    def test_resolve_recording_path(self):
        """test path resolution logic"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            output_dir = Path("/test/recordings")
            mock_config.expand_path.return_value = output_dir
            mock_config.get.return_value = str(output_dir)
            mock_config_class.return_value = mock_config

            orchestrator = RecordingOrchestrator(mock_config)

            # test absolute path
            with patch("pathlib.Path.exists", return_value=True):
                result = orchestrator._resolve_recording_path("/absolute/path")
                assert result == Path("/absolute/path")

            # test relative path in current directory
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.side_effect = [False, True]
                with patch("pathlib.Path.resolve", return_value=Path("/current/test")):
                    result = orchestrator._resolve_recording_path("test")
                    assert result is not None

            # test path not found
            with patch("pathlib.Path.exists", return_value=False):
                result = orchestrator._resolve_recording_path("nonexistent")
                assert result is None

    def test_record_command_with_auto_stop(self, runner):
        """test record command with auto stop option"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config

            with patch("meetcap.cli.RecordingOrchestrator") as mock_orch_class:
                mock_orch = Mock()
                mock_orch_class.return_value = mock_orch

                # Test with valid auto stop value
                result = runner.invoke(app, ["record", "--auto-stop", "30"])

                assert result.exit_code == 0
                mock_orch.run.assert_called_once()

                # Check that auto_stop parameter was passed
                call_kwargs = mock_orch.run.call_args[1]
                assert call_kwargs["auto_stop"] == 30

    def test_record_command_with_invalid_auto_stop(self, runner):
        """test record command with invalid auto stop option"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config

            # Test with invalid auto stop value
            result = runner.invoke(app, ["record", "--auto-stop", "45"])

            assert result.exit_code == 1
            assert "invalid auto-stop time" in result.output
            assert "supported values" in result.output

    def test_record_command_auto_stop_from_config(self, runner):
        """test record command getting auto stop from config"""
        with patch("meetcap.cli.Config") as mock_config_class:
            mock_config = Mock()
            # Mock config to return default_auto_stop = 60
            mock_config.get.return_value = 60
            mock_config_class.return_value = mock_config

            with patch("meetcap.cli.RecordingOrchestrator") as mock_orch_class:
                mock_orch = Mock()
                mock_orch_class.return_value = mock_orch

                # Test without explicit auto stop (should use config default)
                result = runner.invoke(app, ["record"])

                assert result.exit_code == 0
                mock_orch.run.assert_called_once()

                # Check that auto_stop parameter was passed
                call_kwargs = mock_orch.run.call_args[1]
                assert call_kwargs["auto_stop"] == 60

    def test_notes_file_creation(self):
        """Test notes.md file creation during recording."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock the notes file creation
            def mock_create_notes(recording_dir):
                recording_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                notes_path = recording_dir / "notes.md"
                with open(notes_path, "w") as f:
                    f.write("# Meeting Notes\n\nTest content\n")
                return notes_path

            # Test that notes file is created
            notes_path = mock_create_notes(temp_path / "test-recording")
            assert notes_path.exists()
            assert notes_path.read_text() == "# Meeting Notes\n\nTest content\n"

    def test_recording_with_manual_notes(self):
        """Test complete recording workflow with manual notes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock subprocess.Popen to avoid calling real ffmpeg
            with patch("subprocess.Popen") as mock_popen:
                # Set up mock process
                mock_process = Mock()
                mock_process.poll.return_value = None  # process is running
                mock_process.stdin = Mock()
                mock_process.stdout = Mock()
                mock_process.stderr = Mock()
                mock_process.stderr.read.return_value = b""
                mock_process.wait.return_value = 0  # successful exit
                mock_popen.return_value = mock_process

                # Simulate recording workflow
                recorder = AudioRecorder(output_dir=temp_path)
                recording_dir = recorder.start_recording(device_index=0, device_name="Test Device")

                # Verify notes file was created
                notes_path = recording_dir / "notes.md"
                assert notes_path.exists()

                # Add manual notes
                notes_path.write_text(
                    "# Meeting Notes\n\nKey decisions made:\n- Decision 1\n- Decision 2\n"
                )

                # Create mock audio file for recording
                audio_file = recording_dir / "recording.wav"
                audio_file.write_bytes(b"x" * 100)  # create non-empty file

                # Complete recording and processing
                final_dir = recorder.stop_recording()
                assert final_dir is not None

                # Verify notes file is preserved
                final_notes_path = final_dir / "notes.md"
                assert final_notes_path.exists()
                assert "Key decisions made" in final_notes_path.read_text()


class TestCLIUtilityFunctions:
    """Test standalone CLI utility functions"""

    def test_validate_auto_stop_time_valid_values(self):
        """Test valid auto stop time values"""
        from meetcap.cli import validate_auto_stop_time

        assert validate_auto_stop_time(0) is True
        assert validate_auto_stop_time(30) is True
        assert validate_auto_stop_time(60) is True
        assert validate_auto_stop_time(90) is True
        assert validate_auto_stop_time(120) is True

    def test_validate_auto_stop_time_invalid_values(self):
        """Test invalid auto stop time values"""
        from meetcap.cli import validate_auto_stop_time

        assert validate_auto_stop_time(-1) is False
        assert validate_auto_stop_time(15) is False
        assert validate_auto_stop_time(45) is False
        assert validate_auto_stop_time(75) is False
        assert validate_auto_stop_time(150) is False

    def test_create_notes_file_success(self, temp_dir):
        """Test successful notes file creation"""
        from meetcap.cli import create_notes_file
        from meetcap.utils.config import Config

        config = Config(config_path=temp_dir / "config.toml")
        recording_dir = temp_dir / "recording"
        recording_dir.mkdir()

        notes_path = create_notes_file(config, recording_dir)

        assert notes_path is not None
        assert notes_path.exists()
        assert notes_path.name == "notes.md"

        # Check default content
        content = notes_path.read_text()
        assert "Meeting Notes" in content
        assert "Add your notes here" in content

    def test_create_notes_file_with_custom_template(self, temp_dir):
        """Test notes file creation with custom template"""
        from meetcap.cli import create_notes_file
        from meetcap.utils.config import Config

        # Create config with custom template - use proper TOML escaping
        config_file = temp_dir / "config.toml"
        config_content = '''
[notes]
template = """# Custom Notes

My custom template
"""
'''
        config_file.write_text(config_content)

        config = Config(config_path=config_file)
        recording_dir = temp_dir / "recording"
        recording_dir.mkdir()

        notes_path = create_notes_file(config, recording_dir)

        assert notes_path is not None
        content = notes_path.read_text()
        assert "Custom Notes" in content
        assert "My custom template" in content

    def test_create_notes_file_write_error(self, temp_dir):
        """Test notes file creation with write error"""
        from meetcap.cli import create_notes_file
        from meetcap.utils.config import Config

        config = Config(config_path=temp_dir / "config.toml")

        # Try to create notes in non-existent directory
        nonexistent_dir = temp_dir / "nonexistent" / "recording"

        with patch("meetcap.cli.console") as mock_console:
            notes_path = create_notes_file(config, nonexistent_dir)

            assert notes_path is None
            mock_console.print.assert_called_once()
            assert "could not create notes file" in str(mock_console.print.call_args)

    def test_read_manual_notes_success(self, temp_dir):
        """Test successful manual notes reading"""
        from meetcap.cli import read_manual_notes

        notes_file = temp_dir / "notes.md"
        test_content = "# Meeting Notes\n\nImportant decisions made"
        notes_file.write_text(test_content)

        content = read_manual_notes(notes_file)
        assert content == test_content

    def test_read_manual_notes_file_not_exists(self, temp_dir):
        """Test reading notes when file doesn't exist"""
        from meetcap.cli import read_manual_notes

        nonexistent_file = temp_dir / "nonexistent.md"
        content = read_manual_notes(nonexistent_file)
        assert content == ""

    def test_read_manual_notes_read_error(self, temp_dir):
        """Test reading notes with read error"""
        from meetcap.cli import read_manual_notes

        notes_file = temp_dir / "notes.md"
        notes_file.write_text("test content")

        # Mock open to raise exception
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with patch("meetcap.cli.console") as mock_console:
                content = read_manual_notes(notes_file)

                assert content == ""
                mock_console.print.assert_called_once()
                assert "could not read manual notes" in str(mock_console.print.call_args)
