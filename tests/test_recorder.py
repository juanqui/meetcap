"""comprehensive tests for audio recording functionality"""

import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from meetcap.core.recorder import AudioRecorder, RecordingSession


class TestRecordingSession:
    """test the recordingsession dataclass"""

    def test_create_recording_session(self):
        """test creating a recording session"""
        mock_process = Mock()
        output_path = Path("/tmp/test.wav")

        session = RecordingSession(
            process=mock_process,
            output_path=output_path,
            start_time=123.45,
            device_name="Test Device",
            sample_rate=48000,
            channels=2,
        )

        assert session.process == mock_process
        assert session.output_path == output_path
        assert session.start_time == 123.45
        assert session.device_name == "Test Device"
        assert session.sample_rate == 48000
        assert session.channels == 2


class TestAudioRecorder:
    """test audio recorder functionality"""

    @pytest.fixture
    def recorder(self, temp_dir):
        """create a recorder instance with temp directory"""
        return AudioRecorder(output_dir=temp_dir)

    @pytest.fixture
    def mock_popen(self):
        """mock subprocess.Popen"""
        with patch("subprocess.Popen") as mock:
            mock_process = Mock()
            mock_process.poll.return_value = None  # process is running
            mock_process.stdin = Mock()
            mock_process.stdout = Mock()
            mock_process.stderr = Mock()
            mock_process.stderr.read.return_value = b""
            mock.return_value = mock_process
            yield mock

    def test_init_default_directory(self):
        """test initialization with default directory"""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            recorder = AudioRecorder()

            expected_path = Path.home() / "Recordings" / "meetcap"
            assert recorder.output_dir == expected_path
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_init_custom_directory(self, temp_dir):
        """test initialization with custom directory"""
        recorder = AudioRecorder(output_dir=temp_dir, sample_rate=44100, channels=1)

        assert recorder.output_dir == temp_dir
        assert recorder.sample_rate == 44100
        assert recorder.channels == 1
        assert recorder.session is None

    def test_start_recording_success(self, recorder, mock_popen, mock_console):
        """test successful recording start"""
        with patch("time.time", return_value=1234567890):
            with patch("time.strftime", return_value="20240101-120000"):
                with patch("pathlib.Path.mkdir"):
                    output_path = recorder.start_recording(
                        device_index=1, device_name="Test Device"
                    )

        assert recorder.session is not None
        assert recorder.session.device_name == "Test Device"
        assert recorder.session.start_time == 1234567890
        assert output_path.name == "20240101-120000-temp"  # now returns directory

        # verify ffmpeg command
        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "ffmpeg" in cmd
        assert ":1" in cmd  # device index
        assert "48000" in str(cmd)  # sample rate
        assert "2" in str(cmd)  # channels

    def test_start_recording_already_recording(self, recorder):
        """test error when already recording"""
        recorder.session = Mock()  # simulate active session

        with pytest.raises(RuntimeError, match="recording already in progress"):
            recorder.start_recording(device_index=0)

    def test_start_recording_ffmpeg_fails(self, recorder, mock_popen):
        """test handling of ffmpeg startup failure"""
        mock_process = mock_popen.return_value
        mock_process.poll.return_value = 1  # process exited
        mock_process.stderr.read.return_value = b"ffmpeg error message"

        with pytest.raises(RuntimeError, match="ffmpeg failed to start"):
            recorder.start_recording(device_index=0)

        assert recorder.session is None

    def test_start_recording_custom_output_path(self, recorder, mock_popen, temp_dir):
        """test recording with custom output path"""
        custom_path = temp_dir / "custom_recording.wav"

        output_path = recorder.start_recording(
            device_index=0, device_name="Test", output_path=custom_path
        )

        # now returns parent directory
        assert output_path == custom_path.parent

    def test_start_dual_recording_success(self, recorder, mock_popen, mock_console):
        """test successful dual device recording"""
        recorder.start_dual_recording(blackhole_index=1, mic_index=2)

        assert recorder.session is not None
        assert "dual" in recorder.session.device_name.lower()

        # verify ffmpeg command has two inputs
        cmd = mock_popen.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert cmd.count("-i") == 2
        assert ":1" in cmd_str  # blackhole
        assert ":2" in cmd_str  # mic
        assert "amix" in cmd_str  # mixing filter

    def test_start_dual_recording_already_recording(self, recorder):
        """test error when already recording (dual)"""
        recorder.session = Mock()

        with pytest.raises(RuntimeError, match="recording already in progress"):
            recorder.start_dual_recording(blackhole_index=0, mic_index=1)

    def test_stop_recording_graceful(self, recorder, temp_dir):
        """test graceful recording stop"""
        # setup mock session
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.wait.return_value = None  # graceful exit

        output_file = temp_dir / "test.wav"
        output_file.write_bytes(b"x" * 100)  # create non-empty file

        recorder.session = RecordingSession(
            process=mock_process,
            output_path=output_file,
            start_time=time.time() - 10,
            device_name="Test",
            sample_rate=48000,
            channels=2,
        )

        result = recorder.stop_recording()

        assert result == output_file.parent  # now returns directory
        assert recorder.session is None
        mock_process.stdin.write.assert_called_with(b"q")
        mock_process.wait.assert_called()

    def test_stop_recording_timeout_terminate(self, recorder, temp_dir):
        """test recording stop with timeout and terminate"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()

        # first wait times out, terminate succeeds
        mock_process.wait.side_effect = [subprocess.TimeoutExpired("ffmpeg", 2.5), None]

        output_file = temp_dir / "test.wav"
        output_file.write_bytes(b"x" * 100)

        recorder.session = RecordingSession(
            process=mock_process,
            output_path=output_file,
            start_time=time.time(),
            device_name="Test",
            sample_rate=48000,
            channels=2,
        )

        result = recorder.stop_recording()

        assert result == output_file.parent  # now returns directory
        mock_process.terminate.assert_called_once()

    def test_stop_recording_force_kill(self, recorder, temp_dir):
        """test recording stop with force kill"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()

        # all waits timeout except final kill
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired("ffmpeg", 2.5),
            subprocess.TimeoutExpired("ffmpeg", 2.5),
            None,
        ]

        output_file = temp_dir / "test.wav"
        output_file.write_bytes(b"x" * 100)

        recorder.session = RecordingSession(
            process=mock_process,
            output_path=output_file,
            start_time=time.time(),
            device_name="Test",
            sample_rate=48000,
            channels=2,
        )

        result = recorder.stop_recording()

        assert result == output_file.parent  # now returns directory
        mock_process.kill.assert_called_once()

    def test_stop_recording_no_session(self, recorder):
        """test stopping when no recording active"""
        result = recorder.stop_recording()
        assert result is None

    def test_stop_recording_empty_file(self, recorder, temp_dir):
        """test handling of empty recording file"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.wait.return_value = None

        output_file = temp_dir / "test.wav"
        output_file.write_bytes(b"x" * 10)  # too small (< 44 bytes)

        recorder.session = RecordingSession(
            process=mock_process,
            output_path=output_file,
            start_time=time.time(),
            device_name="Test",
            sample_rate=48000,
            channels=2,
        )

        with patch("meetcap.core.recorder.console") as mock_console:
            result = recorder.stop_recording()

            assert result is None
            assert not output_file.exists()  # should be deleted
            mock_console.print.assert_called_with(
                "[yellow]⚠[/yellow] recording file is empty or corrupted"
            )

    def test_stop_recording_exception_handling(self, recorder, mock_console):
        """test exception handling during stop"""
        mock_process = Mock()
        mock_process.wait.side_effect = Exception("test error")

        recorder.session = RecordingSession(
            process=mock_process,
            output_path=Path("/nonexistent/test.wav"),
            start_time=time.time(),
            device_name="Test",
            sample_rate=48000,
            channels=2,
        )

        result = recorder.stop_recording()

        assert result is None
        assert recorder.session is None  # should still clean up

    def test_get_elapsed_time(self, recorder):
        """test getting elapsed recording time"""
        assert recorder.get_elapsed_time() == 0.0  # no session

        with patch("time.time", return_value=1000.0):
            recorder.session = RecordingSession(
                process=Mock(),
                output_path=Path("/tmp/test.wav"),
                start_time=990.0,
                device_name="Test",
                sample_rate=48000,
                channels=2,
            )

            assert recorder.get_elapsed_time() == 10.0

    def test_is_recording(self, recorder):
        """test recording status check"""
        assert recorder.is_recording() is False

        recorder.session = Mock()
        assert recorder.is_recording() is True

        recorder.session = None
        assert recorder.is_recording() is False

    def test_show_progress(self, recorder):
        """test progress display"""
        # test no recording
        recorder.show_progress()  # should return immediately

        # test with recording
        recorder.session = Mock()
        recorder._stop_event.set()  # stop immediately

        with patch("meetcap.core.recorder.Progress") as mock_progress:
            recorder.show_progress()
            mock_progress.assert_called_once()

    def test_show_progress_loop(self, recorder):
        """test progress display loop"""
        recorder.session = Mock()

        # simulate stopping after a short time
        def stop_after_delay():
            time.sleep(0.2)
            recorder.session = None

        stop_thread = threading.Thread(target=stop_after_delay)
        stop_thread.start()

        with patch("meetcap.core.recorder.Progress") as mock_progress:
            mock_progress_instance = mock_progress.return_value.__enter__.return_value
            mock_progress_instance.add_task.return_value = 0

            recorder.show_progress()

            # verify progress was updated
            mock_progress_instance.update.assert_called()

        stop_thread.join()

    def test_cleanup(self, recorder):
        """test cleanup on exit"""
        recorder.session = Mock()

        with patch.object(recorder, "stop_recording") as mock_stop:
            recorder.cleanup()
            mock_stop.assert_called_once_with(timeout=2.0)

    def test_cleanup_no_session(self, recorder):
        """test cleanup when no active session"""
        with patch.object(recorder, "stop_recording") as mock_stop:
            recorder.cleanup()
            mock_stop.assert_not_called()  # should not call if no session


class TestRecorderIntegration:
    """integration tests for recorder functionality"""

    @pytest.fixture
    def mock_popen(self):
        """mock subprocess.Popen for integration tests"""
        with patch("subprocess.Popen") as mock:
            yield mock

    def test_recording_lifecycle(self, temp_dir, mock_popen):
        """test complete recording lifecycle"""
        recorder = AudioRecorder(output_dir=temp_dir)

        # set up mock process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.wait.return_value = 0
        mock_process.stderr = Mock()
        mock_process.stderr.read.return_value = b""
        mock_popen.return_value = mock_process

        # start recording
        output_dir = recorder.start_recording(device_index=0, device_name="Test Device")
        assert recorder.is_recording()
        assert recorder.get_elapsed_time() >= 0

        # create mock file in the directory
        output_file = output_dir / "recording.wav"
        output_file.write_bytes(b"x" * 100)

        # stop recording
        result = recorder.stop_recording()
        assert result == output_dir
        assert not recorder.is_recording()
        assert recorder.get_elapsed_time() == 0.0

    def test_dual_recording_lifecycle(self, temp_dir, mock_popen):
        """test dual device recording lifecycle"""
        recorder = AudioRecorder(output_dir=temp_dir)

        # set up mock process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stdin = Mock()
        mock_process.wait.return_value = 0
        mock_process.stderr = Mock()
        mock_process.stderr.read.return_value = b""
        mock_popen.return_value = mock_process

        # start dual recording
        output_dir = recorder.start_dual_recording(blackhole_index=1, mic_index=2)
        assert recorder.is_recording()

        # create mock file in the directory
        output_file = output_dir / "recording.wav"
        output_file.write_bytes(b"x" * 100)

        # stop recording
        result = recorder.stop_recording()
        assert result == output_dir
        assert not recorder.is_recording()
