"""comprehensive tests for audio device discovery and management"""

import subprocess
from unittest.mock import Mock

import pytest

from meetcap.core.devices import (
    AudioDevice,
    find_device_by_index,
    find_device_by_name,
    list_audio_devices,
    print_devices,
    select_best_device,
)


class TestAudioDevice:
    """test the audiodevice dataclass"""

    def test_create_standard_device(self):
        """test creating a standard audio device"""
        device = AudioDevice(index=0, name="Built-in Microphone")
        assert device.index == 0
        assert device.name == "Built-in Microphone"
        assert device.is_aggregate is False

    def test_create_aggregate_device(self):
        """test creating an aggregate audio device"""
        device = AudioDevice(index=2, name="Aggregate Device", is_aggregate=True)
        assert device.index == 2
        assert device.name == "Aggregate Device"
        assert device.is_aggregate is True


class TestListAudioDevices:
    """test audio device listing functionality"""

    def test_list_devices_success(self, mock_subprocess_run, mock_ffmpeg_devices):
        """test successful device listing"""
        mock_result = Mock()
        mock_result.stderr = mock_ffmpeg_devices
        mock_subprocess_run.return_value = mock_result

        devices = list_audio_devices()

        assert len(devices) == 3
        assert devices[0].index == 0
        assert devices[0].name == "Built-in Microphone"
        assert devices[0].is_aggregate is False

        assert devices[1].index == 1
        assert devices[1].name == "BlackHole 2ch"
        assert devices[1].is_aggregate is True  # detected by keyword

        assert devices[2].index == 2
        assert devices[2].name == "Aggregate Device (BlackHole + Mic)"
        assert devices[2].is_aggregate is True

        mock_subprocess_run.assert_called_once_with(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_list_devices_empty_output(self, mock_subprocess_run):
        """test handling of empty device output"""
        mock_result = Mock()
        mock_result.stderr = "[AVFoundation indev @ 0x7f8b0c704f40] AVFoundation audio devices:\n"
        mock_subprocess_run.return_value = mock_result

        devices = list_audio_devices()
        assert devices == []

    def test_list_devices_timeout(self, mock_subprocess_run, mock_console):
        """test handling of ffmpeg timeout"""
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 10)

        devices = list_audio_devices()

        assert devices == []
        mock_console.print.assert_called_with("[red]error: ffmpeg device listing timed out[/red]")

    def test_list_devices_ffmpeg_not_found(self, mock_subprocess_run, mock_console):
        """test handling of missing ffmpeg"""
        mock_subprocess_run.side_effect = FileNotFoundError()

        devices = list_audio_devices()

        assert devices == []
        mock_console.print.assert_called_with(
            "[red]error: ffmpeg not found. install with: brew install ffmpeg[/red]"
        )

    def test_list_devices_generic_error(self, mock_subprocess_run, mock_console):
        """test handling of generic errors"""
        mock_subprocess_run.side_effect = Exception("unexpected error")

        devices = list_audio_devices()

        assert devices == []
        mock_console.print.assert_called_with("[red]error listing devices: unexpected error[/red]")

    def test_list_devices_with_video_section(self, mock_subprocess_run):
        """test parsing stops at video devices section"""
        mock_result = Mock()
        mock_result.stderr = """
[AVFoundation indev @ 0x7f8b0c704f40] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8b0c704f40] [0] Microphone
[AVFoundation indev @ 0x7f8b0c704f40] AVFoundation video devices:
[AVFoundation indev @ 0x7f8b0c704f40] [0] Camera
"""
        mock_subprocess_run.return_value = mock_result

        devices = list_audio_devices()

        assert len(devices) == 1
        assert devices[0].name == "Microphone"

    def test_list_devices_aggregate_detection(self, mock_subprocess_run):
        """test aggregate device detection heuristics"""
        mock_result = Mock()
        mock_result.stderr = """
[AVFoundation indev @ 0x7f8b0c704f40] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8b0c704f40] [0] Multi-Output Device
[AVFoundation indev @ 0x7f8b0c704f40] [1] Loopback Audio
[AVFoundation indev @ 0x7f8b0c704f40] [2] Regular Mic
"""
        mock_subprocess_run.return_value = mock_result

        devices = list_audio_devices()

        assert devices[0].is_aggregate is True  # multi keyword
        assert devices[1].is_aggregate is True  # loopback keyword
        assert devices[2].is_aggregate is False  # no keywords


class TestFindDevice:
    """test device finding functionality"""

    @pytest.fixture
    def sample_devices(self):
        """create sample devices for testing"""
        return [
            AudioDevice(0, "Built-in Microphone"),
            AudioDevice(1, "BlackHole 2ch", is_aggregate=True),
            AudioDevice(2, "USB Audio Device"),
        ]

    def test_find_device_by_exact_name(self, sample_devices):
        """test finding device by exact name match"""
        device = find_device_by_name(sample_devices, "BlackHole 2ch")
        assert device is not None
        assert device.index == 1
        assert device.name == "BlackHole 2ch"

    def test_find_device_by_substring(self, sample_devices):
        """test finding device by substring match"""
        device = find_device_by_name(sample_devices, "BlackHole")
        assert device is not None
        assert device.index == 1

    def test_find_device_case_insensitive(self, sample_devices):
        """test case-insensitive device name matching"""
        device = find_device_by_name(sample_devices, "blackhole")
        assert device is not None
        assert device.index == 1

    def test_find_device_not_found(self, sample_devices):
        """test when device is not found"""
        device = find_device_by_name(sample_devices, "Nonexistent Device")
        assert device is None

    def test_find_device_by_index(self, sample_devices):
        """test finding device by index"""
        device = find_device_by_index(sample_devices, 2)
        assert device is not None
        assert device.index == 2
        assert device.name == "USB Audio Device"

    def test_find_device_by_invalid_index(self, sample_devices):
        """test finding device with invalid index"""
        device = find_device_by_index(sample_devices, 99)
        assert device is None

    def test_find_device_empty_list(self):
        """test finding device in empty list"""
        assert find_device_by_name([], "any") is None
        assert find_device_by_index([], 0) is None


class TestPrintDevices:
    """test device printing functionality"""

    def test_print_empty_devices(self, mock_console):
        """test printing when no devices available"""
        print_devices([])
        mock_console.print.assert_called_with("[yellow]no audio input devices found[/yellow]")

    def test_print_devices_with_aggregate(self, mock_console):
        """test printing devices including aggregate device"""
        devices = [
            AudioDevice(0, "Built-in Microphone"),
            AudioDevice(1, "Aggregate Device", is_aggregate=True),
        ]

        print_devices(devices)

        # verify table was created and printed
        assert mock_console.print.call_count >= 2

        # check for aggregate device hint
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("aggregate device detected" in str(call).lower() for call in calls)

    def test_print_devices_without_aggregate(self, mock_console):
        """test printing devices without aggregate device"""
        devices = [
            AudioDevice(0, "Built-in Microphone"),
            AudioDevice(1, "USB Microphone"),
        ]

        print_devices(devices)

        # check for no aggregate device warning
        calls = [str(call) for call in mock_console.print.call_args_list]
        assert any("no aggregate device found" in str(call).lower() for call in calls)


class TestSelectBestDevice:
    """test automatic device selection"""

    def test_select_empty_list(self):
        """test selection from empty device list"""
        assert select_best_device([]) is None

    def test_select_aggregate_device(self):
        """test aggregate device is preferred"""
        devices = [
            AudioDevice(0, "Built-in Microphone"),
            AudioDevice(1, "Aggregate Device", is_aggregate=True),
            AudioDevice(2, "BlackHole 2ch"),
        ]

        best = select_best_device(devices)
        assert best is not None
        assert best.index == 1
        assert best.is_aggregate is True

    def test_select_blackhole_fallback(self):
        """test blackhole is selected when no aggregate"""
        devices = [
            AudioDevice(0, "Built-in Microphone"),
            AudioDevice(1, "BlackHole 2ch"),
            AudioDevice(2, "USB Microphone"),
        ]

        best = select_best_device(devices)
        assert best is not None
        assert best.index == 1
        assert "blackhole" in best.name.lower()

    def test_select_first_as_last_resort(self):
        """test first device is selected as last resort"""
        devices = [
            AudioDevice(0, "USB Microphone"),
            AudioDevice(1, "Built-in Microphone"),
        ]

        best = select_best_device(devices)
        assert best is not None
        assert best.index == 0

    def test_select_multiple_aggregates(self):
        """test first aggregate is selected when multiple exist"""
        devices = [
            AudioDevice(0, "Regular Mic"),
            AudioDevice(1, "Aggregate 1", is_aggregate=True),
            AudioDevice(2, "Aggregate 2", is_aggregate=True),
        ]

        best = select_best_device(devices)
        assert best is not None
        assert best.index == 1
