"""comprehensive tests for hotkey management and permissions"""

import subprocess
import threading
from unittest.mock import Mock, patch

import pytest

from meetcap.core.hotkeys import HotkeyManager, PermissionChecker


class TestHotkeyManager:
    """test hotkey manager functionality"""

    @pytest.fixture
    def stop_callback(self):
        """create a mock stop callback"""
        return Mock()

    @pytest.fixture
    def hotkey_manager(self, stop_callback):
        """create a hotkey manager instance"""
        return HotkeyManager(stop_callback)

    def test_init(self, stop_callback):
        """test hotkey manager initialization"""
        manager = HotkeyManager(stop_callback)

        assert manager.stop_callback == stop_callback
        assert manager.listener is None
        assert manager._last_trigger == 0.0
        assert manager._debounce_interval == 0.5
        assert isinstance(manager._stop_event, threading.Event)

    def test_on_stop_hotkey_first_press(self, hotkey_manager, stop_callback):
        """test handling first hotkey press"""
        with patch("meetcap.core.hotkeys.console") as mock_console:
            with patch("time.time", return_value=100.0):
                hotkey_manager._on_stop_hotkey()

            assert hotkey_manager._last_trigger == 100.0
            stop_callback.assert_called_once()
            mock_console.print.assert_called_with("\n[yellow]⏹[/yellow] stop hotkey pressed")

    def test_on_stop_hotkey_debouncing(self, hotkey_manager, stop_callback):
        """test hotkey debouncing prevents rapid triggers"""
        # first press
        with patch("time.time", return_value=100.0):
            hotkey_manager._on_stop_hotkey()

        assert stop_callback.call_count == 1

        # rapid second press (within debounce interval)
        with patch("time.time", return_value=100.3):  # 0.3s later
            hotkey_manager._on_stop_hotkey()

        assert stop_callback.call_count == 1  # should not trigger again

        # third press after debounce interval
        with patch("time.time", return_value=100.6):  # 0.6s after first
            hotkey_manager._on_stop_hotkey()

        assert stop_callback.call_count == 2  # should trigger

    def test_start_success(self, hotkey_manager):
        """test successful hotkey listener start"""
        with patch("pynput.keyboard.GlobalHotKeys") as mock_hotkeys:
            mock_listener = Mock()
            mock_hotkeys.return_value = mock_listener

            with patch("meetcap.core.hotkeys.console") as mock_console:
                hotkey_manager.start("<cmd>+<shift>+s")

                assert hotkey_manager.listener == mock_listener
                mock_listener.start.assert_called_once()

                # verify hotkey mapping
                hotkeys_arg = mock_hotkeys.call_args[0][0]
                assert "<cmd>+<shift>+s" in hotkeys_arg
                # callback should be the wrapper function, not the direct method
                callback = hotkeys_arg["<cmd>+<shift>+s"]
                assert callable(callback)
                # test that the wrapper calls our method
                with patch.object(hotkey_manager, "_on_stop_hotkey") as mock_stop:
                    callback()  # call the wrapper
                    mock_stop.assert_called_once()

                # verify console output
                mock_console.print.assert_called_with("[cyan]⌨[/cyan] press ⌘⇧S to stop recording")

    def test_start_already_listening(self, hotkey_manager):
        """test start when already listening"""
        hotkey_manager.listener = Mock()  # simulate active listener

        with patch("pynput.keyboard.GlobalHotKeys") as mock_hotkeys:
            hotkey_manager.start()
            mock_hotkeys.assert_not_called()  # should not create new listener

    def test_start_exception_handling(self, hotkey_manager):
        """test exception handling during start"""
        with patch("pynput.keyboard.GlobalHotKeys") as mock_hotkeys:
            mock_hotkeys.side_effect = Exception("permission denied")

            with patch("meetcap.core.hotkeys.console") as mock_console:
                hotkey_manager.start()

                assert hotkey_manager.listener is None

                # verify error messages
                calls = mock_console.print.call_args_list
                assert any("error setting up hotkey" in str(call) for call in calls)
                assert any("input monitoring permission" in str(call) for call in calls)

    def test_stop_with_listener(self, hotkey_manager):
        """test stopping active listener"""
        mock_listener = Mock()
        hotkey_manager.listener = mock_listener

        hotkey_manager.stop()

        mock_listener.stop.assert_called_once()
        assert hotkey_manager.listener is None

    def test_stop_without_listener(self, hotkey_manager):
        """test stop when no active listener"""
        hotkey_manager.stop()  # should not raise
        assert hotkey_manager.listener is None

    def test_format_hotkey(self, hotkey_manager):
        """test hotkey formatting for display"""
        test_cases = [
            ("<cmd>+<shift>+s", "⌘⇧S"),
            ("<ctrl>+<alt>+d", "⌃⌥D"),
            ("<cmd>+q", "⌘Q"),
            ("<shift>+<cmd>+<alt>+x", "⇧⌘⌥X"),
            ("f1", "F1"),
        ]

        for input_combo, expected in test_cases:
            assert hotkey_manager._format_hotkey(input_combo) == expected

    def test_custom_hotkey_combo(self, hotkey_manager):
        """test using custom hotkey combination"""
        with patch("pynput.keyboard.GlobalHotKeys") as mock_hotkeys:
            mock_listener = Mock()
            mock_hotkeys.return_value = mock_listener

            with patch("meetcap.core.hotkeys.console") as mock_console:
                hotkey_manager.start("<ctrl>+<alt>+q")

                # verify custom hotkey was registered
                hotkeys_arg = mock_hotkeys.call_args[0][0]
                assert "<ctrl>+<alt>+q" in hotkeys_arg

                # verify formatted output
                mock_console.print.assert_called_with("[cyan]⌨[/cyan] press ⌃⌥Q to stop recording")

    def test_lifecycle(self, hotkey_manager):
        """test complete hotkey manager lifecycle"""
        with patch("pynput.keyboard.GlobalHotKeys") as mock_hotkeys:
            mock_listener = Mock()
            mock_hotkeys.return_value = mock_listener

            # start
            hotkey_manager.start()
            assert hotkey_manager.listener is not None

            # trigger hotkey
            hotkeys_arg = mock_hotkeys.call_args[0][0]
            callback = hotkeys_arg["<cmd>+<shift>+s"]

            with patch("time.time", return_value=100.0):
                callback()  # simulate hotkey press

            # stop
            hotkey_manager.stop()
            assert hotkey_manager.listener is None


class TestPermissionChecker:
    """test permission checking functionality"""

    def test_check_microphone_permission_granted(self, mock_subprocess_run):
        """test microphone permission check when granted"""
        mock_result = Mock()
        mock_result.stderr = b"[AVFoundation indev @ 0x7f8b0c704f40] AVFoundation input device"
        mock_subprocess_run.return_value = mock_result

        result = PermissionChecker.check_microphone_permission()

        assert result is True
        mock_subprocess_run.assert_called_once_with(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            timeout=2,
        )

    def test_check_microphone_permission_denied(self, mock_subprocess_run):
        """test microphone permission check when denied"""
        mock_result = Mock()
        mock_result.stderr = b"Error: no permission"
        mock_subprocess_run.return_value = mock_result

        result = PermissionChecker.check_microphone_permission()

        assert result is False

    def test_check_microphone_permission_timeout(self, mock_subprocess_run):
        """test microphone permission check on timeout"""
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 2)

        result = PermissionChecker.check_microphone_permission()

        assert result is False

    def test_check_microphone_permission_exception(self, mock_subprocess_run):
        """test microphone permission check on exception"""
        mock_subprocess_run.side_effect = Exception("unexpected error")

        result = PermissionChecker.check_microphone_permission()

        assert result is False

    def test_show_permission_guide(self):
        """test permission guide display"""
        with patch("meetcap.core.hotkeys.console") as mock_console:
            PermissionChecker.show_permission_guide()

            # verify all sections are displayed
            calls = [str(call) for call in mock_console.print.call_args_list]
            all_output = " ".join(calls).lower()

            # check for key sections
            assert "permissions setup required" in all_output
            assert "microphone access" in all_output
            assert "input monitoring" in all_output
            assert "blackhole audio setup" in all_output
            assert "multi-output device" in all_output
            assert "aggregate input device" in all_output
            assert "meetcap verify" in all_output

    def test_permission_guide_structure(self):
        """test permission guide has proper structure"""
        with patch("meetcap.core.hotkeys.console") as mock_console:
            PermissionChecker.show_permission_guide()

            # count sections
            calls = mock_console.print.call_args_list

            # verify we have multiple sections
            assert len(calls) >= 10  # should have many instruction lines

        # verify tips and formatting
        formatted_calls = [str(call) for call in calls]
        assert any("[cyan]" in call for call in formatted_calls)
        assert any("[green]" in call for call in formatted_calls)
        assert any("[bold yellow]" in call for call in formatted_calls)


class TestHotkeyIntegration:
    """integration tests for hotkey functionality"""

    def test_hotkey_callback_integration(self):
        """test hotkey triggering callback"""
        callback_event = threading.Event()

        def stop_callback():
            callback_event.set()

        manager = HotkeyManager(stop_callback)

        with patch("pynput.keyboard.GlobalHotKeys"):
            manager.start()

            # simulate hotkey press
            with patch("time.time", return_value=100.0):
                manager._on_stop_hotkey()

            assert callback_event.is_set()

            manager.stop()

    def test_multiple_hotkey_managers(self):
        """test multiple hotkey managers don't interfere"""
        callback1 = Mock()
        callback2 = Mock()

        manager1 = HotkeyManager(callback1)
        manager2 = HotkeyManager(callback2)

        with patch("pynput.keyboard.GlobalHotKeys"):
            manager1.start("<cmd>+<shift>+s")
            manager2.start("<cmd>+<shift>+q")

            # trigger first manager
            with patch("time.time", return_value=100.0):
                manager1._on_stop_hotkey()

            assert callback1.call_count == 1
            assert callback2.call_count == 0

            # trigger second manager
            with patch("time.time", return_value=101.0):
                manager2._on_stop_hotkey()

            assert callback1.call_count == 1
            assert callback2.call_count == 1

            manager1.stop()
            manager2.stop()

    def test_compatible_callback_handles_different_signatures(self):
        """test that the compatible callback wrapper handles both old and new pynput signatures"""
        callback_called = False

        def stop_callback():
            nonlocal callback_called
            callback_called = True

        manager = HotkeyManager(stop_callback)
        wrapper = manager._compatible_callback()

        # test old signature (no arguments)
        callback_called = False
        with patch("time.time", return_value=100.0):
            wrapper()
        assert callback_called

        # test new signature (with injected parameter - as seen in newer pynput versions)
        callback_called = False
        with patch("time.time", return_value=101.0):  # different time to avoid debouncing
            wrapper(injected=True)  # this should not cause an error
        assert callback_called

        # test with arbitrary args/kwargs (should be resilient)
        callback_called = False
        with patch("time.time", return_value=102.0):  # different time to avoid debouncing
            wrapper(some_arg=True, another=False)
        assert callback_called

    def test_monkey_patch_on_press_signature_compatibility(self):
        """test that the monkey-patched _on_press method handles signature changes"""

        def stop_callback():
            pass

        manager = HotkeyManager(stop_callback)

        with patch("pynput.keyboard.GlobalHotKeys") as mock_hotkeys_class:
            mock_listener = Mock()
            mock_hotkeys_class.return_value = mock_listener

            # simulate the original _on_press method
            original_on_press = Mock()
            mock_listener._on_press = original_on_press

            # start the manager (this should apply the monkey patch)
            manager.start("<cmd>+<shift>+s")

            # verify the monkey patch was applied
            assert mock_listener._on_press != original_on_press

            # test that our patched method can handle both signatures
            patched_method = mock_listener._on_press

            # test with different signatures that the patch should handle gracefully
            try:
                # these should not raise errors regardless of the original signature
                patched_method("some_key")
                patched_method("some_key", False)
                patched_method("some_key", injected=True)
            except Exception as e:
                # if we get here, our patch isn't working correctly
                raise AssertionError(f"Monkey patch failed to handle signature: {e}") from e
