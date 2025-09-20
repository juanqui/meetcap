#!/usr/bin/env python3
"""
Tests for timer extension and cancellation functionality.
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from meetcap.cli import RecordingOrchestrator
from meetcap.core.hotkeys import HotkeyManager
from meetcap.utils.config import Config


class TestTimerExtension:
    """Test timer extension functionality."""

    @pytest.fixture
    def config(self, temp_dir):
        """Create test config."""
        config = Config(config_path=temp_dir / "config.toml")
        config.config = {
            "paths": {"out_dir": str(temp_dir)},
            "audio": {"sample_rate": 48000, "channels": 2, "preferred_device": ""},
            "hotkey": {"stop": "<cmd>+<shift>+s"},
            "memory": {"enable_monitoring": False},
        }
        return config

    @pytest.fixture
    def orchestrator(self, config):
        """Create orchestrator instance."""
        return RecordingOrchestrator(config)

    def test_extend_timer_basic(self, orchestrator):
        """Test basic timer extension."""
        # Set up initial timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        # Extend timer
        orchestrator.extend_timer(15)

        # Verify operation was queued
        assert not orchestrator.timer_operations_queue.empty()

        # Process the operation
        orchestrator._process_timer_operation()

        # Verify extension
        assert orchestrator.auto_stop_minutes == 45

    def test_extend_timer_no_active_timer(self, orchestrator):
        """Test extending when no timer is active."""
        # Try to extend without active timer
        orchestrator.extend_timer(15)

        # Process the operation (should handle gracefully without errors)
        orchestrator._process_timer_operation()

        # No timer should be set (silent operation)
        assert orchestrator.auto_stop_minutes is None
        assert orchestrator.auto_stop_start_time is None

    def test_cancel_timer(self, orchestrator):
        """Test timer cancellation."""
        # Set up initial timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        # Cancel timer
        orchestrator.cancel_timer()

        # Process the operation
        orchestrator._process_timer_operation()

        # Verify cancellation
        assert orchestrator.auto_stop_minutes is None
        assert orchestrator.auto_stop_start_time is None

    def test_cancel_timer_no_active_timer(self, orchestrator):
        """Test canceling when no timer is active."""
        # Try to cancel without active timer
        orchestrator.cancel_timer()

        # Process the operation (should handle gracefully without errors)
        orchestrator._process_timer_operation()

        # State should remain unchanged (silent operation)
        assert orchestrator.auto_stop_minutes is None
        assert orchestrator.auto_stop_start_time is None

    def test_set_new_timer(self, orchestrator):
        """Test setting a new timer."""
        old_time = time.time() - 100  # Simulate old timer

        # Set up existing timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = old_time

        # Set new timer
        before_set = time.time()
        orchestrator.set_new_timer(60)

        # Process the operation
        orchestrator._process_timer_operation()

        # Verify new timer
        assert orchestrator.auto_stop_minutes == 60
        assert orchestrator.auto_stop_start_time >= before_set

    def test_set_timer_from_scratch(self, orchestrator):
        """Test setting timer when none exists."""
        # Set new timer from scratch
        before_set = time.time()
        orchestrator.set_new_timer(45)

        # Process the operation
        orchestrator._process_timer_operation()

        # Verify new timer
        assert orchestrator.auto_stop_minutes == 45
        assert orchestrator.auto_stop_start_time >= before_set

    def test_get_timer_status_active(self, orchestrator):
        """Test getting timer status when active."""
        start_time = time.time()
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = start_time

        status = orchestrator.get_timer_status()

        assert status["active"] is True
        assert status["duration_minutes"] == 30
        assert status["start_time"] == start_time
        assert status["elapsed_seconds"] >= 0
        assert status["remaining_seconds"] <= 30 * 60

    def test_get_timer_status_inactive(self, orchestrator):
        """Test getting timer status when inactive."""
        status = orchestrator.get_timer_status()

        assert status["active"] is False

    def test_timer_callback_extend(self, orchestrator):
        """Test timer callback for extension."""
        # Set up initial timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        # Call timer callback
        orchestrator._timer_callback("extend", 20)

        # Process operation
        orchestrator._process_timer_operation()

        assert orchestrator.auto_stop_minutes == 50

    def test_timer_callback_cancel(self, orchestrator):
        """Test timer callback for cancellation."""
        # Set up initial timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        # Call timer callback
        orchestrator._timer_callback("cancel")

        # Process operation
        orchestrator._process_timer_operation()

        assert orchestrator.auto_stop_minutes is None

    def test_timer_callback_set(self, orchestrator):
        """Test timer callback for setting new timer."""
        # Call timer callback
        before_set = time.time()
        orchestrator._timer_callback("set", 45)

        # Process operation
        orchestrator._process_timer_operation()

        assert orchestrator.auto_stop_minutes == 45
        assert orchestrator.auto_stop_start_time >= before_set

    def test_timer_callback_menu(self, orchestrator):
        """Test timer callback for showing menu."""
        # Set up timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        with patch.object(orchestrator, "_show_timer_status") as mock_show:
            orchestrator._timer_callback("menu")
            mock_show.assert_called_once()

    def test_timer_callback_defaults(self, orchestrator):
        """Test timer callback with default values."""
        # Set up initial timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        # Test extend with default (30 minutes)
        orchestrator._timer_callback("extend")
        orchestrator._process_timer_operation()
        assert orchestrator.auto_stop_minutes == 60

        # Test set with default (60 minutes)
        orchestrator._timer_callback("set")
        orchestrator._process_timer_operation()
        assert orchestrator.auto_stop_minutes == 60

    def test_timer_callback_error_handling(self, orchestrator):
        """Test timer callback error handling."""
        # Simulate an error in extend_timer
        with patch.object(orchestrator, "extend_timer", side_effect=Exception("Test error")):
            # Should handle error gracefully without crashing
            orchestrator._timer_callback("extend", 30)

        # Error handling should be silent - operation continues without disruption


class TestTimerThreadSafety:
    """Test thread safety of timer operations."""

    @pytest.fixture
    def orchestrator(self, temp_dir):
        """Create orchestrator for threading tests."""
        config = Config(config_path=temp_dir / "config.toml")
        config.config = {
            "paths": {"out_dir": str(temp_dir)},
            "audio": {"sample_rate": 48000, "channels": 2, "preferred_device": ""},
            "hotkey": {"stop": "<cmd>+<shift>+s"},
            "memory": {"enable_monitoring": False},
        }
        return RecordingOrchestrator(config)

    def test_concurrent_timer_operations(self, orchestrator):
        """Test concurrent timer operations don't cause race conditions."""
        # Set up initial timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        # Queue multiple operations concurrently
        operations = [("extend", 15), ("extend", 10), ("cancel", None), ("set", 60), ("extend", 30)]

        # Queue all operations
        for op, value in operations:
            orchestrator.timer_operations_queue.put((op, value))

        # Process all operations
        while not orchestrator.timer_operations_queue.empty():
            orchestrator._process_timer_operation()

        # Should end up with 90 minutes (60 + 30 from last two operations)
        assert orchestrator.auto_stop_minutes == 90

    def test_timer_operations_with_lock(self, orchestrator):
        """Test that timer operations properly use locking."""
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        results = []

        def worker():
            """Worker function that extends timer."""
            for _ in range(5):
                orchestrator.extend_timer(10)
                orchestrator._process_timer_operation()
                results.append(orchestrator.auto_stop_minutes)
                time.sleep(0.01)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Final timer should be 30 + (15 operations * 10 minutes)
        assert orchestrator.auto_stop_minutes == 180

    def test_get_timer_status_thread_safety(self, orchestrator):
        """Test that get_timer_status is thread-safe."""
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        statuses = []

        def status_worker():
            """Worker that gets timer status."""
            for _ in range(10):
                status = orchestrator.get_timer_status()
                statuses.append(status)
                time.sleep(0.01)

        def modify_worker():
            """Worker that modifies timer."""
            for _ in range(5):
                orchestrator.extend_timer(5)
                orchestrator._process_timer_operation()
                time.sleep(0.02)

        # Start both workers
        status_thread = threading.Thread(target=status_worker)
        modify_thread = threading.Thread(target=modify_worker)

        status_thread.start()
        modify_thread.start()

        status_thread.join()
        modify_thread.join()

        # All status calls should have succeeded
        assert len(statuses) == 10
        assert all(isinstance(status, dict) for status in statuses)
        assert all("active" in status for status in statuses)


class TestHotkeyManagerEnhancements:
    """Test HotkeyManager enhancements for timer support."""

    def test_hotkey_manager_init_with_timer_callback(self):
        """Test HotkeyManager initialization with timer callback."""
        stop_callback = Mock()
        timer_callback = Mock()

        manager = HotkeyManager(stop_callback, timer_callback, "<ctrl>+a")

        assert manager.stop_callback == stop_callback
        assert manager.timer_callback == timer_callback
        assert manager.prefix_key == "<ctrl>+a"

    def test_hotkey_manager_init_without_timer_callback(self):
        """Test HotkeyManager initialization without timer callback."""
        stop_callback = Mock()

        manager = HotkeyManager(stop_callback)

        assert manager.stop_callback == stop_callback
        assert manager.timer_callback is None
        assert manager.prefix_key == "<ctrl>+a"  # default

    def test_on_action_key_with_callback(self):
        """Test action key handling after prefix with callback."""
        stop_callback = Mock()
        timer_callback = Mock()

        manager = HotkeyManager(stop_callback, timer_callback, "<ctrl>+a")
        manager._last_trigger = 0  # Reset debounce
        manager._prefix_active = True  # Simulate prefix key pressed
        manager._waiting_for_action = True  # Simulate waiting for action

        manager._on_action_key("e")  # Press 'e' for extend

        timer_callback.assert_called_once_with("extend", 10)

    def test_on_action_key_without_callback(self):
        """Test action key handling without callback."""
        stop_callback = Mock()

        manager = HotkeyManager(stop_callback)
        manager._last_trigger = 0  # Reset debounce
        manager._prefix_active = True  # Simulate prefix key pressed
        manager._waiting_for_action = True  # Simulate waiting for action

        # Should not raise exception
        manager._on_action_key("e")

    def test_action_key_debouncing(self):
        """Test action key debouncing."""
        stop_callback = Mock()
        timer_callback = Mock()

        manager = HotkeyManager(stop_callback, timer_callback)
        manager._prefix_active = True  # Simulate prefix key pressed
        manager._waiting_for_action = True  # Simulate waiting for action

        # First call should work
        manager._on_action_key("e")
        assert timer_callback.call_count == 1

        # Reset prefix for second call
        manager._prefix_active = True
        manager._waiting_for_action = True
        # Immediate second call should be debounced
        manager._on_action_key("e")
        assert timer_callback.call_count == 1  # Still 1

        # After debounce interval, should work again
        manager._last_trigger = 0
        manager._prefix_active = True
        manager._waiting_for_action = True
        manager._on_action_key("e")
        assert timer_callback.call_count == 2

    def test_prefix_key_activation(self):
        """Test prefix key activation and timeout."""
        stop_callback = Mock()
        timer_callback = Mock()

        manager = HotkeyManager(stop_callback, timer_callback)

        # Initially prefix should not be active
        assert not manager._prefix_active

        # Mock the single key listener setup to avoid actually starting it
        with patch.object(manager, "_setup_single_key_listener"):
            # Activate prefix (silent operation)
            manager._on_prefix_key()
            assert manager._prefix_active
            assert manager._waiting_for_action

        # Deactivate manually
        manager._deactivate_prefix()
        assert not manager._prefix_active
        assert not manager._waiting_for_action

    def test_action_key_mapping(self):
        """Test action key to operation mapping."""
        stop_callback = Mock()
        timer_callback = Mock()

        manager = HotkeyManager(stop_callback, timer_callback)
        manager._last_trigger = 0  # Reset debounce

        # Test all action keys individually
        test_cases = [
            ("c", "cancel", None),
            ("e", "extend", 10),
            ("t", "menu", None),
            ("1", "extend", 5),
            ("2", "extend", 10),
            ("3", "extend", 15),
        ]

        for key_char, expected_action, expected_value in test_cases:
            timer_callback.reset_mock()  # Reset mock for each test
            manager._prefix_active = True  # Set prefix active
            manager._waiting_for_action = True  # Set waiting for action
            manager._last_trigger = 0  # Reset debounce for each test

            manager._on_action_key(key_char)
            timer_callback.assert_called_once_with(expected_action, expected_value)

        # Test unknown key (silent operation)
        manager._prefix_active = True
        manager._waiting_for_action = True
        # Unknown key should be handled silently
        manager._on_action_key("x")  # Unknown key - should not raise error


class TestTimerIntegration:
    """Integration tests for timer functionality."""

    @pytest.fixture
    def orchestrator(self, temp_dir):
        """Create orchestrator for integration tests."""
        config = Config(config_path=temp_dir / "config.toml")
        config.config = {
            "paths": {"out_dir": str(temp_dir)},
            "audio": {"sample_rate": 48000, "channels": 2, "preferred_device": ""},
            "hotkey": {"stop": "<cmd>+<shift>+s", "prefix": "<ctrl>+a"},
            "memory": {"enable_monitoring": False},
        }
        return RecordingOrchestrator(config)

    def test_auto_stop_worker_processes_operations(self, orchestrator):
        """Test that auto_stop_worker processes timer operations."""
        # Set up timer
        orchestrator.auto_stop_minutes = 1  # 1 minute for quick test
        orchestrator.auto_stop_start_time = time.time()

        # Queue an extension operation
        orchestrator.extend_timer(1)  # Extend by 1 minute

        # Run worker for a short time
        worker_thread = threading.Thread(target=orchestrator._auto_stop_worker, daemon=True)
        worker_thread.start()

        # Give worker time to process operation
        time.sleep(0.1)

        # Operation should have been processed
        assert orchestrator.timer_operations_queue.empty()
        assert orchestrator.auto_stop_minutes == 2  # Extended from 1 to 2

    def test_show_timer_status(self, orchestrator):
        """Test timer status display."""
        # Test with no active timer
        with patch("meetcap.cli.console") as mock_console:
            orchestrator._show_timer_status()
            mock_console.print.assert_called()
            call_text = mock_console.print.call_args[0][0]
            assert "No timer active" in call_text

        # Test with active timer
        orchestrator.auto_stop_minutes = 30
        orchestrator.auto_stop_start_time = time.time()

        with patch("meetcap.cli.console") as mock_console:
            orchestrator._show_timer_status()
            mock_console.print.assert_called()
            call_text = mock_console.print.call_args[0][0]
            assert "Timer:" in call_text
            assert "30min" in call_text
