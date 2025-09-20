"""global hotkey management for recording control"""

import threading
import time
from collections.abc import Callable

# NOTE: pynput may show "This process is not trusted!" warning on macOS
# This warning is harmless and appears even when permissions are correctly granted.
# It comes from deep in the macOS system libraries and cannot be suppressed reliably.
# The hotkey functionality works perfectly regardless of this warning.
from pynput import keyboard
from rich.console import Console

console = Console()


class HotkeyManager:
    """manages global hotkeys for recording control"""

    def __init__(
        self,
        stop_callback: Callable[[], None],
        timer_callback: Callable[[str, int], None] = None,
        prefix_key: str = "<ctrl>+a",
    ):
        """
        initialize hotkey manager.

        args:
            stop_callback: function to call when stop hotkey is pressed
            timer_callback: function to call for timer operations (action, value)
            prefix_key: prefix key combination for timer operations (default: Ctrl+A)
        """
        self.stop_callback = stop_callback
        self.timer_callback = timer_callback
        self.prefix_key = prefix_key
        self.listener: keyboard.GlobalHotKeys | None = None
        self._last_trigger = 0.0
        self._debounce_interval = 0.5  # seconds
        self._stop_event = threading.Event()

        # Prefix key state tracking
        self._prefix_active = False
        self._prefix_timeout = 3.0  # seconds to wait for action key after prefix
        self._prefix_timer = None
        self._waiting_for_action = False  # Flag to distinguish prefix vs action keys

    def _on_stop_hotkey(self) -> None:
        """handle stop hotkey press with debouncing."""
        current_time = time.time()
        if current_time - self._last_trigger < self._debounce_interval:
            return  # ignore rapid repeated presses

        self._last_trigger = current_time
        console.print("\n[yellow]⏹[/yellow] stop hotkey pressed")
        self.stop_callback()

    def _on_prefix_key(self) -> None:
        """handle prefix key press - activate prefix mode."""
        if not self.timer_callback:
            return

        self._prefix_active = True
        self._waiting_for_action = True
        # Silent operation - no console output to avoid disrupting progress display

        # Start timeout timer
        if self._prefix_timer:
            self._prefix_timer.cancel()
        self._prefix_timer = threading.Timer(self._prefix_timeout, self._deactivate_prefix)
        self._prefix_timer.start()

        # Start listening for next single keypress
        self._setup_single_key_listener()

    def _deactivate_prefix(self) -> None:
        """deactivate prefix mode."""
        if self._prefix_active:
            self._prefix_active = False
            self._waiting_for_action = False
            self._cleanup_single_key_listener()
            # Silent timeout - no console output

    def _on_action_key(self, key_char: str) -> None:
        """handle action key press after prefix."""
        if not self._prefix_active or not self.timer_callback or not self._waiting_for_action:
            return

        # Deactivate prefix mode
        self._prefix_active = False
        self._waiting_for_action = False
        self._cleanup_single_key_listener()
        if self._prefix_timer:
            self._prefix_timer.cancel()
            self._prefix_timer = None

        # Map action keys to timer operations
        action_map = {
            "c": ("cancel", None),
            "e": ("extend", 10),  # 10 minutes
            "t": ("menu", None),
            "1": ("extend", 5),  # 5 minutes
            "2": ("extend", 10),  # 10 minutes
            "3": ("extend", 15),  # 15 minutes
        }

        if key_char.lower() in action_map:
            action, value = action_map[key_char.lower()]

            # Apply debouncing
            current_time = time.time()
            if current_time - self._last_trigger < self._debounce_interval:
                return
            self._last_trigger = current_time

            self.timer_callback(action, value)
        else:
            # Silent - unknown keys are ignored without disrupting display
            pass

    def _compatible_callback(self, callback_func):
        """wrapper to handle both old and new pynput callback signatures"""

        def wrapper(*args, **kwargs):
            # call our callback regardless of signature (old or new pynput versions)
            return callback_func()

        return wrapper

    def start(self, hotkey_combo: str = "<cmd>+<shift>+s") -> None:
        """
        start listening for hotkeys.

        args:
            hotkey_combo: hotkey combination string (pynput format)
        """
        if self.listener is not None:
            return  # already listening

        try:
            # Use only GlobalHotKeys for well-defined combinations
            # and handle action keys through a timeout-based approach
            hotkeys = {hotkey_combo: self._compatible_callback(self._on_stop_hotkey)}

            # Add prefix key for timer operations if timer callback is available
            if self.timer_callback:
                hotkeys[self.prefix_key] = self._compatible_callback(self._on_prefix_key)

            self.listener = keyboard.GlobalHotKeys(hotkeys)

            # monkey-patch the _on_press method to handle signature compatibility issues
            original_on_press = self.listener._on_press

            def compatible_on_press(*args, **kwargs):
                # handle both old and new pynput signatures dynamically
                import inspect

                sig = inspect.signature(original_on_press)

                # extract key from args (first argument after self)
                key = args[0] if args else kwargs.get("key")
                injected = args[1] if len(args) > 1 else kwargs.get("injected", False)

                if len(sig.parameters) > 1:  # new signature with injected parameter
                    return original_on_press(key, injected)
                else:  # old signature without injected parameter
                    return original_on_press(key)

            # bind the method to the instance
            import types

            self.listener._on_press = types.MethodType(compatible_on_press, self.listener)

            self.listener.start()

            if self.timer_callback:
                console.print(
                    f"[cyan]⌨[/cyan] hotkeys: {self._format_hotkey(hotkey_combo)}=stop, "
                    f"{self._format_hotkey(self.prefix_key)} then [c=cancel, e=extend, t=timer, 1/2/3=quick]"
                )
            else:
                console.print(
                    f"[cyan]⌨[/cyan] press {self._format_hotkey(hotkey_combo)} to stop recording"
                )

        except Exception as e:
            console.print(f"[red]error setting up hotkey: {e}[/red]")
            console.print(
                "[yellow]tip:[/yellow] grant input monitoring permission in "
                "system preferences > privacy & security > input monitoring"
            )

    def _setup_single_key_listener(self):
        """set up a temporary single key listener for the next action key."""

        def on_single_key_press(key):
            if not self._waiting_for_action:
                return

            try:
                # Handle character keys
                if hasattr(key, "char") and key.char:
                    self._on_action_key(key.char)
                    return False  # Stop this listener
            except AttributeError:
                # Handle special keys (numbers)
                key_name = str(key).replace("Key.", "").replace("_", "")
                if key_name.isdigit():
                    self._on_action_key(key_name)
                    return False  # Stop this listener

        # Start a temporary listener just for the next keypress
        self._single_key_listener = keyboard.Listener(on_press=on_single_key_press)
        self._single_key_listener.start()

    def _cleanup_single_key_listener(self):
        """clean up the single key listener."""
        if hasattr(self, "_single_key_listener") and self._single_key_listener:
            try:
                self._single_key_listener.stop()
            except Exception:
                pass  # Ignore errors during cleanup
            self._single_key_listener = None

    def stop(self) -> None:
        """stop listening for hotkeys."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

        # Stop any active single key listener
        self._cleanup_single_key_listener()

        # Cancel any active prefix timer
        if self._prefix_timer:
            self._prefix_timer.cancel()
            self._prefix_timer = None
        self._prefix_active = False
        self._waiting_for_action = False

    def _format_hotkey(self, combo: str) -> str:
        """
        format hotkey combo for display.

        args:
            combo: pynput hotkey string

        returns:
            human-readable hotkey string
        """
        # convert pynput format to human-readable
        replacements = {
            "<cmd>": "⌘",
            "<shift>": "⇧",
            "<alt>": "⌥",
            "<ctrl>": "⌃",
            "+": "",
        }

        formatted = combo
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)

        return formatted.upper()


class PermissionChecker:
    """check and guide through macos permissions"""

    @staticmethod
    def check_microphone_permission() -> bool:
        """
        check if microphone permission is granted.

        returns:
            true if permission likely granted (heuristic)
        """
        # on macos, we can't directly check permission status
        # but we can try to list audio devices as a proxy
        import subprocess

        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                timeout=2,
            )
            # if we see device listings, permission is likely granted
            return "AVFoundation input device" in result.stderr.decode()
        except Exception:
            return False

    @staticmethod
    def show_permission_guide() -> None:
        """display permission setup guide."""
        console.print("\n[bold yellow]permissions setup required:[/bold yellow]")
        console.print("\n1. [cyan]microphone access:[/cyan]")
        console.print("   system preferences > privacy & security > microphone")
        console.print("   → enable for terminal/iterm")

        console.print("\n2. [cyan]input monitoring (for hotkeys):[/cyan]")
        console.print("   system preferences > privacy & security > input monitoring")
        console.print("   → enable for terminal/iterm")

        console.print("\n3. [cyan]blackhole audio setup:[/cyan]")
        console.print("   a. install blackhole: brew install blackhole-2ch")
        console.print("   b. open audio midi setup")
        console.print("   c. create multi-output device:")
        console.print("      → add built-in output + blackhole")
        console.print("      → use as system output")
        console.print("   d. create aggregate input device:")
        console.print("      → add blackhole + microphone")
        console.print("      → set microphone as clock source")
        console.print("      → enable drift correction")

        console.print("\n[green]tip:[/green] run 'meetcap verify' to check your setup")
