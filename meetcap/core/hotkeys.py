"""global hotkey management for recording control"""

import threading
import time
from collections.abc import Callable

from pynput import keyboard
from rich.console import Console

console = Console()


class HotkeyManager:
    """manages global hotkeys for recording control"""

    def __init__(self, stop_callback: Callable[[], None]):
        """
        initialize hotkey manager.

        args:
            stop_callback: function to call when stop hotkey is pressed
        """
        self.stop_callback = stop_callback
        self.listener: keyboard.GlobalHotKeys | None = None
        self._last_trigger = 0.0
        self._debounce_interval = 0.5  # seconds
        self._stop_event = threading.Event()

    def _on_stop_hotkey(self) -> None:
        """handle stop hotkey press with debouncing."""
        current_time = time.time()
        if current_time - self._last_trigger < self._debounce_interval:
            return  # ignore rapid repeated presses

        self._last_trigger = current_time
        console.print("\n[yellow]⏹[/yellow] stop hotkey pressed")
        self.stop_callback()

    def start(self, hotkey_combo: str = "<cmd>+<shift>+s") -> None:
        """
        start listening for hotkeys.

        args:
            hotkey_combo: hotkey combination string (pynput format)
        """
        if self.listener is not None:
            return  # already listening

        try:
            # create hotkey listener
            hotkeys = {hotkey_combo: self._on_stop_hotkey}

            self.listener = keyboard.GlobalHotKeys(hotkeys)
            self.listener.start()

            console.print(
                f"[cyan]⌨[/cyan] press {self._format_hotkey(hotkey_combo)} to stop recording"
            )

        except Exception as e:
            console.print(f"[red]error setting up hotkey: {e}[/red]")
            console.print(
                "[yellow]tip:[/yellow] grant input monitoring permission in "
                "system preferences > privacy & security > input monitoring"
            )

    def stop(self) -> None:
        """stop listening for hotkeys."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

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
