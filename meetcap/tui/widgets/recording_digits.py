from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Digits


class RecordingDigits(Widget):
    """large elapsed time display for recording screen."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seconds: float = 0.0

    def compose(self) -> ComposeResult:
        yield Digits("00:00:00", id="recording-time")

    def update_time(self, elapsed_seconds: float) -> None:
        """update the displayed time."""
        self._seconds = elapsed_seconds
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        secs = int(elapsed_seconds % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        try:
            digits = self.query_one("#recording-time", Digits)
            digits.update(time_str)
        except Exception:
            pass

    @property
    def elapsed(self) -> float:
        return self._seconds
