from __future__ import annotations

from textual.widget import Widget
from textual.widgets import Sparkline


class AudioLevelMeter(Widget):
    """audio level meter showing recent peak levels."""

    def __init__(self, max_points: int = 60, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_points = max_points
        self._levels: list[float] = []

    def compose(self):
        yield Sparkline(data=[], id="level-sparkline")

    def update_level(self, level_db: float) -> None:
        """add a new audio level reading (0.0 = max, negative = quieter)."""
        # normalize from dB to 0-100 scale (dB range roughly -60 to 0)
        normalized = max(0.0, min(100.0, (level_db + 60.0) * (100.0 / 60.0)))
        self._levels.append(normalized)
        if len(self._levels) > self._max_points:
            self._levels = self._levels[-self._max_points :]
        sparkline = self.query_one("#level-sparkline", Sparkline)
        sparkline.data = list(self._levels)

    def reset(self) -> None:
        self._levels.clear()
        try:
            sparkline = self.query_one("#level-sparkline", Sparkline)
            sparkline.data = []
        except Exception:
            pass
