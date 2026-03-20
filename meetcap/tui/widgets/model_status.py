from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label


class ModelStatusIndicator(Widget):
    """indicator showing model download/ready status."""

    def __init__(self, model_name: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_name = model_name
        self._status = "unknown"  # unknown, downloading, ready, missing
        self._size = ""

    def compose(self) -> ComposeResult:
        yield Label(self._format_status(), id="model-status-label")

    def _format_status(self) -> str:
        icons = {
            "unknown": "?",
            "downloading": "\u2193",
            "ready": "\u2713",
            "missing": "\u2717",
        }
        icon = icons.get(self._status, "?")
        size_str = f" ({self._size})" if self._size else ""
        return f"{icon} {self._model_name}{size_str}: {self._status}"

    def set_status(self, status: str, size: str = "") -> None:
        self._status = status
        if size:
            self._size = size
        try:
            label = self.query_one("#model-status-label", Label)
            label.update(self._format_status())
        except Exception:
            pass

    def set_model_name(self, name: str) -> None:
        self._model_name = name
        try:
            label = self.query_one("#model-status-label", Label)
            label.update(self._format_status())
        except Exception:
            pass
