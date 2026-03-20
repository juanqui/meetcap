from __future__ import annotations

import shutil
from pathlib import Path

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label


class SystemStatus(Widget):
    """system status panel showing device, stt, llm, and disk info."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._device = "Unknown"
        self._stt_engine = "Unknown"
        self._llm_model = "Unknown"
        self._disk_free = "Unknown"

    def compose(self) -> ComposeResult:
        yield Label("", id="status-content")

    def on_mount(self) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        try:
            from meetcap.utils.config import Config

            config = Config()
            self._device = config.get("audio", "preferred_device", "Unknown")
            self._stt_engine = config.get("models", "stt_engine", "Unknown")
            llm = config.get("models", "llm_model_name", "Unknown")
            self._llm_model = llm.split("/")[-1] if "/" in llm else llm
        except Exception:
            pass

        try:
            out_dir = Path.home() / "Recordings" / "meetcap"
            total, used, free = shutil.disk_usage(out_dir if out_dir.exists() else Path.home())
            self._disk_free = f"{free // (1024**3)} GB free"
        except Exception:
            self._disk_free = "Unknown"

        self._update_display()

    def _update_display(self) -> None:
        text = (
            f"Device: {self._device}"
            f"    STT: {self._stt_engine}\n"
            f"LLM: {self._llm_model}"
            f"    Disk: {self._disk_free}"
        )
        try:
            label = self.query_one("#status-content", Label)
            label.update(text)
        except Exception:
            pass

    def refresh_info(self) -> None:
        self._refresh_status()
