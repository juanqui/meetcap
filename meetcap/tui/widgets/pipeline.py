from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Label, ProgressBar


@dataclass
class StageInfo:
    """information about a pipeline stage."""

    name: str
    label: str
    status: str = "pending"  # pending, active, done, error
    progress: float = 0.0
    timing: float = 0.0
    detail: str = ""


class StageWidget(Widget):
    """individual pipeline stage display."""

    def __init__(self, stage: StageInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stage = stage

    def compose(self) -> ComposeResult:
        yield Label(
            self._format_label(),
            id=f"stage-label-{self._stage.name}",
        )
        yield ProgressBar(
            total=100,
            show_eta=False,
            id=f"stage-bar-{self._stage.name}",
        )

    def _format_label(self) -> str:
        s = self._stage
        status_icon = {
            "pending": "\u25cb",
            "active": "\u25c9",
            "done": "\u2713",
            "error": "\u2717",
        }.get(s.status, "\u25cb")
        timing = f"  {s.timing:.1f}s" if s.timing > 0 else ""
        detail = f"  {s.detail}" if s.detail else ""
        return f"{status_icon}  {s.label}{timing}{detail}"

    def update_stage(
        self,
        status: str,
        progress: float = 0.0,
        timing: float = 0.0,
        detail: str = "",
    ) -> None:
        self._stage.status = status
        self._stage.progress = progress
        self._stage.timing = timing
        self._stage.detail = detail
        try:
            label = self.query_one(f"#stage-label-{self._stage.name}", Label)
            label.update(self._format_label())
            bar = self.query_one(f"#stage-bar-{self._stage.name}", ProgressBar)
            bar.update(progress=progress)
            if status == "done":
                bar.update(progress=100)
        except Exception:
            pass


class PipelineProgress(Widget):
    """multi-stage processing pipeline progress display."""

    DEFAULT_STAGES = [
        StageInfo(name="stt", label="STT"),
        StageInfo(name="diarization", label="Diarization"),
        StageInfo(name="summarization", label="Summarization"),
        StageInfo(name="organize", label="File Organization"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stages: dict[str, StageInfo] = {}
        self._widgets: dict[str, StageWidget] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="pipeline-stages"):
            for stage_info in self.DEFAULT_STAGES:
                info = StageInfo(name=stage_info.name, label=stage_info.label)
                self._stages[info.name] = info
                widget = StageWidget(info, classes="stage-widget")
                self._widgets[info.name] = widget
                yield widget

    def update_stage(
        self,
        name: str,
        status: str,
        progress: float = 0.0,
        timing: float = 0.0,
        detail: str = "",
    ) -> None:
        if name in self._widgets:
            self._widgets[name].update_stage(status, progress, timing, detail)

    def reset(self) -> None:
        for _name, widget in self._widgets.items():
            widget.update_stage("pending", 0.0, 0.0, "")
