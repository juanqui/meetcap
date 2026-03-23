from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class StopConfirmModal(ModalScreen[str]):
    """confirmation modal when stopping a recording.

    Returns: "process", "skip", or "cancel"
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="stop-confirm-dialog"):
            yield Label("Stop Recording?", id="stop-confirm-title")
            yield Static(
                "Choose what to do with the recording:",
                id="stop-confirm-body",
            )
            with Horizontal(id="stop-confirm-buttons"):
                yield Button("Stop & Process", variant="primary", id="btn-process")
                yield Button("Stop & Skip", variant="default", id="btn-skip")
                yield Button("Cancel", variant="error", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-process":
            self.dismiss("process")
        elif event.button.id == "btn-skip":
            self.dismiss("skip")
        elif event.button.id == "btn-cancel":
            self.dismiss("cancel")

    def action_cancel(self) -> None:
        self.dismiss("cancel")


class ReprocessModal(ModalScreen[str]):
    """modal for choosing reprocessing mode.

    Returns: "stt" (full), "summary" (summary only), or "cancel"
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, recording_title: str = "", has_transcript: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._recording_title = recording_title
        self._has_transcript = has_transcript

    def compose(self) -> ComposeResult:
        title = self._recording_title or "this recording"
        with Vertical(id="reprocess-dialog"):
            yield Label("Reprocess Recording", id="reprocess-title")
            yield Static(
                f"Reprocess '{title}'?\nChoose what to regenerate:",
                id="reprocess-body",
            )
            with Horizontal(id="reprocess-buttons"):
                yield Button(
                    "Full (STT + Summary)",
                    variant="primary",
                    id="btn-reprocess-full",
                )
                if self._has_transcript:
                    yield Button(
                        "Summary Only",
                        variant="default",
                        id="btn-reprocess-summary",
                    )
                yield Button("Cancel", variant="error", id="btn-reprocess-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-reprocess-full":
            self.dismiss("stt")
        elif event.button.id == "btn-reprocess-summary":
            self.dismiss("summary")
        elif event.button.id == "btn-reprocess-cancel":
            self.dismiss("cancel")

    def action_cancel(self) -> None:
        self.dismiss("cancel")


class DeleteConfirmModal(ModalScreen[bool]):
    """confirmation modal for deleting a recording.

    Returns: True if confirmed, False if cancelled
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, recording_title: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._recording_title = recording_title

    def compose(self) -> ComposeResult:
        title = self._recording_title or "this recording"
        with Vertical(id="delete-confirm-dialog"):
            yield Label("Delete Recording?", id="delete-confirm-title")
            yield Static(
                f"Delete '{title}'? This cannot be undone.",
                id="delete-confirm-body",
            )
            with Horizontal(id="delete-confirm-buttons"):
                yield Button("Delete", variant="error", id="btn-delete")
                yield Button("Cancel", variant="default", id="btn-cancel-delete")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-delete":
            self.dismiss(True)
        elif event.button.id == "btn-cancel-delete":
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)
