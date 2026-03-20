from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ErrorModal(ModalScreen[bool]):
    """modal dialog for displaying errors with context."""

    BINDINGS = [
        ("escape", "dismiss_modal", "Close"),
    ]

    def __init__(
        self,
        error_title: str = "Error",
        error_message: str = "",
        suggestion: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._error_title = error_title
        self._error_message = error_message
        self._suggestion = suggestion

    def compose(self) -> ComposeResult:
        with Vertical(id="error-dialog"):
            yield Label(self._error_title, id="error-title")
            yield Static(self._error_message, id="error-message")
            if self._suggestion:
                yield Static(
                    f"Suggestion: {self._suggestion}",
                    id="error-suggestion",
                )
            yield Button("Close", variant="primary", id="btn-close-error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close-error":
            self.dismiss(True)

    def action_dismiss_modal(self) -> None:
        self.dismiss(True)
