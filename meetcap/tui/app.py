"""meetcap tui application root."""

from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from meetcap.tui.commands import MeetcapCommands
from meetcap.utils.config import Config

MEETCAP_DARK = Theme(
    name="meetcap-dark",
    primary="#4fc1ff",
    secondary="#7c8fa6",
    accent="#ff6b6b",
    warning="#ffb347",
    error="#ff4444",
    success="#4ade80",
    surface="#1a1e2e",
    panel="#232839",
    dark=True,
)


class MeetcapApp(App):
    """meetcap terminal user interface."""

    TITLE = "meetcap"

    CSS_PATH = [
        "css/theme.tcss",
        "css/home.tcss",
        "css/record.tcss",
        "css/process.tcss",
        "css/history.tcss",
        "css/settings.tcss",
        "css/setup.tcss",
        "css/modals.tcss",
    ]

    ENABLE_COMMAND_PALETTE = True
    COMMANDS = App.COMMANDS | {MeetcapCommands}

    BINDINGS = [
        Binding("r", "push_screen('record')", "record", show=True),
        Binding("h", "push_screen('history')", "history", show=True),
        Binding("s", "push_screen('settings')", "settings", show=True),
        Binding("question_mark", "help", "help", show=True),
        Binding("q", "quit", "quit", show=True),
    ]

    def __init__(
        self,
        initial_screen: str = "home",
        record_args: dict | None = None,
        process_file: Path | None = None,
    ) -> None:
        super().__init__()
        self._initial_screen = initial_screen
        self.record_args = record_args
        self.process_file = process_file
        self.config = Config()

    def on_mount(self) -> None:
        """set up screens, theme, and initial navigation."""
        from meetcap.tui.screens.history import HistoryScreen
        from meetcap.tui.screens.home import HomeScreen
        from meetcap.tui.screens.process import ProcessScreen
        from meetcap.tui.screens.record import RecordScreen
        from meetcap.tui.screens.settings import SettingsScreen
        from meetcap.tui.screens.setup import SetupScreen

        self.install_screen(HomeScreen(), name="home")
        self.install_screen(RecordScreen(), name="record")
        self.install_screen(ProcessScreen(), name="process")
        self.install_screen(HistoryScreen(), name="history")
        self.install_screen(SettingsScreen(), name="settings")
        self.install_screen(SetupScreen(), name="setup")

        self.register_theme(MEETCAP_DARK)
        self.theme = "meetcap-dark"

        self.push_screen("home")

        if not self.config.is_configured():
            self.push_screen("setup")
        elif self._initial_screen != "home":
            self.push_screen(self._initial_screen)

    def action_help(self) -> None:
        """show help information."""
        self.push_screen("setup")
