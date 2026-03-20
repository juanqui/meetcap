"""home screen — main landing page with readiness checks."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, DataTable, Footer, Header, Label, Static


class QuickActions(Widget):
    """grid of quick action buttons."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="quick-action-buttons"):
            yield Button("[R] Start Recording", id="btn-record", variant="primary")
            yield Button("[H] Browse History", id="btn-history", variant="default")
            yield Button("[S] Settings", id="btn-settings", variant="default")
            yield Button("[Q] Quit", id="btn-quit", variant="error")


class HomeScreen(Screen):
    """home screen - main landing page."""

    BINDINGS = [
        Binding("r", "start_record", "Record", show=True),
        Binding("h", "app.push_screen('history')", "History", show=True),
        Binding("s", "app.push_screen('settings')", "Settings", show=True),
        Binding("escape", "app.quit", "Quit", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="home-container"):
            yield Static("meetcap", id="home-title")
            yield Static("offline meeting capture", id="home-subtitle")
            # readiness banner — hidden when all clear
            yield Label("", id="readiness-banner", classes="hidden")
            yield Button("Run Setup", id="btn-run-setup", variant="warning", classes="hidden")
            yield QuickActions(id="quick-actions")
            yield Static("Recent Recordings", id="recent-title")
            yield DataTable(id="recent-recordings")
            yield Static("System Status", id="status-title")
            from meetcap.tui.widgets.system_status import SystemStatus

            yield SystemStatus(id="system-status")
        yield Footer()

    def on_mount(self) -> None:
        """populate recent recordings and run readiness checks."""
        table = self.query_one("#recent-recordings", DataTable)
        table.add_columns("Date", "Title", "Duration", "Files")
        self._load_recent_recordings(table)
        self._run_readiness_check()

    @work(thread=True)
    def _run_readiness_check(self) -> None:
        """run readiness checks in background thread."""
        try:
            from meetcap.tui.readiness import check_readiness

            result = check_readiness()
            if not result.ready:
                # build error message
                lines = ["⚠ Not ready to record:"]
                for issue in result.errors:
                    lines.append(f"  ✗ {issue.message} — {issue.fix_hint}")
                for issue in result.warnings:
                    lines.append(f"  ! {issue.message} — {issue.fix_hint}")
                msg = "\n".join(lines)
                self.app.call_from_thread(self._show_readiness_banner, msg, True)
                # disable record button
                self.app.call_from_thread(self._set_record_enabled, False)
            elif result.warnings:
                lines = ["⚠ Warnings:"]
                for issue in result.warnings:
                    lines.append(f"  ! {issue.message} — {issue.fix_hint}")
                msg = "\n".join(lines)
                self.app.call_from_thread(self._show_readiness_banner, msg, False)
            else:
                self.app.call_from_thread(self._show_readiness_banner, "✓ All systems ready", False)
        except Exception:
            pass

    def _show_readiness_banner(self, text: str, is_error: bool) -> None:
        """show or update the readiness banner."""
        try:
            banner = self.query_one("#readiness-banner", Label)
            banner.update(text)
            banner.remove_class("hidden")
            if is_error:
                banner.add_class("readiness-error")
                # show setup button
                try:
                    self.query_one("#btn-run-setup", Button).remove_class("hidden")
                except Exception:
                    pass
            else:
                banner.remove_class("readiness-error")
                # hide setup button if no errors
                try:
                    self.query_one("#btn-run-setup", Button).add_class("hidden")
                except Exception:
                    pass
        except Exception:
            pass

    def _set_record_enabled(self, enabled: bool) -> None:
        """enable or disable the record button."""
        try:
            btn = self.query_one("#btn-record", Button)
            btn.disabled = not enabled
        except Exception:
            pass

    def action_start_record(self) -> None:
        """start recording — blocked if not ready."""
        try:
            btn = self.query_one("#btn-record", Button)
            if btn.disabled:
                self.notify(
                    "Cannot record: system not ready. Check warnings above.",
                    severity="error",
                )
                return
        except Exception:
            pass
        self.app.push_screen("record")

    def _load_recent_recordings(self, table: DataTable) -> None:
        """scan recordings directory and populate table."""
        try:
            from meetcap.utils.config import Config

            config = Config()
            out_dir = config.expand_path(config.get("paths", "out_dir", "~/Recordings/meetcap"))
            if not out_dir.exists():
                return

            dirs = sorted(
                [d for d in out_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )[:5]

            for d in dirs:
                name = d.name
                files = []
                for ext in [".opus", ".wav", ".flac"]:
                    if list(d.glob(f"*{ext}")):
                        files.append("a")
                        break
                if list(d.glob("*.transcript.*")):
                    files.append("t")
                if list(d.glob("*.summary.md")):
                    files.append("s")
                if (d / "notes.md").exists():
                    files.append("n")

                table.add_row(
                    name[:16],
                    name[17:] if len(name) > 17 else name,
                    "",
                    " ".join(files),
                    key=str(d),
                )
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """handle quick action button presses."""
        if event.button.id == "btn-record":
            self.action_start_record()
        elif event.button.id == "btn-history":
            self.app.push_screen("history")
        elif event.button.id == "btn-settings":
            self.app.push_screen("settings")
        elif event.button.id == "btn-run-setup":
            self.app.push_screen("setup")
        elif event.button.id == "btn-quit":
            self.app.exit()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """open selected recording in history screen."""
        self.app.push_screen("history")
