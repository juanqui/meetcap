"""recording history browser screen."""

from __future__ import annotations

import shutil
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Markdown,
    Select,
    Static,
)


class HistoryScreen(Screen):
    """recording history browser."""

    BINDINGS = [
        Binding("r", "reprocess", "Reprocess", show=True),
        Binding("d", "delete", "Delete", show=True),
        Binding("slash", "focus_search", "Search", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._recordings: list[dict] = []
        self._selected_dir: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="history-container"):
            with Horizontal(id="history-controls"):
                yield Input(
                    placeholder="Search recordings...",
                    id="history-search",
                )
                yield Select(
                    [
                        ("Date (newest)", "date_desc"),
                        ("Date (oldest)", "date_asc"),
                        ("Title", "title"),
                    ],
                    value="date_desc",
                    id="history-sort",
                    allow_blank=False,
                )
            yield DataTable(id="history-table", cursor_type="row")
            yield Static("Preview", id="preview-title")
            yield Markdown(
                "*Select a recording to preview*",
                id="summary-preview",
            )
        yield Footer()

    def on_mount(self) -> None:
        """populate the recordings table."""
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Date", "Title", "Duration", "Speakers", "Files")
        self._load_recordings()

    def _load_recordings(self, search_query: str = "") -> None:
        """scan recordings directory and populate table."""
        table = self.query_one("#history-table", DataTable)
        table.clear()
        self._recordings.clear()

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
            )

            for d in dirs:
                name = d.name
                if search_query and search_query.lower() not in name.lower():
                    continue

                # parse recording metadata
                files_present = self._detect_files(d)

                # try to extract date/title from directory name
                parts = name.split("_", 3)
                date_str = "_".join(parts[:3]) if len(parts) >= 3 else name[:10]
                title = parts[3] if len(parts) > 3 else name

                rec_info = {
                    "path": d,
                    "name": name,
                    "date": date_str,
                    "title": title,
                    "files": " ".join(files_present),
                }
                self._recordings.append(rec_info)

                table.add_row(
                    date_str,
                    title[:30],
                    "",
                    "",
                    " ".join(files_present),
                    key=str(d),
                )
        except Exception:
            pass

    @staticmethod
    def _detect_files(directory: Path) -> list[str]:
        """detect which output files exist in a recording dir."""
        files_present: list[str] = []
        for ext in [".opus", ".wav", ".flac"]:
            if list(directory.glob(f"*{ext}")):
                files_present.append("a")
                break
        if list(directory.glob("*.transcript.*")):
            files_present.append("t")
        if list(directory.glob("*.summary.md")):
            files_present.append("s")
        if (directory / "notes.md").exists():
            files_present.append("n")
        return files_present

    def on_input_changed(self, event: Input.Changed) -> None:
        """filter recordings when search input changes."""
        if event.input.id == "history-search":
            self._load_recordings(search_query=event.value)

    def on_select_changed(self, event: Select.Changed) -> None:
        """re-sort recordings when sort order changes."""
        if event.select.id == "history-sort":
            self._apply_sort(str(event.value))

    def _apply_sort(self, sort_key: str) -> None:
        """sort the recordings table by the given key."""
        search = ""
        try:
            search = self.query_one("#history-search", Input).value
        except Exception:
            pass
        self._load_recordings(search_query=search)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """show preview of highlighted recording."""
        if event.row_key is None:
            return
        dir_path = Path(str(event.row_key.value))
        self._selected_dir = dir_path
        self._show_preview(dir_path)

    def _show_preview(self, dir_path: Path) -> None:
        """show summary preview for selected recording."""
        try:
            md_widget = self.query_one("#summary-preview", Markdown)
            summary_files = list(dir_path.glob("*.summary.md"))
            if summary_files:
                content = summary_files[0].read_text(encoding="utf-8")
                md_widget.update(content[:2000])
            else:
                md_widget.update("*No summary available*")
        except Exception:
            pass

    def action_reprocess(self) -> None:
        """reprocess selected recording."""
        if not self._selected_dir:
            return
        # find audio file and push a fresh ProcessScreen
        for ext in [".opus", ".wav", ".flac"]:
            audio_files = list(self._selected_dir.glob(f"*{ext}"))
            if audio_files:
                from meetcap.tui.screens.process import ProcessScreen

                self.app.push_screen(ProcessScreen(audio_path=audio_files[0]))
                return
        self.notify("No audio file found", severity="warning")

    def action_delete(self) -> None:
        """delete selected recording with confirmation."""
        if not self._selected_dir:
            return

        from meetcap.tui.modals.confirm import DeleteConfirmModal

        title = self._selected_dir.name

        def handle_delete(confirmed: bool) -> None:
            if confirmed and self._selected_dir:
                try:
                    shutil.rmtree(self._selected_dir)
                    self.notify(
                        f"Deleted: {title}",
                        severity="information",
                    )
                    self._load_recordings()
                except Exception as e:
                    self.notify(f"Delete failed: {e}", severity="error")

        self.app.push_screen(
            DeleteConfirmModal(recording_title=title),
            callback=handle_delete,
        )

    def action_focus_search(self) -> None:
        """focus the search input."""
        try:
            self.query_one("#history-search", Input).focus()
        except Exception:
            pass

    def action_back(self) -> None:
        """return to previous screen."""
        self.app.pop_screen()
