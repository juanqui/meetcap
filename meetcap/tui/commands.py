"""command palette provider for meetcap actions."""

from __future__ import annotations

from functools import partial

from textual.command import Hit, Hits, Provider


class MeetcapCommands(Provider):
    """command palette provider for meetcap actions."""

    async def search(self, query: str) -> Hits:
        """search for meetcap commands matching the query."""
        matcher = self.matcher(query)
        commands = [
            ("Start Recording", "record", "begin a new meeting recording"),
            ("Browse History", "history", "view past recordings"),
            ("Open Settings", "settings", "configure meetcap"),
            ("Run Setup Wizard", "setup", "run first-time setup"),
            ("Quit", None, "exit meetcap"),
        ]
        for name, screen, help_text in commands:
            score = matcher.match(name)
            if score > 0:
                if screen:
                    yield Hit(
                        score,
                        matcher.highlight(name),
                        partial(self.app.push_screen, screen),
                        help=help_text,
                    )
                else:
                    yield Hit(
                        score,
                        matcher.highlight(name),
                        self.app.action_quit,
                        help=help_text,
                    )
