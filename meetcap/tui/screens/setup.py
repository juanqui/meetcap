"""first-run setup wizard screen."""

from __future__ import annotations

import subprocess
import sys

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    Select,
    Static,
)

_STEP_IDS = [
    "step-dependencies",
    "step-audio",
    "step-stt",
    "step-output",
]

_STEP_NAMES = [
    "Check Dependencies",
    "Audio Device",
    "STT Engine",
    "Output Directory",
]


class SetupScreen(Screen):
    """first-run setup wizard."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_step = 0
        self._total_steps = len(_STEP_IDS)

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="setup-container"):
            yield Static("meetcap Setup Wizard", id="setup-title")
            yield Label(
                "Step 1 of 4: Check Dependencies",
                id="setup-step-label",
            )

            # step 1: dependencies
            with Vertical(id="step-dependencies", classes="setup-step"):
                yield Static(
                    "Checking system dependencies...",
                    id="dep-status",
                )
                yield Label("", id="dep-ffmpeg")
                yield Label("", id="dep-python")

            # step 2: audio device
            with Vertical(id="step-audio", classes="setup-step hidden"):
                yield Static("Select Audio Device", id="audio-step-title")
                yield Select(
                    [("Loading devices...", "loading")],
                    id="setup-device-select",
                    allow_blank=False,
                )

            # step 3: stt engine
            with Vertical(id="step-stt", classes="setup-step hidden"):
                yield Static(
                    "Select Speech-to-Text Engine",
                    id="stt-step-title",
                )
                yield Select(
                    [
                        (
                            "Faster Whisper (recommended)",
                            "faster-whisper",
                        ),
                        (
                            "MLX Whisper (Apple Silicon)",
                            "mlx-whisper",
                        ),
                        ("Vosk", "vosk"),
                    ],
                    value="faster-whisper",
                    id="setup-stt-select",
                    allow_blank=False,
                )
                yield Label("", id="setup-model-status")
                yield ProgressBar(
                    total=100,
                    show_eta=False,
                    id="setup-model-progress",
                )

            # step 4: output directory
            with Vertical(id="step-output", classes="setup-step hidden"):
                yield Static("Output Directory", id="output-step-title")
                yield Input(
                    value="~/Recordings/meetcap",
                    id="setup-out-dir",
                )

            # navigation buttons
            with Horizontal(id="setup-nav"):
                yield Button("Back", id="setup-back", variant="default")
                yield Button("Next", id="setup-next", variant="primary")
                yield Button(
                    "Finish",
                    id="setup-finish",
                    variant="success",
                    classes="hidden",
                )
        yield Footer()

    def on_mount(self) -> None:
        """check dependencies on mount."""
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """check if required dependencies are installed."""
        # check ffmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.split("\n")[0]
                self.query_one("#dep-ffmpeg", Label).update(f"FFmpeg: {version[:50]}")
            else:
                self.query_one("#dep-ffmpeg", Label).update(
                    "FFmpeg: not found (install with: brew install ffmpeg)"
                )
        except Exception:
            self.query_one("#dep-ffmpeg", Label).update(
                "FFmpeg: not found (install with: brew install ffmpeg)"
            )

        self.query_one("#dep-python", Label).update(f"Python: {sys.version.split()[0]}")
        self.query_one("#dep-status", Static).update("Dependencies checked")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """handle navigation button presses."""
        if event.button.id == "setup-next":
            self._next_step()
        elif event.button.id == "setup-back":
            self._prev_step()
        elif event.button.id == "setup-finish":
            self._finish_setup()

    def _next_step(self) -> None:
        """advance to the next setup step."""
        if self._current_step >= self._total_steps - 1:
            return
        self._hide_step(self._current_step)
        self._current_step += 1
        self._show_step(self._current_step)
        self._update_step_label()

        # load data for the new step
        if self._current_step == 1:
            self._load_audio_devices()
        elif self._current_step == self._total_steps - 1:
            # show finish button on last step
            try:
                self.query_one("#setup-next").add_class("hidden")
                self.query_one("#setup-finish").remove_class("hidden")
            except Exception:
                pass

    def _prev_step(self) -> None:
        """go back to the previous step."""
        if self._current_step <= 0:
            return
        self._hide_step(self._current_step)
        self._current_step -= 1
        self._show_step(self._current_step)
        self._update_step_label()

        # hide finish, show next
        try:
            self.query_one("#setup-next").remove_class("hidden")
            self.query_one("#setup-finish").add_class("hidden")
        except Exception:
            pass

    def _hide_step(self, step: int) -> None:
        """hide the specified wizard step."""
        if 0 <= step < len(_STEP_IDS):
            try:
                self.query_one(f"#{_STEP_IDS[step]}").add_class("hidden")
            except Exception:
                pass

    def _show_step(self, step: int) -> None:
        """show the specified wizard step."""
        if 0 <= step < len(_STEP_IDS):
            try:
                self.query_one(f"#{_STEP_IDS[step]}").remove_class("hidden")
            except Exception:
                pass

    def _update_step_label(self) -> None:
        """update the step indicator label."""
        name = _STEP_NAMES[self._current_step]
        try:
            self.query_one("#setup-step-label", Label).update(
                f"Step {self._current_step + 1} of {self._total_steps}: {name}"
            )
        except Exception:
            pass

    def _load_audio_devices(self) -> None:
        """load available audio devices into the select widget."""
        try:
            from meetcap.core.devices import list_audio_devices

            devices = list_audio_devices()
            select = self.query_one("#setup-device-select", Select)
            if devices:
                options = [(d.name, d.name) for d in devices]
                select.set_options(options)
            else:
                select.set_options(
                    [
                        ("No devices found", "none"),
                    ]
                )
        except Exception:
            pass

    def _finish_setup(self) -> None:
        """save setup configuration and return to home."""
        try:
            from meetcap.utils.config import Config

            config = Config()

            # save audio device
            try:
                device = str(self.query_one("#setup-device-select", Select).value)
                if device not in ("loading", "none", ""):
                    config.config["audio"]["preferred_device"] = device
            except Exception:
                pass

            # save stt engine
            try:
                stt = str(self.query_one("#setup-stt-select", Select).value)
                config.config["models"]["stt_engine"] = stt
            except Exception:
                pass

            # save output directory
            try:
                out_dir = self.query_one("#setup-out-dir", Input).value
                config.config["paths"]["out_dir"] = out_dir
            except Exception:
                pass

            config.save()
            self.notify("Setup complete!", severity="information")
        except Exception as e:
            self.notify(f"Setup error: {e}", severity="error")

        self.app.pop_screen()

    def action_back(self) -> None:
        """return to previous screen."""
        self.app.pop_screen()
