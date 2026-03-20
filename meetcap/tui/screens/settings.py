"""settings editor screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import (
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)

# mirrors the actual engine options in cli.py setup wizard
STT_ENGINES = [
    ("Parakeet TDT (recommended, fastest)", "parakeet"),
    ("MLX Whisper (Apple Silicon)", "mlx-whisper"),
    ("Faster Whisper (universal)", "faster-whisper"),
    ("Vosk (offline, speaker ID)", "vosk"),
    ("Whisper.cpp (CLI fallback)", "whispercpp"),
]

# mirrors cli.py step 6 llm model list
LLM_MODELS = [
    ("Qwen3.5-4B (~2.9 GB, default)", "mlx-community/Qwen3.5-4B-MLX-4bit"),
    ("Qwen3.5-9B (~5.6 GB, higher quality)", "mlx-community/Qwen3.5-9B-MLX-4bit"),
]

AUDIO_FORMATS = [
    ("OPUS (recommended, 98% smaller)", "opus"),
    ("FLAC (lossless compression)", "flac"),
    ("WAV (uncompressed)", "wav"),
]

SAMPLE_RATES = [
    ("48000 Hz (recommended)", 48000),
    ("44100 Hz", 44100),
    ("32000 Hz", 32000),
    ("16000 Hz", 16000),
]

TEMPERATURES = [
    ("0.2 (focused)", 0.2),
    ("0.4 (balanced, default)", 0.4),
    ("0.6 (creative)", 0.6),
    ("0.8 (exploratory)", 0.8),
]


class SettingsScreen(Screen):
    """settings editor screen."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=True),
        Binding("escape", "back", "Back", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="settings-container"):
            yield Static("Settings", id="settings-title")

            # audio section
            yield Static("Audio", id="audio-section")
            yield Label("Device:")
            yield Select(
                [("Loading devices...", "")],
                value="",
                id="setting-device",
                allow_blank=False,
            )
            yield Label("Format:")
            yield Select(
                AUDIO_FORMATS,
                value="opus",
                id="setting-format",
                allow_blank=False,
            )
            yield Label("Sample Rate:")
            yield Select(
                SAMPLE_RATES,
                value=48000,
                id="setting-sample-rate",
                allow_blank=False,
            )

            # stt section
            yield Static("Speech-to-Text", id="stt-section")
            yield Label("Engine:")
            yield Select(
                STT_ENGINES,
                value="parakeet",
                id="setting-stt-engine",
                allow_blank=False,
            )

            # llm section
            yield Static("Summarization", id="llm-section")
            yield Label("Model:")
            yield Select(
                LLM_MODELS,
                value="mlx-community/Qwen3.5-4B-MLX-4bit",
                id="setting-llm-model",
                allow_blank=False,
            )
            yield Label("Temperature:")
            yield Select(
                TEMPERATURES,
                value=0.4,
                id="setting-temperature",
                allow_blank=False,
            )

            # diarization section
            yield Static("Diarization", id="diarization-section")
            yield Checkbox(
                "Enable speaker diarization",
                value=True,
                id="setting-diarization",
            )

            # output section
            yield Static("Output", id="output-section")
            yield Label("Directory:")
            yield Input(
                value="~/Recordings/meetcap",
                id="setting-out-dir",
            )
            yield Checkbox(
                "Create notes.md",
                value=True,
                id="setting-notes",
            )
        yield Footer()

    def on_mount(self) -> None:
        """load current settings into form."""
        try:
            from meetcap.utils.config import Config

            self._config = Config()
            self._load_devices()
            self._populate_form()
        except Exception:
            pass

    def _load_devices(self) -> None:
        """populate the device dropdown with real audio devices."""
        try:
            from meetcap.core.devices import list_audio_devices

            devices = list_audio_devices()
            if devices:
                options = [(d.name, d.name) for d in devices]
                self.query_one("#setting-device", Select).set_options(options)
            else:
                self.query_one("#setting-device", Select).set_options([("No devices found", "")])
        except Exception:
            self.query_one("#setting-device", Select).set_options([("Could not list devices", "")])

    def _populate_form(self) -> None:
        """fill form fields from current configuration."""
        c = self._config
        if c is None:
            return

        # device
        device_name = str(c.get("audio", "preferred_device", ""))
        try:
            self.query_one("#setting-device", Select).value = device_name
        except Exception:
            pass

        self.query_one("#setting-format", Select).value = str(c.get("audio", "format", "opus"))

        # sample rate
        rate = c.get("audio", "sample_rate", 48000)
        try:
            self.query_one("#setting-sample-rate", Select).value = int(rate)
        except (ValueError, TypeError):
            pass

        self.query_one("#setting-stt-engine", Select).value = str(
            c.get("models", "stt_engine", "faster-whisper")
        )
        self.query_one("#setting-llm-model", Select).value = str(
            c.get("models", "llm_model_name", "mlx-community/Qwen3.5-4B-MLX-4bit")
        )

        # temperature
        temp = c.get("llm", "temperature", 0.4)
        try:
            self.query_one("#setting-temperature", Select).value = float(temp)
        except (ValueError, TypeError):
            pass

        self.query_one("#setting-diarization", Checkbox).value = bool(
            c.get("models", "enable_speaker_diarization", False)
        )
        self.query_one("#setting-out-dir", Input).value = str(
            c.get("paths", "out_dir", "~/Recordings/meetcap")
        )
        self.query_one("#setting-notes", Checkbox).value = bool(c.get("notes", "enable", True))

    def action_save(self) -> None:
        """save settings to config file."""
        if not self._config:
            from meetcap.utils.config import Config

            self._config = Config()

        c = self._config
        try:
            device_val = self.query_one("#setting-device", Select).value
            if device_val:
                c.config["audio"]["preferred_device"] = str(device_val)

            c.config["audio"]["format"] = str(self.query_one("#setting-format", Select).value)

            rate_val = self.query_one("#setting-sample-rate", Select).value
            if rate_val is not None:
                c.config["audio"]["sample_rate"] = int(rate_val)

            c.config["models"]["stt_engine"] = str(
                self.query_one("#setting-stt-engine", Select).value
            )
            c.config["models"]["llm_model_name"] = str(
                self.query_one("#setting-llm-model", Select).value
            )

            temp_val = self.query_one("#setting-temperature", Select).value
            if temp_val is not None:
                c.config["llm"]["temperature"] = float(temp_val)

            c.config["models"]["enable_speaker_diarization"] = self.query_one(
                "#setting-diarization", Checkbox
            ).value
            c.config["paths"]["out_dir"] = self.query_one("#setting-out-dir", Input).value
            c.config["notes"]["enable"] = self.query_one("#setting-notes", Checkbox).value

            c.save()
            self.notify("Settings saved!", severity="information")

            # run readiness check after save to warn about missing deps/models
            self._check_readiness_after_save()
        except Exception as e:
            self.notify(f"Error saving: {e}", severity="error")

    def _check_readiness_after_save(self) -> None:
        """check if the newly saved settings are ready to use."""
        try:
            from meetcap.tui.readiness import check_readiness

            result = check_readiness()
            for issue in result.errors:
                self.notify(
                    f"{issue.message} — {issue.fix_hint}",
                    severity="error",
                    timeout=10,
                )
            for issue in result.warnings:
                self.notify(
                    f"{issue.message} — {issue.fix_hint}",
                    severity="warning",
                    timeout=8,
                )
            if result.ready and not result.warnings:
                self.notify("All systems ready!", severity="information")
        except Exception:
            pass

    def action_back(self) -> None:
        """return to previous screen."""
        self.app.pop_screen()
