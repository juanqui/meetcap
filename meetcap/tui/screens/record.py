from __future__ import annotations

import time
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, RichLog, Static


class RecordScreen(Screen):
    """recording screen with live timer and audio info."""

    BINDINGS = [
        Binding("space", "stop_recording", "Stop", show=True),
        Binding("e", "extend_timer", "Extend +30m", show=True),
        Binding("c", "cancel_timer", "Cancel Timer", show=True),
        Binding("escape", "confirm_stop", "Back", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._recording = False
        self._recorder = None
        self._recording_dir: Path | None = None
        self._start_time: float = 0.0
        self._stop_requested = False
        self._timer_seconds: int = 0  # auto-stop timer (0 = disabled)
        self._timer_remaining: float = 0.0
        self._last_timer_display: int = -1  # throttle label updates to 1/s

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="record-container"):
            from meetcap.tui.widgets.recording_digits import (
                RecordingDigits,
            )

            yield RecordingDigits(id="recording-digits")

            yield Static("Audio", id="audio-section-title")
            yield Label("Device: --", id="audio-device")
            yield Label(
                "Format: --  Rate: --  Channels: --",
                id="audio-format",
            )
            from meetcap.tui.widgets.audio_level import AudioLevelMeter

            yield AudioLevelMeter(id="audio-level")

            yield Static("Timer", id="timer-section-title")
            yield Label("Auto-stop: disabled", id="timer-info")

            yield Static("Log", id="log-section-title")
            yield RichLog(id="recording-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        """start recording when screen mounts."""
        # reset all state for fresh recording (supports consecutive recordings)
        self._recording = False
        self._recorder = None
        self._recording_dir = None
        self._start_time = 0.0
        self._stop_requested = False
        self._timer_seconds = 0
        self._timer_remaining = 0.0
        self._last_timer_display = -1

        # reset UI elements
        try:
            self.query_one("#timer-info", Label).update("Auto-stop: disabled")
            from meetcap.tui.widgets.recording_digits import RecordingDigits

            self.query_one("#recording-digits", RecordingDigits).update_time(0.0)
        except Exception:
            pass

        self._log("Preparing to record...")
        self._start_recording_flow()

    def _log(self, message: str) -> None:
        """add a timestamped message to the recording log."""
        timestamp = time.strftime("%H:%M:%S")
        try:
            log = self.query_one("#recording-log", RichLog)
            log.write(f"[dim]{timestamp}[/dim]  {message}")
        except Exception:
            pass

    def _start_recording_flow(self) -> None:
        """initialize and start recording."""
        try:
            from meetcap.utils.config import AudioFormat, Config

            config = Config()

            device_name = config.get("audio", "preferred_device", "default")
            sample_rate = config.get("audio", "sample_rate", 48000)
            channels = config.get("audio", "channels", 2)
            audio_format_str = config.get("audio", "format", "opus")
            opus_bitrate = config.get("audio", "opus_bitrate", 32)
            out_dir = config.expand_path(config.get("paths", "out_dir", "~/Recordings/meetcap"))

            # resolve audio format enum
            try:
                audio_format = AudioFormat(audio_format_str)
            except ValueError:
                audio_format = AudioFormat.OPUS

            # update UI with device info
            try:
                self.query_one("#audio-device", Label).update(f"Device: {device_name}")
                self.query_one("#audio-format", Label).update(
                    f"Format: {audio_format.value.upper()}  "
                    f"Rate: {sample_rate} Hz  Channels: {channels}"
                )
            except Exception:
                pass

            # get record_args from app if available
            record_args = getattr(self.app, "_record_args", {})
            auto_stop = record_args.get("auto_stop", 0)
            if auto_stop and auto_stop > 0:
                self._timer_seconds = auto_stop * 60
                self._timer_remaining = float(self._timer_seconds)
                self._update_timer_label()

            self._log(f"Device: {device_name}")
            self._log(f"Format: {audio_format.value.upper()} @ {sample_rate}Hz")
            self._start_recording_worker(
                out_dir,
                device_name,
                sample_rate,
                channels,
                audio_format,
                opus_bitrate,
            )
        except Exception as e:
            self._log(f"[red]Error: {e}[/red]")

    @work(thread=True)
    def _start_recording_worker(  # pragma: no cover
        self,
        out_dir: Path,
        device_name: str,
        sample_rate: int,
        channels: int,
        audio_format: object,
        opus_bitrate: int,
    ) -> None:
        """run recording in a background thread."""
        try:
            from meetcap.core.devices import (
                find_device_by_name,
                list_audio_devices,
                select_best_device,
            )
            from meetcap.core.recorder import AudioRecorder

            # find device
            devices = list_audio_devices()
            device = find_device_by_name(devices, device_name)
            if not device:
                device = select_best_device(devices)
            if not device:
                self.app.call_from_thread(self._log, "[red]No audio device found[/red]")
                return

            recorder = AudioRecorder(
                output_dir=out_dir,
                sample_rate=sample_rate,
                channels=channels,
            )
            self._recorder = recorder
            self._recording = True
            self._start_time = time.time()
            self._stop_requested = False

            # start recording
            recording_dir = recorder.start_recording(
                device_index=device.index,
                device_name=device.name,
                audio_format=audio_format,
                opus_bitrate=opus_bitrate,
            )
            self._recording_dir = recording_dir
            self.app.call_from_thread(self._log, "[green]Recording started[/green]")

            # start timer update loop
            self.app.call_from_thread(self._start_timer_update)

        except Exception as e:
            self.app.call_from_thread(self._log, f"[red]Recording failed: {e}[/red]")

    def _start_timer_update(self) -> None:
        """start the periodic timer update."""
        self.set_interval(1 / 10, self._update_display)

    def _update_timer_label(self) -> None:
        """update the #timer-info label with current remaining time."""
        try:
            label = self.query_one("#timer-info", Label)
            if self._timer_seconds <= 0:
                label.update("Auto-stop: disabled")
                return

            remaining = max(0, self._timer_remaining)
            mins = int(remaining // 60)
            secs = int(remaining % 60)

            if remaining <= 60:
                label.update(f"Auto-stop: {secs}s remaining")
            else:
                label.update(f"Auto-stop: {mins}:{secs:02d} remaining")
        except Exception:
            pass

    def _update_display(self) -> None:
        """update the timer display (called at 10fps)."""
        if not self._recording:
            return
        elapsed = time.time() - self._start_time
        try:
            from meetcap.tui.widgets.recording_digits import (
                RecordingDigits,
            )

            digits = self.query_one("#recording-digits", RecordingDigits)
            digits.update_time(elapsed)
        except Exception:
            pass

        # check auto-stop timer
        if self._timer_seconds > 0:
            self._timer_remaining = self._timer_seconds - elapsed

            # update countdown label once per second (avoid excessive redraws)
            current_second = int(self._timer_remaining)
            if current_second != self._last_timer_display:
                self._last_timer_display = current_second
                self._update_timer_label()

            if self._timer_remaining <= 0:
                self._log("[yellow]Auto-stop timer reached[/yellow]")
                self.action_stop_recording()

    def action_stop_recording(self) -> None:
        """stop the current recording."""
        if not self._recording:
            return
        self._stop_requested = True
        self._recording = False
        self._log("Stopping recording...")

        if self._recorder:
            try:
                final_path = self._recorder.stop_recording()
                if final_path:
                    self._recording_dir = final_path
                    self._log(f"[green]Recording saved: {final_path.name}[/green]")
                    self._show_stop_confirm()
                else:
                    self._log("[yellow]No recording to save[/yellow]")
                    self.app.pop_screen()
            except Exception as e:
                self._log(f"[red]Error stopping: {e}[/red]")
                self.app.pop_screen()

    def _find_audio_file(self) -> Path | None:
        """find the audio file in the recording directory."""
        if not self._recording_dir or not self._recording_dir.exists():
            return None
        for ext in [".opus", ".wav", ".flac"]:
            files = list(self._recording_dir.glob(f"*{ext}"))
            if files:
                return files[0]
        return None

    def _show_stop_confirm(self) -> None:
        """show the stop confirmation modal."""
        from meetcap.tui.modals.confirm import StopConfirmModal

        def handle_result(result: str) -> None:
            if result == "process":
                audio_file = self._find_audio_file()
                if audio_file:
                    # push a fresh ProcessScreen with the audio path
                    from meetcap.tui.screens.process import ProcessScreen

                    self.app.push_screen(ProcessScreen(audio_path=audio_file))
                else:
                    self._log("[red]No audio file found in recording[/red]")
                    self.app.pop_screen()
            elif result == "skip":
                self.app.pop_screen()
            else:
                # cancel - but recording already stopped
                self.app.pop_screen()

        self.app.push_screen(StopConfirmModal(), callback=handle_result)

    def action_confirm_stop(self) -> None:
        """handle escape key - confirm stop."""
        if self._recording:
            self.action_stop_recording()
        else:
            self.app.pop_screen()

    def action_extend_timer(self) -> None:
        """extend auto-stop timer by 30 minutes."""
        if self._timer_seconds <= 0:
            # enable timer if it was disabled (start from current elapsed)
            elapsed = time.time() - self._start_time if self._start_time else 0
            self._timer_seconds = int(elapsed) + 30 * 60
        else:
            self._timer_seconds += 30 * 60
        self._timer_remaining = self._timer_seconds - (time.time() - self._start_time)
        self._update_timer_label()
        self._log("Timer extended by 30 minutes")

    def action_cancel_timer(self) -> None:
        """cancel auto-stop timer."""
        self._timer_seconds = 0
        self._timer_remaining = 0
        self._last_timer_display = -1
        self._update_timer_label()
        self._log("Timer cancelled")
