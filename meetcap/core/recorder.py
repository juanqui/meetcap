"""audio recording via ffmpeg subprocess"""

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


@dataclass
class RecordingSession:
    """represents an active recording session"""

    process: subprocess.Popen
    output_path: Path
    start_time: float
    device_name: str
    sample_rate: int
    channels: int


class AudioRecorder:
    """manages ffmpeg subprocess for audio capture"""

    def __init__(
        self,
        output_dir: Path = None,
        sample_rate: int = 48000,
        channels: int = 2,
    ):
        """
        initialize recorder.

        args:
            output_dir: directory for recordings (default: ~/Recordings/meetcap)
            sample_rate: audio sample rate in hz
            channels: number of audio channels
        """
        if output_dir is None:
            output_dir = Path.home() / "Recordings" / "meetcap"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = sample_rate
        self.channels = channels
        self.session: RecordingSession | None = None
        self._stop_event = threading.Event()

    def start_recording(
        self,
        device_index: int,
        device_name: str = "Unknown Device",
        output_path: Path | None = None,
    ) -> Path:
        """
        start recording from specified device.

        args:
            device_index: avfoundation device index
            device_name: human-readable device name
            output_path: optional custom output path

        returns:
            path to the recording directory (not the file)

        raises:
            runtimeerror: if already recording or ffmpeg fails
        """
        if self.session is not None:
            raise RuntimeError("recording already in progress")

        # generate output directory and filename if not provided
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # create temporary directory for this recording session
            recording_dir = self.output_dir / f"{timestamp}-temp"
            recording_dir.mkdir(parents=True, exist_ok=True)
            output_path = recording_dir / "recording.wav"
        else:
            # for custom output paths, recording_dir is the parent directory
            recording_dir = output_path.parent

        # build ffmpeg command for single aggregate input
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-f",
            "avfoundation",
            "-i",
            f":{device_index}",
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        try:
            # start ffmpeg process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # use bytes for stdin control
            )

            self.session = RecordingSession(
                process=process,
                output_path=output_path,
                start_time=time.time(),
                device_name=device_name,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )

            # give ffmpeg a moment to initialize
            time.sleep(0.5)

            # check if process is still running
            if process.poll() is not None:
                stderr = process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg failed to start: {stderr[:500]}")

            # create notes.md file in the recording directory
            notes_path = recording_dir / "notes.md"
            try:
                with open(notes_path, "w", encoding="utf-8") as f:
                    f.write("# Meeting Notes\n\n")
                    f.write("*Add your notes here during or after the meeting*\n\n")
                    f.write("*This file will be included in the final summary*\n")
                console.print(f"[green]✓[/green] notes file created: {notes_path.name}")
                console.print(f"  path: {notes_path.absolute()}")
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] could not create notes file: {e}")

            console.print(f"[green]✓[/green] recording started: {output_path.name}")
            console.print(f"  device: {device_name} (index {device_index})")
            console.print(f"  format: {self.sample_rate} hz, {self.channels} channels")

            # return the directory path, not the file path
            return output_path.parent

        except Exception as e:
            self.session = None
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"failed to start recording: {e}") from e

    def start_dual_recording(
        self,
        blackhole_index: int,
        mic_index: int,
        output_path: Path | None = None,
    ) -> Path:
        """
        start recording from two devices with amix filter.

        args:
            blackhole_index: blackhole device index
            mic_index: microphone device index
            output_path: optional custom output path

        returns:
            path to the recording directory (not the file)

        raises:
            runtimeerror: if already recording or ffmpeg fails
        """
        if self.session is not None:
            raise RuntimeError("recording already in progress")

        # generate output directory and filename if not provided
        if output_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # create temporary directory for this recording session
            recording_dir = self.output_dir / f"{timestamp}-temp"
            recording_dir.mkdir(parents=True, exist_ok=True)
            output_path = recording_dir / "recording.wav"
        else:
            # for custom output paths, recording_dir is the parent directory
            recording_dir = output_path.parent

        # build ffmpeg command for dual input with amix
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-f",
            "avfoundation",
            "-i",
            f":{blackhole_index}",
            "-f",
            "avfoundation",
            "-i",
            f":{mic_index}",
            "-filter_complex",
            "amix=inputs=2:duration=longest:normalize=0",
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        try:
            # start ffmpeg process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )

            self.session = RecordingSession(
                process=process,
                output_path=output_path,
                start_time=time.time(),
                device_name="BlackHole + Microphone (dual)",
                sample_rate=self.sample_rate,
                channels=self.channels,
            )

            # give ffmpeg a moment to initialize
            time.sleep(0.5)

            # check if process is still running
            if process.poll() is not None:
                stderr = process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg failed to start: {stderr[:500]}")

            console.print(f"[green]✓[/green] dual recording started: {output_path.name}")
            console.print(f"  blackhole index: {blackhole_index}")
            console.print(f"  microphone index: {mic_index}")
            console.print(f"  format: {self.sample_rate} hz, mixed to stereo")

            # return the directory path, not the file path
            return output_path.parent

        except Exception as e:
            self.session = None
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"failed to start dual recording: {e}") from e

    def stop_recording(self, timeout: float = 5.0) -> Path | None:
        """
        stop the current recording gracefully.

        args:
            timeout: max seconds to wait for graceful shutdown

        returns:
            path to recording directory or none if no recording
        """
        if self.session is None:
            return None

        process = self.session.process
        output_path = self.session.output_path

        try:
            # try graceful stop first (send 'q' to stdin)
            if process.stdin and process.poll() is None:
                try:
                    process.stdin.write(b"q")
                    process.stdin.flush()
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            # wait for graceful exit
            try:
                process.wait(timeout=timeout / 2)
            except subprocess.TimeoutExpired:
                # fallback to terminate
                process.terminate()
                try:
                    process.wait(timeout=timeout / 2)
                except subprocess.TimeoutExpired:
                    # last resort: kill
                    process.kill()
                    process.wait(timeout=1)

            # check if file was created
            if output_path.exists() and output_path.stat().st_size > 44:  # wav header is 44 bytes
                duration = time.time() - self.session.start_time
                console.print(
                    f"[green]✓[/green] recording saved: {output_path.name} ({duration:.1f} seconds)"
                )
                # return the directory path, not the file path
                return output_path.parent
            else:
                console.print("[yellow]⚠[/yellow] recording file is empty or corrupted")
                if output_path.exists():
                    output_path.unlink()
                return None

        except Exception as e:
            console.print(f"[red]error stopping recording: {e}[/red]")
            return None
        finally:
            self.session = None

    def get_elapsed_time(self) -> float:
        """
        get elapsed recording time in seconds.

        returns:
            elapsed time or 0 if not recording
        """
        if self.session is None:
            return 0.0
        return time.time() - self.session.start_time

    def is_recording(self) -> bool:
        """check if currently recording."""
        return self.session is not None

    def show_progress(self) -> None:
        """display recording progress with elapsed time."""
        if not self.is_recording():
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Recording...[/bold blue]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("", total=None)

            while self.is_recording() and not self._stop_event.is_set():
                time.sleep(0.1)
                progress.update(task)

    def cleanup(self) -> None:
        """cleanup any active recording on exit."""
        if self.session is not None:
            self.stop_recording(timeout=2.0)
