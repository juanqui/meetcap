"""device discovery and enumeration for macos audio"""

import re
import subprocess
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class AudioDevice:
    """represents an avfoundation audio input device"""

    index: int
    name: str
    is_aggregate: bool = False


def list_audio_devices() -> list[AudioDevice]:
    """
    list all available avfoundation input devices.

    returns:
        list of audiodevice objects with index and name
    """
    try:
        # run ffmpeg to list devices
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,  # increased timeout
        )

        # ffmpeg writes device list to stderr
        output = result.stderr

        devices = []

        # look for the audio devices section
        audio_section = False
        for line in output.split("\n"):
            # detect start of audio devices section
            if "AVFoundation audio devices:" in line:
                audio_section = True
                continue

            # stop at video devices or error section
            if audio_section and (
                "AVFoundation video devices:" in line or "Error opening input" in line
            ):
                break

            # parse audio device lines like: [0] MacBook Air Microphone
            if audio_section and line.strip():
                # match pattern [AVFoundation indev @ 0x...] [index] device_name
                match = re.match(r"\[AVFoundation indev @ [^\]]+\]\s+\[(\d+)\]\s+(.+)", line)
                if match:
                    index = int(match.group(1))
                    name = match.group(2).strip()

                    # heuristic: check if likely aggregate device
                    is_aggregate = any(
                        keyword in name.lower()
                        for keyword in [
                            "aggregate",
                            "multi",
                            "blackhole",
                            "multi-output",
                            "loopback",
                        ]
                    )

                    devices.append(AudioDevice(index=index, name=name, is_aggregate=is_aggregate))

        return devices

    except subprocess.TimeoutExpired:
        console.print("[red]error: ffmpeg device listing timed out[/red]")
        return []
    except FileNotFoundError:
        console.print("[red]error: ffmpeg not found. install with: brew install ffmpeg[/red]")
        return []
    except Exception as e:
        console.print(f"[red]error listing devices: {e}[/red]")
        return []


def find_device_by_name(devices: list[AudioDevice], name: str) -> AudioDevice | None:
    """
    find a device by name (exact or substring match).

    args:
        devices: list of available devices
        name: device name or substring to search

    returns:
        matching device or none
    """
    # try exact match first
    for device in devices:
        if device.name == name:
            return device

    # try case-insensitive substring match
    name_lower = name.lower()
    for device in devices:
        if name_lower in device.name.lower():
            return device

    return None


def find_device_by_index(devices: list[AudioDevice], index: int) -> AudioDevice | None:
    """
    find a device by its index.

    args:
        devices: list of available devices
        index: device index

    returns:
        matching device or none
    """
    for device in devices:
        if device.index == index:
            return device
    return None


def print_devices(devices: list[AudioDevice]) -> None:
    """
    print devices in a formatted table.

    args:
        devices: list of devices to display
    """
    if not devices:
        console.print("[yellow]no audio input devices found[/yellow]")
        return

    table = Table(title="Available Audio Input Devices", show_header=True)
    table.add_column("Index", style="cyan", width=6)
    table.add_column("Name", style="white")
    table.add_column("Type", style="green")

    for device in devices:
        type_label = "Aggregate (Recommended)" if device.is_aggregate else "Standard"
        style = "bold green" if device.is_aggregate else ""
        table.add_row(str(device.index), device.name, type_label, style=style)

    console.print(table)

    # print helpful hints
    if any(d.is_aggregate for d in devices):
        console.print("\n[green]✓[/green] aggregate device detected (recommended for sync)")
    else:
        console.print(
            "\n[yellow]⚠[/yellow] no aggregate device found. "
            "create one in audio midi setup for best results."
        )
        console.print(
            "  combine: blackhole (for system audio) + your microphone\n"
            "  enable drift correction with mic as clock source"
        )


def select_best_device(devices: list[AudioDevice]) -> AudioDevice | None:
    """
    automatically select the best available device.

    prioritizes:
    1. aggregate devices
    2. blackhole devices
    3. first available device

    args:
        devices: list of available devices

    returns:
        best device or none if no devices
    """
    if not devices:
        return None

    # prefer aggregate devices
    for device in devices:
        if device.is_aggregate:
            return device

    # then blackhole
    for device in devices:
        if "blackhole" in device.name.lower():
            return device

    # fallback to first device
    return devices[0]
