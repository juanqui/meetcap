# meetcap

Offline meeting recorder & summarizer for macOS

## Features

- Records both system audio and microphone simultaneously
- 100% offline operation - no network connections
- Local transcription using Whisper
- Local summarization using Qwen3-4B via llama.cpp
- Simple CLI workflow: start recording → stop with hotkey → get transcript & summary

## Installation

### Prerequisites

1. **Install ffmpeg**:
   ```bash
   brew install ffmpeg
   ```

2. **Install BlackHole** (for system audio capture):
   - Download from: https://github.com/ExistentialAudio/BlackHole
   - Install the 2ch version

   > **Why BlackHole?** macOS doesn't provide a native way to capture system audio (sounds from other apps, video calls, etc.) as an input source. BlackHole creates a virtual audio device that routes system output into an input that meetcap can record.

3. **Configure Audio Routing**:

   Open **Audio MIDI Setup** (Applications → Utilities) and create two devices:

   **Step 1: Multi-Output Device** (so you can hear AND record system audio):
   - Click "+" → Create Multi-Output Device
   - Check: **Built-in Output** (or your preferred speakers/headphones)
   - Check: **BlackHole 2ch**
   - Right-click this device → "Use This Device For Sound Output"
   - Name it "Multi-Output Device" (or similar)

   **Step 2: Aggregate Device** (combines system audio + microphone):
   - Click "+" → Create Aggregate Device
   - Check: **BlackHole 2ch** (captures system audio routed from step 1)
   - Check: **Your Microphone** (Built-in Microphone, USB mic, etc.)
   - Set your microphone as **clock source** (right-click → Use This Device as Clock Source)
   - Check **Drift Correction** for your microphone
   - Name it "Aggregate Device" (or similar)

   > **How it works:** System audio flows to Multi-Output → BlackHole → Aggregate Device, where it combines with your microphone for recording.

4. **Models** (selected and downloaded during setup):
   - **Whisper models**: large-v3 (default), large-v3-turbo, or small
   - **LLM models**: Qwen3-4B-Thinking (default), Qwen3-4B-Instruct, or GPT-OSS-20B

### Install meetcap

**macOS (Apple Silicon recommended):**
```bash
pip install "meetcap[mlx-stt]"
```

**Other platforms or Intel Macs:**
```bash
pip install "meetcap[stt]"
```

**First-time setup:**
```bash
# Run interactive setup wizard (downloads models, tests permissions)
meetcap setup

# Quick system verification
meetcap verify

# See all available commands
meetcap --help
```

> **Note:** The `mlx-stt` extra includes MLX Whisper for faster transcription on Apple Silicon. The `stt` extra uses faster-whisper which works on all platforms.

## Usage

### Basic Recording

```bash
# Start recording (uses default/best audio device)
meetcap record

# Press ⌘+⇧+S or Ctrl-C to stop recording
```

### Process Existing Audio Files

```bash
# Transcribe and summarize an existing audio file
meetcap summarize path/to/meeting.m4a

# Specify output directory
meetcap summarize recording.m4a --out ./results
```

### Commands

```bash
# First-time setup (interactive wizard)
meetcap setup

# List available audio devices
meetcap devices

# Quick system verification
meetcap verify

# Start recording a meeting
meetcap record --device "Aggregate Device" --out ~/MyRecordings

# Process existing audio file (m4a, wav, mp3, etc.)
meetcap summarize samples/meeting.m4a --out ./processed
```

### Configuration

Edit `~/.meetcap/config.toml` to customize:
- Default audio device
- Model settings (defaults to auto-downloaded models)
- Hotkey combinations
- Output directories

Models are automatically downloaded to `~/.meetcap/models/` on first use.

## Permissions

Grant these permissions to your terminal app:
1. **Microphone**: System Preferences → Privacy & Security → Microphone
2. **Input Monitoring**: System Preferences → Privacy & Security → Input Monitoring

## Troubleshooting

### Audio Setup Issues

**No audio devices found:**
- Ensure BlackHole is installed and Audio MIDI Setup devices are created
- Run `meetcap devices` to list available devices
- Try restarting your terminal or system after setup

**Can't hear system audio during recording:**
- Check that Multi-Output Device is set as system output (right-click in Audio MIDI Setup)
- Ensure Built-in Output is checked in your Multi-Output Device
- Test by playing music - you should hear it through your speakers

**Recording only captures microphone or only system audio:**
- Verify Aggregate Device includes both BlackHole 2ch AND your microphone
- Check that microphone is set as clock source in Aggregate Device
- Ensure Drift Correction is enabled for your microphone
- Use `meetcap record --device "Aggregate Device"` to specify the device explicitly

**Volume control keys causing errors:**
- This is a known compatibility issue with some pynput versions - the app should continue working normally despite the error message

## Output Files

Each recording session creates:
- `YYYYmmdd-HHMMSS.wav` - Audio recording
- `YYYYmmdd-HHMMSS.transcript.txt` - Plain text transcript
- `YYYYmmdd-HHMMSS.transcript.json` - Transcript with timestamps
- `YYYYmmdd-HHMMSS.summary.md` - Meeting summary with decisions and action items

## License

MIT
