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

3. **Configure Audio Routing**:
   
   a. Create Multi-Output Device (for monitoring):
   - Open Audio MIDI Setup
   - Click "+" → Create Multi-Output Device
   - Add: Built-in Output + BlackHole 2ch
   - Set as system output device
   
   b. Create Aggregate Input Device (for recording):
   - Click "+" → Create Aggregate Device
   - Add: BlackHole 2ch + Your Microphone
   - Set microphone as clock source
   - Enable drift correction

4. **Models** (selected and downloaded during setup):
   - **Whisper models**: large-v3 (default), large-v3-turbo, or small
   - **LLM models**: Qwen3-4B-Thinking (default), Qwen3-4B-Instruct, or GPT-OSS-20B

### Install meetcap

```bash
# Clone repository
git clone https://github.com/yourusername/meetcap.git
cd meetcap

# Create virtual environment with hatch
hatch env create

# Install ML dependencies (required for transcription/summarization)
hatch run pip install faster-whisper
CMAKE_ARGS='-DLLAMA_METAL=on' hatch run pip install llama-cpp-python

# Run interactive setup wizard (first time only)
# This will test permissions and let you choose models to download
hatch run meetcap setup

# Quick verification (check status without downloads)
hatch run meetcap verify

# Run meetcap
hatch run meetcap --help
```

## Usage

### Basic Recording

```bash
# Start recording (uses default/best audio device)
hatch run meetcap record

# Press ⌘+⇧+S or Ctrl-C to stop recording
```

### Process Existing Audio Files

```bash
# Transcribe and summarize an existing audio file
hatch run meetcap summarize path/to/meeting.m4a

# Specify output directory
hatch run meetcap summarize recording.m4a --out ./results
```

### Commands

```bash
# First-time setup (interactive wizard)
hatch run meetcap setup

# List available audio devices
hatch run meetcap devices

# Quick system verification
hatch run meetcap verify

# Start recording a meeting
hatch run meetcap record --device "Aggregate Device" --out ~/MyRecordings

# Process existing audio file (m4a, wav, mp3, etc.)
hatch run meetcap summarize samples/meeting.m4a --out ./processed
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

## Output Files

Each recording session creates:
- `YYYYmmdd-HHMMSS.wav` - Audio recording
- `YYYYmmdd-HHMMSS.transcript.txt` - Plain text transcript
- `YYYYmmdd-HHMMSS.transcript.json` - Transcript with timestamps
- `YYYYmmdd-HHMMSS.summary.md` - Meeting summary with decisions and action items

## License

MIT