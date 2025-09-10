# meetcap Project Rules

This file provides comprehensive guidance for agentic coding assistants working with the meetcap project.

## Project Overview

meetcap is an offline meeting recorder & summarizer for macOS that:
- Captures system audio and microphone simultaneously
- Transcribes locally with Whisper
- Summarizes with a local LLM
- All processing happens 100% offline without any network connections

## Development Setup

### Environment Requirements
- macOS (Apple Silicon recommended for MLX-Whisper performance)
- Python 3.11+ (managed via Hatch)
- FFmpeg for audio recording
- BlackHole for system audio capture

### Initial Setup Commands
```bash
# Create/update Hatch environment
hatch env create

# Install for development (with all extras)
pip install -e ".[dev,stt,mlx-stt,vosk-stt]"

# Pre-commit hooks setup
pre-commit install
```

## Development Commands

### Testing
```bash
# Run tests with coverage
hatch run test
hatch run test-cov

# Run specific test file
hatch run pytest tests/test_cli.py -v

# Run single test
hatch run pytest tests/test_cli.py::test_record_command -v
```

### Code Quality
```bash
# Format code (auto-fix)
hatch run fmt
hatch run format

# Lint code (check only)
hatch run lint
```

### Building
```bash
# Build distribution packages
hatch build
```

### Application Commands (Development Mode)
**Important**: Always use `hatch run` prefix for development to ensure proper environment activation.

```bash
# Recording operations
hatch run meetcap record                    # Start recording with default device
hatch run meetcap record --stt mlx          # Use MLX-Whisper (Apple Silicon)
hatch run meetcap record --stt vosk         # Use Vosk with speaker identification
hatch run meetcap record --device "Aggregate Device"  # Specify device

# Processing operations
hatch run meetcap summarize audio.m4a       # Process existing audio file
hatch run meetcap summarize --stt mlx audio.wav  # Process with specific STT engine

# Reprocessing operations
hatch run meetcap reprocess 2025_Jan_15_TeamStandup  # Reprocess both transcript & summary
hatch run meetcap reprocess recording_dir --mode summary  # Reprocess summary only
hatch run meetcap reprocess --stt mlx --yes recording  # Change STT engine, skip confirm

# System operations
hatch run meetcap devices                   # List audio devices
hatch run meetcap verify                    # Verify system setup
hatch run meetcap setup                     # Interactive setup wizard
```

## Architecture & Key Components

### Modular Service Architecture
The codebase uses a service-oriented architecture with clear separation of concerns:

1. **Core Layer** (`meetcap/core/`)
   - [`recorder.py`](meetcap/core/recorder.py): FFmpeg-based audio recording with subprocess management
   - [`devices.py`](meetcap/core/devices.py): Audio device discovery and selection logic
   - [`hotkeys.py`](meetcap/core/hotkeys.py): Global hotkey management using pynput

2. **Services Layer** (`meetcap/services/`)
   - [`transcription.py`](meetcap/services/transcription.py): Multi-engine STT with automatic fallback (MLX → Faster-Whisper → Whisper.cpp)
   - [`summarization.py`](meetcap/services/summarization.py): LLM-based summarization with chunking for long transcripts
   - [`model_download.py`](meetcap/services/model_download.py): Automatic model downloading and verification

3. **CLI Interface** ([`cli.py`](meetcap/cli.py))
   - Typer-based CLI with rich console output
   - Commands: record, summarize, reprocess, devices, verify, setup
   - Path resolution with fuzzy matching for recordings

### Audio Processing Pipeline
```
Audio Input → FFmpeg Recording → WAV File → STT Service → Transcript → LLM Service → Summary
```

### STT Engine Strategy
The system implements a sophisticated fallback mechanism:
1. **Primary (Apple Silicon)**: MLX-Whisper with Metal acceleration
2. **Fallback**: Faster-Whisper with CTranslate2 optimization
3. **Alternative**: Vosk for speaker diarization
4. **Emergency**: Whisper.cpp CLI executable

Engine selection is automatic based on:
- Hardware capabilities (Apple Silicon detection)
- Available dependencies
- User configuration
- Runtime errors (automatic fallback)

### LLM Summarization
- Uses llama-cpp-python with Metal GPU acceleration
- Automatic context batching for long transcripts
- Structured markdown output generation
- Special handling for thinking models (removes `<thinking>` tags)

### Reprocessing System
The reprocess command uses a backup/restore pattern:
- `BackupManager` creates `.backup` files before reprocessing
- Supports two modes: full STT reprocessing or summary-only
- Automatic cleanup of backup files on success
- Restoration on failure

## Configuration System

### Configuration Hierarchy (highest to lowest priority)
1. Command-line arguments
2. Environment variables (`MEETCAP_*`)
3. Config file (`~/.meetcap/config.toml`)
4. Default values

### Key Environment Variables
```bash
MEETCAP_DEVICE              # Audio device name
MEETCAP_STT_ENGINE          # STT engine: faster-whisper, mlx, vosk
MEETCAP_STT_MODEL           # Faster-whisper model path
MEETCAP_MLX_STT_MODEL       # MLX-whisper model name
MEETCAP_VOSK_MODEL          # Vosk model name
MEETCAP_LLM_MODEL           # Path to GGUF file
MEETCAP_OUT_DIR             # Output directory
MEETCAP_ENABLE_DIARIZATION  # Enable speaker identification (true/false)
```

## Testing Patterns

### Test Organization
- Unit tests in `tests/` mirror source structure
- Mock models in `tests/mock_models.py` to avoid large model downloads
- Shared fixtures in `tests/conftest.py`
- Coverage requirement: 75% minimum

### Key Testing Patterns
```python
# Always use temp directories for file operations
def test_with_files(temp_dir):
    file = temp_dir / "test.txt"

# Mock external commands (ffmpeg)
def test_recording(mock_subprocess_run):
    mock_subprocess_run.return_value.returncode = 0

# Reset environment variables
@pytest.fixture(autouse=True)
def reset_environment():
    # Automatically cleans MEETCAP_* env vars
```

## Output Structure

Recordings are organized by date and AI-generated title:
```
~/Recordings/meetcap/
├── 2025_Jan_15_TeamStandup/
│   ├── recording.wav              # Raw audio
│   ├── recording.transcript.txt   # Plain text
│   ├── recording.transcript.json  # With timestamps
│   └── recording.summary.md       # AI summary
```

## Model Management

Models are automatically downloaded to `~/.meetcap/models/`:
- **STT Models**: Whisper variants (MLX, Faster-Whisper, Vosk)
- **LLM Models**: GGUF format for llama.cpp
- **Speaker Models**: Vosk speaker identification model

The [`model_download.py`](meetcap/services/model_download.py) service handles:
- Progress bars with rich console
- SHA256 verification
- Automatic retries
- Local caching

## Version Management

### Using bump2version
This project uses `bump2version` to manage application versions. The configuration is in [`.bumpversion.cfg`](.bumpversion.cfg).

#### Version Information
- Current version: 1.2.2 (defined in [`meetcap/__init__.py`](meetcap/__init__.py:3))
- Version bumping commits changes and creates git tags
- Tag format: `v{new_version}`

#### Version Bumping Commands
**Important**: Always use `hatch run` prefix for version bumping to ensure proper environment activation.

```bash
# Bump version (major, minor, or patch)
hatch run bump2version major
hatch run bump2version minor
hatch run bump2version patch

# Bump version without committing/tagging (for testing)
hatch run bump2version --no-commit major
```

#### Version File Locations
- Primary version definition: [`meetcap/__init__.py`](meetcap/__init__.py:3)
- Configuration: [`.bumpversion.cfg`](.bumpversion.cfg)

## macOS-Specific Considerations

### Audio Device Setup
1. BlackHole installation required for system audio capture
2. Multi-Output Device for audio monitoring during recording
3. Aggregate Device combines BlackHole + Microphone inputs
4. Clock source must be set to microphone with drift correction

### Required Permissions
- Microphone access for audio recording
- Input Monitoring for global hotkeys

### FFmpeg Integration
- Uses AVFoundation for macOS audio capture
- Device indices from `ffmpeg -f avfoundation -list_devices true -i ""`
- Graceful shutdown via stdin 'q' command instead of SIGTERM

## Code Style & Conventions

### Python Style
- Black formatter with 100-char line length
- Type hints throughout (enforced by mypy)
- Lowercase inline comments
- Rich console for all user output

### Commit Message Format
```
type: brief description

- Detailed change 1
- Detailed change 2
```

Types: feat, fix, chore, docs, refactor, test

### Error Handling
- Graceful fallbacks for all services
- User-friendly error messages with rich formatting
- Automatic recovery attempts where possible
- Detailed logging to `~/.meetcap/debug.log`

## Critical Implementation Details

1. **Subprocess Management**: FFmpeg runs as a subprocess with careful stdin/stdout/stderr handling
2. **Hotkey Debouncing**: Prevents double-triggers with 0.5s debounce period
3. **Model Lazy Loading**: Models load on first use to reduce startup time
4. **Transcript Chunking**: Automatic chunking for transcripts exceeding LLM context
5. **Backup Safety**: All reprocessing operations create backups before modification
6. **Path Resolution**: Fuzzy matching for recording directories in reprocess command

## Development Workflow Tips

1. **Always use `hatch run`** for any Python execution in development
2. **Test audio recording** with short durations first (10-30 seconds)
3. **Check permissions** with `meetcap verify` after setup
4. **Use environment variables** for quick model/engine switching during development
5. **Monitor `~/.meetcap/debug.log`** for detailed troubleshooting
