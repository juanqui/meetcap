# CLI Modularization Specification

## Overview

The current `meetcap/cli.py` file has grown to 1999 lines and contains multiple responsibilities that should be separated for better maintainability, testability, and code organization. This specification outlines a plan to modularize the CLI into smaller, focused components.

## Current State Analysis

### File Size & Complexity
- **Lines of code**: 1999 lines
- **Main classes**: 3 (BackupManager, RecordingOrchestrator, standalone functions)
- **Commands**: 6 (record, summarize, reprocess, devices, setup, verify)
- **Mixed responsibilities**: CLI commands, orchestration logic, UI presentation, backup management

### Current Structure
```
cli.py
├── Imports & Constants (50 lines)
├── Utility Functions (10 lines)
├── BackupManager class (150 lines)
├── RecordingOrchestrator class (850 lines)
├── Command Functions (900+ lines)
└── Main entry point (10 lines)
```

## Proposed Modular Structure

### 1. Core CLI Module (`meetcap/cli/__init__.py`)
**Purpose**: Main entry point and command registration
**Size**: ~50 lines

```python
"""Main CLI entry point and command registration."""

import typer
from rich.console import Console

from meetcap import __version__
from .commands import app as commands_app

console = Console()
app = typer.Typer(
    name="meetcap",
    help="offline meeting recorder & summarizer for macos",
    add_completion=False,
)

# Register command groups
app.add_typer(commands_app, name="")

@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="show version"),
) -> None:
    """meetcap - offline meeting recorder & summarizer for macos"""
    if version:
        console.print(f"meetcap v{__version__}")
        raise typer.Exit()

if __name__ == "__main__":
    app()
```

### 2. Commands Module (`meetcap/cli/commands.py`)
**Purpose**: Command definitions and argument parsing
**Size**: ~150 lines

```python
"""CLI command definitions and argument parsing."""

import typer
from pathlib import Path

from .handlers import (
    RecordHandler,
    SummarizeHandler,
    ReprocessHandler,
    DevicesHandler,
    SetupHandler,
    VerifyHandler
)

app = typer.Typer()

@app.command()
def record(
    device: str | None = typer.Option(None, "--device", "-d", help="audio device name or index"),
    out: str | None = typer.Option(None, "--out", "-o", help="output directory"),
    rate: int | None = typer.Option(None, "--rate", "-r", help="sample rate (hz)"),
    channels: int | None = typer.Option(None, "--channels", "-c", help="number of channels"),
    stt: str | None = typer.Option(None, "--stt", help="stt engine: fwhisper, mlx, vosk, or whispercpp"),
    llm: str | None = typer.Option(None, "--llm", help="path to llm gguf model"),
    seed: int | None = typer.Option(None, "--seed", help="random seed for llm"),
    log_file: str | None = typer.Option(None, "--log-file", help="path to log file"),
    auto_stop: int | None = typer.Option(None, "--auto-stop", help="auto stop recording after minutes (30, 60, 90, 120)"),
) -> None:
    """start recording a meeting with optional scheduled stop"""
    RecordHandler().handle(
        device=device, out=out, rate=rate, channels=channels,
        stt=stt, llm=llm, seed=seed, log_file=log_file, auto_stop=auto_stop
    )

@app.command()
def summarize(
    audio_file: str = typer.Argument(..., help="path to audio file (m4a, wav, mp3, etc.)"),
    stt: str | None = typer.Option(None, "--stt", help="stt engine: fwhisper, mlx, vosk, or whispercpp"),
    llm: str | None = typer.Option(None, "--llm", help="path to llm gguf model"),
    seed: int | None = typer.Option(None, "--seed", help="random seed for llm"),
    out: str | None = typer.Option(None, "--out", "-o", help="output directory for results"),
    log_file: str | None = typer.Option(None, "--log-file", help="path to log file"),
) -> None:
    """process an existing audio file (transcribe and summarize)"""
    SummarizeHandler().handle(
        audio_file=audio_file, stt=stt, llm=llm, seed=seed, out=out, log_file=log_file
    )

# ... other command definitions
```

### 3. Command Handlers (`meetcap/cli/handlers/`)
**Purpose**: Business logic for each command, separated from CLI concerns
**Structure**:
- `__init__.py` - exports
- `base.py` - base handler class (~50 lines)
- `record.py` - record command logic (~200 lines)
- `summarize.py` - summarize command logic (~100 lines)
- `reprocess.py` - reprocess command logic (~150 lines)
- `devices.py` - devices command logic (~50 lines)
- `setup.py` - setup wizard logic (~400 lines)
- `verify.py` - verification logic (~100 lines)

#### Base Handler (`meetcap/cli/handlers/base.py`)
```python
"""Base handler class for CLI commands."""

from abc import ABC, abstractmethod
from pathlib import Path

from meetcap.utils.config import Config
from meetcap.utils.logger import logger


class BaseHandler(ABC):
    """Base class for CLI command handlers."""

    def __init__(self):
        self.config = Config()

    def setup_logging(self, log_file: str | None) -> None:
        """Setup file logging if requested."""
        if log_file:
            logger.add_file_handler(Path(log_file))

    @abstractmethod
    def handle(self, **kwargs) -> None:
        """Handle the command with given arguments."""
        pass
```

#### Record Handler (`meetcap/cli/handlers/record.py`)
```python
"""Record command handler."""

import sys
from .base import BaseHandler
from ..orchestration import RecordingOrchestrator
from ..validation import validate_auto_stop_time
from ..ui import RecordingUI


class RecordHandler(BaseHandler):
    """Handler for the record command."""

    def handle(
        self,
        device: str | None = None,
        out: str | None = None,
        rate: int | None = None,
        channels: int | None = None,
        stt: str | None = None,
        llm: str | None = None,
        seed: int | None = None,
        log_file: str | None = None,
        auto_stop: int | None = None,
    ) -> None:
        """Handle the record command."""
        self.setup_logging(log_file)

        # Handle auto-stop configuration
        auto_stop = self._configure_auto_stop(auto_stop)

        # Validate auto-stop time
        if auto_stop is not None and not validate_auto_stop_time(auto_stop):
            RecordingUI.show_auto_stop_error(auto_stop)
            sys.exit(1)

        # Run orchestrator
        orchestrator = RecordingOrchestrator(self.config)
        try:
            orchestrator.run(
                device=device, output_dir=out, sample_rate=rate,
                channels=channels, stt_engine=stt, llm_path=llm,
                seed=seed, auto_stop=auto_stop
            )
        except KeyboardInterrupt:
            sys.exit(0)

    def _configure_auto_stop(self, auto_stop: int | None) -> int | None:
        """Configure auto-stop based on args, env, config, and user prompt."""
        # Implementation extracted from current CLI
        pass
```

### 4. Orchestration Module (`meetcap/cli/orchestration.py`)
**Purpose**: High-level workflow orchestration (moved from cli.py)
**Size**: ~400 lines

```python
"""Recording and processing workflow orchestration."""

import threading
import time
from pathlib import Path

from meetcap.utils.config import Config
from meetcap.core.recorder import AudioRecorder
from meetcap.core.hotkeys import HotkeyManager
from .ui import RecordingUI
from .backup import BackupManager


class RecordingOrchestrator:
    """Orchestrates the recording, transcription, and summarization workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.recorder = None
        self.hotkey_manager = None
        # ... other initialization

    def run(self, **kwargs) -> None:
        """Run the complete recording workflow."""
        # Implementation moved from current CLI
        pass

    def _process_recording(self, **kwargs) -> None:
        """Process recorded audio: transcribe and summarize."""
        # Implementation moved from current CLI
        pass

    # ... other orchestration methods
```

### 5. UI Components (`meetcap/cli/ui.py`)
**Purpose**: Console output formatting and user interaction
**Size**: ~200 lines

```python
"""CLI user interface components and formatting."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

console = Console()


class RecordingUI:
    """UI components for recording workflow."""

    @staticmethod
    def show_recording_banner(version: str) -> None:
        """Show the recording start banner."""
        console.print(
            Panel(
                f"[bold cyan]meetcap v{version}[/bold cyan]\\n"
                f"[green]starting recording...[/green]",
                title="🎙️ meeting recorder",
                expand=False,
            )
        )

    @staticmethod
    def show_auto_stop_error(minutes: int) -> None:
        """Show auto-stop validation error."""
        console.print(f"[red]error: invalid auto-stop time {minutes} minutes[/red]")
        console.print("[yellow]supported values: 0, 30, 60, 90, 120[/yellow]")

    @staticmethod
    def show_completion_panel(paths: dict, is_recording_workflow: bool = True) -> None:
        """Show completion panel with output files."""
        # Implementation moved from current CLI
        pass


class SetupUI:
    """UI components for setup wizard."""

    @staticmethod
    def show_setup_banner() -> None:
        """Show setup wizard banner."""
        # Implementation moved from current CLI
        pass

    @staticmethod
    def prompt_stt_engine_choice(is_apple_silicon: bool) -> dict:
        """Prompt user to select STT engine."""
        # Implementation moved from current CLI
        pass

# ... other UI classes
```

### 6. Backup Management (`meetcap/cli/backup.py`)
**Purpose**: File backup and restore functionality
**Size**: ~100 lines

```python
"""File backup and restore functionality for reprocessing operations."""

from pathlib import Path
from typing import List

from meetcap.utils.logger import logger


class BackupManager:
    """Manages file backups for reprocessing operations."""

    def __init__(self):
        self.backups: List[Path] = []

    def create_backup(self, file_path: Path) -> Path | None:
        """Create backup with .backup extension."""
        # Implementation moved from current CLI
        pass

    def restore_backup(self, file_path: Path) -> bool:
        """Restore from backup if exists."""
        # Implementation moved from current CLI
        pass

    # ... other backup methods
```

### 7. Validation Utils (`meetcap/cli/validation.py`)
**Purpose**: Input validation functions
**Size**: ~50 lines

```python
"""Input validation utilities for CLI commands."""

def validate_auto_stop_time(minutes: int) -> bool:
    """Validate that auto stop time is one of the supported options."""
    return minutes in [0, 30, 60, 90, 120]

def validate_stt_engine(engine: str) -> bool:
    """Validate STT engine name."""
    return engine in ["fwhisper", "mlx", "vosk", "whisper.cpp"]

def validate_audio_format(file_path: Path) -> bool:
    """Validate audio file format."""
    supported_formats = [".m4a", ".wav", ".mp3", ".mp4", ".aac", ".flac", ".ogg", ".opus", ".webm"]
    return file_path.suffix.lower() in supported_formats
```

### 8. Processing Module (`meetcap/cli/processing.py`)
**Purpose**: Audio processing pipeline management
**Size**: ~300 lines

```python
"""Audio processing pipeline for transcription and summarization."""

from pathlib import Path
from typing import Tuple, Optional

from meetcap.services.transcription import *
from meetcap.services.summarization import SummarizationService
from meetcap.utils.config import Config
from .ui import console


class AudioProcessor:
    """Handles audio processing pipeline."""

    def __init__(self, config: Config):
        self.config = config

    def process_audio_to_transcript(
        self,
        audio_file: Path,
        base_path: Path,
        stt_engine: str | None = None,
    ) -> Tuple[Path, Path] | None:
        """Process audio file to transcript."""
        # Implementation moved from RecordingOrchestrator
        pass

    def process_transcript_to_summary(
        self,
        transcript_path: Path,
        base_path: Path,
        llm_path: str | None = None,
        seed: int | None = None,
    ) -> Path | None:
        """Process transcript to summary."""
        # Implementation moved from RecordingOrchestrator
        pass
```

## Migration Plan

### Phase 1: Create New Module Structure
1. Create `meetcap/cli/` directory
2. Create all new module files with stub implementations
3. Update imports in existing code to maintain compatibility

### Phase 2: Move Utility Classes
1. Extract and move `BackupManager` → `meetcap/cli/backup.py`
2. Extract validation functions → `meetcap/cli/validation.py`
3. Create UI utility classes → `meetcap/cli/ui.py`

### Phase 3: Extract Command Handlers
1. Create base handler class
2. Extract each command's logic into separate handler classes
3. Update command functions to use handlers

### Phase 4: Move Orchestration Logic
1. Extract `RecordingOrchestrator` → `meetcap/cli/orchestration.py`
2. Extract audio processing methods → `meetcap/cli/processing.py`
3. Update imports and references

### Phase 5: Create New CLI Entry Point
1. Create new `meetcap/cli/__init__.py` with command registration
2. Update `meetcap/cli.py` to import from new structure
3. Maintain backward compatibility

### Phase 6: Testing & Cleanup
1. Run full test suite to ensure no regressions
2. Update tests to import from new modules
3. Remove old `cli.py` file and update imports
4. Update documentation

## Benefits

### Code Organization
- **Single Responsibility**: Each module has a focused purpose
- **Maintainability**: Easier to locate and modify specific functionality
- **Testability**: Smaller, focused modules are easier to unit test
- **Reusability**: Components can be reused across different contexts

### Development Experience
- **Faster Navigation**: Developers can quickly find relevant code
- **Parallel Development**: Multiple developers can work on different commands
- **Code Reviews**: Smaller, focused changes are easier to review
- **Onboarding**: New developers can understand individual components

### Technical Benefits
- **Import Speed**: Only load needed modules
- **Memory Usage**: Smaller loaded modules reduce memory footprint
- **Dependency Management**: Clear separation of dependencies per module
- **Error Isolation**: Errors are contained within specific modules

## Backward Compatibility

The modularization will maintain full backward compatibility:
- Existing CLI commands will work unchanged
- Import paths will be maintained using `__all__` exports
- Configuration and behavior will remain identical
- Tests will continue to pass without modification

## Testing Strategy

### Unit Tests
- Create focused unit tests for each new module
- Test handlers independently of CLI framework
- Mock external dependencies (services, file system)

### Integration Tests
- Test command handlers with real orchestration
- Verify UI components render correctly
- Test backup/restore functionality

### Regression Tests
- Ensure all existing CLI tests pass
- Verify command-line behavior unchanged
- Test error handling and edge cases

## Future Enhancements

The modular structure enables future improvements:

### Plugin Architecture
- Command handlers could be dynamically loaded
- Third-party STT/LLM engines as plugins
- Custom workflow extensions

### API Extraction
- Handlers could be exposed as Python API
- Enable programmatic usage without CLI
- Integration with other tools/services

### Configuration Management
- Centralized configuration validation
- Profile-based configurations
- Environment-specific settings

## File Size Estimates

| Current File | New Module | Estimated Lines |
|--------------|------------|-----------------|
| cli.py (1999) | cli/__init__.py | 50 |
| | cli/commands.py | 150 |
| | cli/handlers/*.py | 1050 |
| | cli/orchestration.py | 400 |
| | cli/ui.py | 200 |
| | cli/backup.py | 100 |
| | cli/validation.py | 50 |
| | cli/processing.py | 300 |

**Total**: ~2300 lines across 8+ focused modules (vs 1999 lines in single file)

The slight increase in total lines is due to:
- Module docstrings and imports
- Explicit interfaces between modules
- Better error handling and validation
- Improved code documentation

This trade-off provides significant benefits in maintainability and organization.
