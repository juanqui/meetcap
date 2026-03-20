# Scheduled Stop Feature Specification

**Document**: Scheduled Stop Feature for Meeting Recorder
**Version**: 1.0
**Last Updated**: September 10, 2025
**Author**: meetcap development team

## 1. Overview

This specification defines the implementation of a scheduled stop feature for the meetcap meeting recorder. Users will be able to set a time limit for recordings that will automatically stop the recording process after the specified duration.

### 1.1 Current Problem

The meetcap application currently only supports manual stopping of recordings through:
1. A configured hotkey combination
2. Ctrl-C keyboard interrupt

There is no built-in way to automatically stop a recording after a specified time duration, which could be useful for users who want to record meetings of known length or limit recording sessions to prevent excessively long recordings.

### 1.2 Goals

- **Add timed recording capability**: Allow users to specify automatic stop times
- **Improve user experience**: Provide an intuitive prompt for scheduling options
- **Maintain backward compatibility**: Existing manual stopping methods continue to work
- **Prevent resource exhaustion**: Limit recordings to reasonable durations
- **Support common meeting durations**: Include 30 min, 1 hr, 1.5 hr, and 2 hr options

## 2. Architecture Overview

### 2.1 User Interaction Flow

The new scheduled stop feature will be integrated into the existing `meetcap record` command:

```
User runs "meetcap record"
↓
Prompt for scheduled stop option appears:
  1. No automatic stop (manual only)
  2. Stop after 30 minutes
  3. Stop after 1 hour
  4. Stop after 1.5 hours
  5. Stop after 2 hours
↓
User selects option
↓
Recording starts with selected schedule
↓
Recording continues until either:
  - Manual stop (hotkey or Ctrl-C)
  - Automatic stop (when timer expires)
```

### 2.2 Implementation Strategy

The scheduled stop will be implemented as a background timer that monitors the recording duration and automatically triggers the stop event when the time limit is reached. The implementation will follow this approach:

1. Add a timer thread that checks elapsed recording time periodically
2. When time limit is reached, trigger the same stop event used by hotkey/Ctrl-C
3. The orchestrator handles stopping uniformly regardless of trigger source

## 3. Implementation Requirements

### 3.1 CLI Interface Requirements

Update the `record` command to prompt for scheduled stop option:

```python
@app.command()
def record(
    device: str | None = typer.Option(
        None,
        "--device",
        "-d",
        help="audio device name or index",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        "-o",
        help="output directory",
    ),
    rate: int | None = typer.Option(
        None,
        "--rate",
        "-r",
        help="sample rate (hz)",
    ),
    channels: int | None = typer.Option(
        None,
        "--channels",
        "-c",
        help="number of channels",
    ),
    stt: str | None = typer.Option(
        None,
        "--stt",
        help="stt engine: fwhisper, mlx, vosk, or whispercpp (defaults to config)",
    ),
    llm: str | None = typer.Option(
        None,
        "--llm",
        help="path to llm gguf model",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="random seed for llm",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="path to log file",
    ),
    # New parameter for scheduled stop
    auto_stop: int | None = typer.Option(
        None,
        "--auto-stop",
        help="auto stop recording after minutes (30, 60, 90, 120)",
    ),
) -> None:
    """start recording a meeting with optional scheduled stop"""

    # If auto_stop is not specified, prompt user
    if auto_stop is None:
        console.print("[bold]⏱️ Scheduled Stop Options[/bold]\n")
        console.print("1. No automatic stop (manual only)")
        console.print("2. Stop after 30 minutes")
        console.print("3. Stop after 1 hour")
        console.print("4. Stop after 1.5 hours")
        console.print("5. Stop after 2 hours\n")

        choice = typer.prompt("Select option (1-5)", default="1")
        try:
            choice_idx = int(choice)
            if choice_idx == 2:
                auto_stop = 30
            elif choice_idx == 3:
                auto_stop = 60
            elif choice_idx == 4:
                auto_stop = 90
            elif choice_idx == 5:
                auto_stop = 120
            # choice_idx == 1 means no automatic stop
        except ValueError:
            pass  # Default to no automatic stop
```

### 3.2 Recording Orchestrator Requirements

Update the `RecordingOrchestrator` to support automatic stopping:

```python
class RecordingOrchestrator:
    def __init__(self, config: Config):
        """initialize orchestrator with config."""
        self.config = config
        self.recorder = None
        self.hotkey_manager = None
        self.stop_event = threading.Event()
        self.auto_stop_timer = None
        self.auto_stop_minutes = None
        # ... other existing attributes

    def run(
        self,
        device: str | None = None,
        output_dir: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        stt_engine: str | None = None,
        llm_path: str | None = None,
        seed: int | None = None,
        auto_stop: int | None = None,  # New parameter
    ) -> None:
        """
        run the complete recording workflow.

        args:
            # ... existing parameters
            auto_stop: minutes after which to automatically stop recording
        """
        # Store auto_stop_minutes for later use
        self.auto_stop_minutes = auto_stop

        # ... existing initialization code

        try:
            # start recording
            recording_dir = self.recorder.start_recording(
                device_index=selected_device.index,
                device_name=selected_device.name,
            )

            # If auto_stop is specified, start timer thread
            if self.auto_stop_minutes is not None and self.auto_stop_minutes > 0:
                self._start_auto_stop_timer()

            # ... rest of existing implementation

        except Exception as e:
            # ... existing error handling
        finally:
            # ... existing cleanup including timer cleanup
            if self.auto_stop_timer and self.auto_stop_timer.is_alive():
                self.auto_stop_timer.join(timeout=1.0)

    def _start_auto_stop_timer(self) -> None:
        """start background timer for automatic stopping."""
        self.auto_stop_timer = threading.Thread(
            target=self._auto_stop_worker,
            daemon=True
        )
        self.auto_stop_timer.start()

    def _auto_stop_worker(self) -> None:
        """background worker that monitors recording time and triggers auto stop."""
        import time

        if not self.auto_stop_minutes:
            return

        stop_seconds = self.auto_stop_minutes * 60

        while not self.stop_event.is_set():
            elapsed = self.recorder.get_elapsed_time()
            if elapsed >= stop_seconds:
                console.print(f"\n[yellow]⏱️[/yellow] automatically stopping recording after {self.auto_stop_minutes} minutes")
                self._stop_recording()
                break
            time.sleep(1)  # Check every second

    def _stop_recording(self) -> None:
        """callback for hotkey to stop recording."""
        self.stop_event.set()
        # Signal to the progress display thread to stop
        if self.recorder:
            self.recorder._stop_event.set()
```

### 3.3 Audio Recorder Requirements

Update the `AudioRecorder` to support the timer implementation:

```python
class AudioRecorder:
    def __init__(
        self,
        output_dir: Path = None,
        sample_rate: int = 48000,
        channels: int = 2,
    ):
        """
        initialize recorder.

        # ... existing documentation
        """
        # ... existing initialization
        self._stop_event = threading.Event()  # Add this for timer coordination

    def get_elapsed_time(self) -> float:
        """
        get elapsed recording time in seconds.

        returns:
            elapsed time or 0 if not recording
        """
        if self.session is None:
            return 0.0
        return time.time() - self.session.start_time
```

## 4. User Experience

### 4.1 Default Behavior

When a user runs `meetcap record` without any new options, they will be prompted with the scheduled stop options:

```bash
$ meetcap record
⏱️ Scheduled Stop Options

1. No automatic stop (manual only)
2. Stop after 30 minutes
3. Stop after 1 hour
4. Stop after 1.5 hours
5. Stop after 2 hours

Select option (1-5) [1]:
```

### 4.2 Command Line Options

Users can bypass the prompt by specifying the auto-stop time directly:

```bash
# Stop after 1 hour
meetcap record --auto-stop 60

# Stop after 90 minutes
meetcap record --auto-stop 90
```

### 4.3 Timer Status Display

During recording, if an automatic stop is scheduled, the progress display will show both elapsed time and time remaining:

```
Recording... 00:15:32 (⏱️ auto-stop in 44:28)
```

## 5. Configuration Options

### 5.1 Default Scheduled Stop Time

Add a configuration option to set a default scheduled stop time:

```toml
[recording]
# Default scheduled stop time in minutes (0 = no auto stop)
default_auto_stop = 0
```

### 5.2 Environment Variables

Support scheduled stop configuration via environment variables:

```bash
MEETCAP_RECORDING_AUTO_STOP=60
```

## 6. Error Handling

### 6.1 Invalid Time Limits

Handle cases where users specify invalid auto-stop times:

```python
def validate_auto_stop_time(minutes: int) -> bool:
    """validate that auto stop time is one of the supported options."""
    return minutes in [0, 30, 60, 90, 120]

def run(self, ... auto_stop: int | None = None):
    """run the complete recording workflow."""
    # Validate auto_stop value
    if auto_stop is not None and not validate_auto_stop_time(auto_stop):
        console.print(f"[red]error: invalid auto-stop time {auto_stop} minutes[/red]")
        console.print("[yellow]supported values: 0, 30, 60, 90, 120[/yellow]")
        raise typer.Exit(1)
```

### 6.2 Timer Interaction with Manual Stops

Ensure proper handling when both automatic and manual stop methods are used:

1. If a manual stop occurs before the timer expires, the timer thread should terminate gracefully
2. If the timer expires, it should trigger the same stop mechanism used by manual stops
3. The orchestrator should handle stop events uniformly regardless of source

## 7. Testing Strategy

### 7.1 Unit Tests

Implement unit tests for the scheduled stop functionality:

```python
def test_auto_stop_validation():
    """Test auto stop time validation."""
    assert validate_auto_stop_time(0) == True
    assert validate_auto_stop_time(30) == True
    assert validate_auto_stop_time(60) == True
    assert validate_auto_stop_time(90) == True
    assert validate_auto_stop_time(120) == True
    assert validate_auto_stop_time(45) == False
    assert validate_auto_stop_time(-30) == False

def test_auto_stop_timer_thread():
    """Test auto stop timer thread creation and termination."""
    orchestrator = RecordingOrchestrator(config)
    orchestrator.auto_stop_minutes = 30
    orchestrator._start_auto_stop_timer()

    assert orchestrator.auto_stop_timer is not None
    assert orchestrator.auto_stop_timer.is_alive() == True

    # Test that timer can be stopped
    orchestrator.stop_event.set()
    orchestrator.auto_stop_timer.join(timeout=2.0)
    assert orchestrator.auto_stop_timer.is_alive() == False
```

### 7.2 Integration Tests

Test the complete workflow with scheduled stop:

```python
def test_recording_with_auto_stop():
    """Test recording with automatic stopping."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        recorder = AudioRecorder(output_dir=temp_path)

        # Mock the timer to stop after 3 seconds instead of 30 minutes
        orchestrator = RecordingOrchestrator(config)
        orchestrator.auto_stop_minutes = 30  # Would be 30 minutes normally

        # Override the worker to use a shorter time for testing
        def _test_auto_stop_worker(self):
            """Test version that stops after 3 seconds."""
            time.sleep(3)
            self._stop_recording()

        orchestrator._auto_stop_worker = _test_auto_stop_worker.__get__(orchestrator)

        # Start recording in a separate thread
        recording_thread = threading.Thread(
            target=orchestrator.run,
            kwargs={
                "output_dir": str(temp_path),
                "auto_stop": 30
            }
        )
        recording_thread.start()

        # Wait for completion
        recording_thread.join(timeout=5.0)

        # Verify recording completed automatically
        assert orchestrator.stop_event.is_set() == True
```

## 8. Migration Path

### 8.1 Phase 1: Core Implementation

1. Add `auto_stop_minutes` parameter to `RecordingOrchestrator.run()`
2. Implement `_start_auto_stop_timer()` and `_auto_stop_worker()` methods
3. Add timer thread management to the orchestrator

### 8.2 Phase 2: CLI Integration

1. Add `--auto-stop` option to the `record` command
2. Implement the scheduled stop prompt when option is not provided
3. Add validation for auto-stop time values

### 8.3 Phase 3: Configuration and UX

1. Add configuration options for default scheduled stop
2. Update progress display to show time remaining when auto-stop is active
3. Add environment variable support

### 8.4 Phase 4: Testing and Documentation

1. Implement unit tests for timer functionality
2. Add integration tests for the complete workflow
3. Update user documentation with scheduled stop feature

## 9. Success Criteria

The implementation will be considered successful when:

1. **Scheduled stopping works**: Recording automatically stops at the specified time
2. **Manual stopping still works**: Hotkey and Ctrl-C continue to function as before
3. **User prompting functions**: Default prompt appears with the correct options
4. **Backward compatibility**: Existing commands continue to work without modification
5. **Error handling**: Invalid times are properly rejected with helpful messages
6. **Resource cleanup**: Timer threads are properly terminated when no longer needed

## 10. Implementation Considerations

### 10.1 Thread Safety

The timer implementation must be thread-safe:

- Use proper synchronization with `threading.Event` for signaling
- Ensure timer thread terminates gracefully when manual stop occurs
- Verify that only one timer thread is active per recording session

### 10.2 User Experience

- The prompt should be clear and intuitive
- Default selection should be "1" (no automatic stop)
- Progress display should enhance rather than complicate the UI

### 10.3 Error Handling

- Invalid auto-stop values should be caught early with clear error messages
- Timer worker should gracefully handle recorder state changes
- The system should fail safely if timer thread encounters issues

---
**Next Steps**: Implement Phase 1 by adding the core timer functionality to the RecordingOrchestrator class.
