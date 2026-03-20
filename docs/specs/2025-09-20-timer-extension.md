# Timer Extension and Cancellation Feature

**Document**: Timer Extension and Cancellation for Auto-Stop Recording
**Version**: 1.0
**Date**: September 20, 2025
**Author**: meetcap development team

## 1. Overview

This specification defines a feature that allows users to extend, modify, or cancel the auto-stop timer during an active recording session. This addresses the common scenario where meetings run longer than expected or when users need to dynamically adjust recording duration.

### 1.1 Problem Statement

The current auto-stop timer implementation (fixed in the previous iteration) works reliably but lacks flexibility:

- Users must commit to a fixed duration when starting recording
- No way to extend time when meetings run longer than expected
- No way to cancel timer if meeting ends early but user wants to keep recording
- No way to modify timer mid-recording for different meeting types

### 1.2 Goals

- **Dynamic Timer Control**: Allow real-time modification of auto-stop timer
- **Ergonomic Interface**: Provide multiple intuitive ways to control timer
- **Non-Disruptive**: Don't interfere with ongoing recording or UI
- **Backward Compatible**: Maintain existing auto-stop functionality
- **Robust**: Handle edge cases and race conditions gracefully

## 2. Current State Analysis

### 2.1 Existing Timer Architecture

The current implementation (from `meetcap/cli.py`) includes:

```python
class RecordingOrchestrator:
    def __init__(self, config: Config):
        # ... existing attributes
        self.auto_stop_minutes = None
        self.auto_stop_timer = None
        self.auto_stop_start_time = None

    def _auto_stop_worker(self) -> None:
        """Independent timer worker that monitors elapsed time."""
        while not self.stop_event.is_set():
            elapsed = time.time() - self.auto_stop_start_time
            if elapsed >= stop_seconds:
                self._stop_recording()
                break
            time.sleep(1)
```

### 2.2 Current User Interaction Points

1. **CLI Options**: `--auto-stop 30|60|90|120`
2. **Interactive Prompt**: Menu selection at recording start
3. **Progress Display**: Shows remaining time during recording
4. **Hotkey System**: `Cmd+Shift+S` to stop recording manually
5. **Ctrl-C Handler**: Single/double interrupt handling

### 2.3 Current Limitations

- Timer duration is set once and cannot be changed
- No user feedback for timer operations during recording
- Single-threaded timer worker with limited extensibility

## 3. Design Principles

### 3.1 User Experience Principles

1. **Discoverability**: Features should be visible during recording
2. **Simplicity**: Common operations should be simple and fast
3. **Safety**: Prevent accidental timer modifications
4. **Feedback**: Clear confirmation of timer changes
5. **Accessibility**: Multiple input methods for different preferences

### 3.2 Technical Principles

1. **Thread Safety**: Handle concurrent timer modifications safely
2. **State Consistency**: Maintain timer state integrity
3. **Performance**: Minimal impact on recording performance
4. **Extensibility**: Design for future timer-related features

## 4. User Interface Design

### 4.1 Input Methods Analysis

#### Option A: Keyboard Shortcuts (Recommended)
**Pros:**
- Non-intrusive, works during recording
- Fast for power users
- Follows existing hotkey patterns
- Works with current `HotkeyManager` system

**Cons:**
- Requires memorizing shortcuts
- Limited by available key combinations
- Requires input monitoring permissions

#### Option B: CLI Commands in Separate Terminal
**Pros:**
- Full command flexibility
- Easy to script/automate
- Clear command structure

**Cons:**
- Requires opening second terminal
- May disrupt workflow
- Complex IPC implementation needed

#### Option C: Interactive Menu During Recording
**Pros:**
- Very discoverable
- Self-documenting options
- Easy for beginners

**Cons:**
- Interrupts progress display
- May be disruptive during meetings
- Complex UI state management

#### Option D: File-Based Control
**Pros:**
- Simple implementation
- Scriptable from outside
- No additional UI complexity

**Cons:**
- Not discoverable
- Requires file system polling
- Not ergonomic for real-time use

### 4.2 Recommended Approach: Multi-Modal Input

Combine **keyboard shortcuts** (primary) with **Ctrl-C menu** (secondary):

1. **Primary Interface**: Dedicated hotkeys for common operations
2. **Secondary Interface**: Enhanced Ctrl-C menu for advanced operations
3. **Discovery**: Progress display shows available shortcuts

## 5. Feature Specification

### 5.1 Timer Operations

#### 5.1.1 Extend Timer
- **Purpose**: Add time to current timer
- **Options**: +15min, +30min, +60min, custom amount
- **Behavior**: Modify `auto_stop_minutes` and update display

#### 5.1.2 Cancel Timer
- **Purpose**: Disable auto-stop, continue recording indefinitely
- **Behavior**: Set `auto_stop_minutes = None`, update display

#### 5.1.3 Set New Timer
- **Purpose**: Replace current timer with new duration
- **Options**: 30min, 60min, 90min, 120min, custom
- **Behavior**: Reset timer with new duration from current time

#### 5.1.4 Query Timer Status
- **Purpose**: Show detailed timer information
- **Display**: Current timer, time remaining, time elapsed

### 5.2 Keyboard Shortcuts Design

```
Primary Timer Operations:
├── Cmd+Shift+T     → Show timer menu / status
├── Cmd+Shift+E     → Extend timer (+30 min)
├── Cmd+Shift+C     → Cancel timer
└── Cmd+Shift+R     → Reset timer (new 60min from now)

Quick Extensions:
├── Cmd+Shift+1     → +15 minutes
├── Cmd+Shift+2     → +30 minutes
└── Cmd+Shift+3     → +60 minutes

Existing (unchanged):
└── Cmd+Shift+S     → Stop recording
```

### 5.3 Enhanced Progress Display

Current display:
```
recording 12:34 (⏱️ auto-stop in 17:26) (⌘⇧S or ⌃C to stop)
```

Enhanced display:
```
recording 12:34 (⏱️ auto-stop in 17:26) (⌘⇧S=stop ⌘⇧T=timer ⌘⇧E=+30min)
```

### 5.4 Timer Menu Interface

Accessed via `Cmd+Shift+T` or enhanced Ctrl-C menu:

```
⏱️ TIMER CONTROL MENU
====================
Current: 60 minutes (started 12:34 ago)
Remaining: 17:26

1. Extend +15 minutes
2. Extend +30 minutes
3. Extend +60 minutes
4. Set new timer...
5. Cancel timer
6. Show timer status
0. Return to recording

Enter choice (0-6):
```

## 6. Technical Implementation

### 6.1 Enhanced RecordingOrchestrator

```python
class RecordingOrchestrator:
    def __init__(self, config: Config):
        # ... existing attributes
        self.auto_stop_minutes = None
        self.auto_stop_timer = None
        self.auto_stop_start_time = None

        # NEW: Timer control attributes
        self.timer_lock = threading.Lock()
        self.timer_extension_event = threading.Event()
        self.timer_operations_queue = queue.Queue()

    def _auto_stop_worker(self) -> None:
        """Enhanced timer worker with dynamic control."""
        while not self.stop_event.is_set():
            # Check for timer operations
            if not self.timer_operations_queue.empty():
                self._process_timer_operation()

            # Check timer status with lock
            with self.timer_lock:
                if not self.auto_stop_minutes or not self.auto_stop_start_time:
                    time.sleep(1)
                    continue

                elapsed = time.time() - self.auto_stop_start_time
                stop_seconds = self.auto_stop_minutes * 60

                if elapsed >= stop_seconds:
                    console.print(f"\n⏱️ Auto-stopping after {self.auto_stop_minutes} minutes")
                    self._stop_recording()
                    break

            time.sleep(1)

    def extend_timer(self, minutes: int) -> None:
        """Extend current timer by specified minutes."""
        self.timer_operations_queue.put(('extend', minutes))

    def cancel_timer(self) -> None:
        """Cancel the current auto-stop timer."""
        self.timer_operations_queue.put(('cancel', None))

    def set_new_timer(self, minutes: int) -> None:
        """Set new timer duration from current time."""
        self.timer_operations_queue.put(('set', minutes))

    def _process_timer_operation(self) -> None:
        """Process queued timer operations safely."""
        try:
            operation, value = self.timer_operations_queue.get_nowait()

            with self.timer_lock:
                if operation == 'extend':
                    if self.auto_stop_minutes:
                        self.auto_stop_minutes += value
                        console.print(f"\n⏱️ Timer extended by {value} minutes")

                elif operation == 'cancel':
                    self.auto_stop_minutes = None
                    console.print("\n⏱️ Auto-stop timer cancelled")

                elif operation == 'set':
                    self.auto_stop_minutes = value
                    self.auto_stop_start_time = time.time()
                    console.print(f"\n⏱️ New {value}-minute timer set")

        except queue.Empty:
            pass
```

### 6.2 Enhanced HotkeyManager

```python
class HotkeyManager:
    def __init__(self, stop_callback: Callable, timer_callback: Callable):
        self.stop_callback = stop_callback
        self.timer_callback = timer_callback  # NEW
        # ... existing attributes

    def start(self, hotkey_combo: str = "<cmd>+<shift>+s") -> None:
        # Enhanced hotkey mappings
        hotkeys = {
            hotkey_combo: self._compatible_callback(self.stop_callback),
            "<cmd>+<shift>+t": self._compatible_callback(lambda: self.timer_callback('menu')),
            "<cmd>+<shift>+e": self._compatible_callback(lambda: self.timer_callback('extend', 30)),
            "<cmd>+<shift>+c": self._compatible_callback(lambda: self.timer_callback('cancel')),
            "<cmd>+<shift>+1": self._compatible_callback(lambda: self.timer_callback('extend', 15)),
            "<cmd>+<shift>+2": self._compatible_callback(lambda: self.timer_callback('extend', 30)),
            "<cmd>+<shift>+3": self._compatible_callback(lambda: self.timer_callback('extend', 60)),
        }
        # ... rest of implementation
```

### 6.3 Timer Menu Implementation

```python
def _show_timer_menu(self) -> None:
    """Display interactive timer control menu."""
    # Pause progress display
    self.stop_event.set()

    try:
        while True:
            # Clear screen and show menu
            self._display_timer_menu()

            choice = input("Enter choice (0-6): ").strip()

            if choice == '0':
                break
            elif choice == '1':
                self.extend_timer(15)
                break
            elif choice == '2':
                self.extend_timer(30)
                break
            # ... other menu options

    finally:
        # Resume progress display
        self.stop_event.clear()
```

## 7. User Experience Flow

### 7.1 Happy Path: Extend Timer During Recording

1. **Start Recording**: `meetcap record --auto-stop 30`
2. **Begin Recording**: Timer shows "auto-stop in 30:00"
3. **Need More Time**: Meeting running long at 25 minutes
4. **Extend Timer**: Press `Cmd+Shift+E`
5. **Confirmation**: "⏱️ Timer extended by 30 minutes"
6. **Continue**: Now shows "auto-stop in 35:00"
7. **Complete**: Recording continues with extended time

### 7.2 Advanced: Use Timer Menu

1. **Access Menu**: Press `Cmd+Shift+T` during recording
2. **Menu Display**: Timer control menu appears
3. **Choose Action**: Select "4. Set new timer..."
4. **Input Duration**: Enter custom duration (e.g., 90)
5. **Confirmation**: "⏱️ New 90-minute timer set"
6. **Resume**: Return to recording with new timer

### 7.3 Edge Case: Cancel Timer

1. **During Recording**: Meeting ending earlier than expected
2. **Cancel Timer**: Press `Cmd+Shift+C`
3. **Confirmation**: "⏱️ Auto-stop timer cancelled"
4. **Continue**: Recording continues indefinitely
5. **Manual Stop**: User stops with `Cmd+Shift+S` when ready

## 8. Error Handling and Edge Cases

### 8.1 Race Conditions

- **Timer Expires During Extension**: Cancel expiration, apply extension
- **Multiple Timer Operations**: Queue operations, process sequentially
- **Stop During Timer Menu**: Gracefully handle recording stop

### 8.2 Invalid Operations

- **Extend Without Timer**: Show error, suggest setting new timer
- **Negative Time Remaining**: Reset to minimum 1 minute
- **Invalid Menu Input**: Show error, re-prompt

### 8.3 System Edge Cases

- **Permission Denied**: Graceful fallback to Ctrl-C menu
- **Memory Pressure**: Lightweight timer operations only
- **Thread Crashes**: Restart timer thread if needed

## 9. Testing Strategy

### 9.1 Unit Tests

```python
def test_timer_extension():
    """Test timer extension functionality."""
    orchestrator = RecordingOrchestrator(config)
    orchestrator.auto_stop_minutes = 30
    orchestrator.auto_stop_start_time = time.time()

    # Extend timer
    orchestrator.extend_timer(15)
    time.sleep(0.1)  # Allow processing

    assert orchestrator.auto_stop_minutes == 45

def test_timer_cancellation():
    """Test timer cancellation."""
    # ... similar test structure

def test_timer_race_conditions():
    """Test concurrent timer operations."""
    # ... test concurrent extend/cancel operations
```

### 9.2 Integration Tests

```python
def test_timer_extension_integration():
    """Test timer extension with mock hotkeys."""
    # Simulate full recording workflow with timer extensions

def test_timer_menu_interaction():
    """Test interactive timer menu."""
    # Mock user input and test menu operations
```

### 9.3 Manual Testing Scenarios

1. **Quick Timer Tests**: Use 1-2 minute timers for rapid validation
2. **Hotkey Tests**: Verify all shortcut combinations work
3. **Menu Navigation**: Test all menu options and edge cases
4. **Real Meeting Simulation**: Test with actual meeting scenarios

## 10. Future Considerations

### 10.1 Potential Enhancements

- **Timer Presets**: Save/recall custom timer configurations
- **Timer Notifications**: Sound/visual alerts before auto-stop
- **Meeting Templates**: Pre-configured timers for different meeting types
- **Remote Control**: Web interface or mobile app control
- **Smart Extensions**: AI-based meeting end detection

### 10.2 Configuration Options

```toml
[timer]
# Default extension amounts (minutes)
quick_extensions = [15, 30, 60]

# Warning time before auto-stop (minutes)
warning_time = 5

# Enable/disable timer operations during recording
allow_runtime_changes = true

# Default hotkeys (pynput format)
[timer.hotkeys]
menu = "<cmd>+<shift>+t"
extend_default = "<cmd>+<shift>+e"
cancel = "<cmd>+<shift>+c"
```

## 11. Implementation Plan

### 11.1 Phase 1: Core Timer Control
- Implement timer extension/cancellation backend
- Add basic hotkey support for extend/cancel
- Update progress display with new shortcuts

### 11.2 Phase 2: Enhanced UI
- Implement interactive timer menu
- Add full hotkey suite
- Comprehensive error handling

### 11.3 Phase 3: Polish and Testing
- Edge case handling
- Comprehensive test suite
- Documentation and examples

### 11.4 Phase 4: Advanced Features
- Configuration options
- Custom timer presets
- Smart notifications

## 12. Success Criteria

The implementation will be considered successful when:

1. **Users can extend timers**: Seamlessly add time during recording
2. **Users can cancel timers**: Remove auto-stop restriction when needed
3. **Multiple input methods work**: Both hotkeys and menus functional
4. **No recording disruption**: Timer changes don't affect audio quality
5. **Intuitive operation**: Users can discover and use features easily
6. **Robust operation**: Handles edge cases and errors gracefully

---

**Next Steps**: Review this design with stakeholders, iterate based on feedback, and begin Phase 1 implementation.
