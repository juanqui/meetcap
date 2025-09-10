# Manual Notes Feature Specification

**Document**: Manual Notes Feature for Meeting Recorder
**Version**: 1.0
**Last Updated**: September 10, 2025
**Author**: meetcap development team

## 1. Overview

This specification defines the implementation of a manual notes feature for the meetcap meeting recorder. Users will be able to create and edit markdown notes that complement the automatically generated transcripts and summaries, providing a comprehensive record of meetings that includes both AI-generated content and user observations.

### 1.1 Current Problem

The meetcap application currently only supports automatic transcription and summarization of meetings:
1. Audio is recorded and transcribed using STT services
2. Transcripts are summarized using LLM services
3. Final outputs are saved as structured files in the recording directory

There is no built-in way for users to add their own notes, observations, or context that might not be captured by the audio recording or that they want to preserve alongside the AI-generated content.

### 1.2 Goals

- **Enable manual note-taking**: Allow users to create and edit markdown notes alongside AI-generated content
- **Enhance meeting documentation**: Provide a comprehensive record combining automatic and manual notes
- **Improve user workflow**: Make it easy to access and edit notes during and after meetings
- **Preserve user input**: Ensure manual notes are never deleted and are included in final summaries
- **Maintain backward compatibility**: Existing recording and summarization workflows continue to work

## 2. Architecture Overview

### 2.1 User Interaction Flow

The manual notes feature will be integrated into the existing `meetcap record` command workflow:

```
User runs "meetcap record"
↓
Recording starts and notes.md file is created in recording directory
↓
Full path to notes.md is displayed in terminal for easy access
↓
User can open notes.md file and add manual notes during/after recording
↓
Recording stops and STT/summarization completes
↓
Manual notes are read and included in the final summary
↓
Final summary includes both AI-generated content and manual notes
↓
Recording directory is organized with all files including notes.md
```

### 2.2 Implementation Strategy

The manual notes feature will be implemented by:

1. **File Creation**: Create a `notes.md` file when recording starts in the same directory as the recording
2. **Path Display**: Show the full path to `notes.md` in the terminal for easy access
3. **Content Integration**: Read manual notes and include them in the summarization prompt
4. **Preservation**: Ensure `notes.md` is never deleted and remains with the recording files

## 3. Implementation Requirements

### 3.1 Recording Orchestrator Updates

#### 3.1.1 File Creation in `start_recording`

Modify the [`AudioRecorder.start_recording()`](meetcap/core/recorder.py:55) method to create a `notes.md` file:

```python
def start_recording(
    self,
    device_index: int,
    device_name: str = "Unknown Device",
    output_path: Path | None = None,
) -> Path:
    # ... existing code ...

    # create notes.md file in the recording directory
    notes_path = recording_dir / "notes.md"
    try:
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write("# Meeting Notes\n\n")
            f.write("*Add your notes here during or after the meeting*\n\n")
            f.write "*This file will be included in the final summary*\n"
        console.print(f"[green]✓[/green] notes file created: {notes_path.name}")
        console.print(f"  path: {notes_path.absolute()}")
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] could not create notes file: {e}")

    # ... rest of existing code ...
```

#### 3.1.2 Path Display in `_show_recording_progress`

Update the [`RecordingOrchestrator._show_recording_progress()`](meetcap/cli.py:387) method to display the notes file path:

```python
def _show_recording_progress(self) -> None:
    # ... existing code ...

    # Add notes file path display
    if self.recorder:
        recording_dir = self.recorder.session.output_path.parent
        notes_path = recording_dir / "notes.md"
        if notes_path.exists():
            progress_str += f" [dim]notes: {notes_path.absolute()}[/dim]"

    console.print(progress_str, end="\r")
    # ... rest of existing code ...
```

### 3.2 Summarization Service Updates

#### 3.2.1 Manual Notes Reading

Modify the [`SummarizationService.summarize()`](meetcap/services/summarization.py:127) method to read manual notes:

```python
def summarize(
    self,
    transcript_text: str,
    meeting_title: str | None = None,
    attendees: list[str] | None = None,
    has_speaker_info: bool = False,
    manual_notes_path: Path | None = None,  # New parameter
) -> str:
    # ... existing code ...

    # read manual notes if available
    manual_notes_text = ""
    if manual_notes_path and manual_notes_path.exists():
        try:
            with open(manual_notes_path, "r", encoding="utf-8") as f:
                manual_notes_text = f.read()
            console.print("[dim]manual notes found, including in summary[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] could not read manual notes: {e}")

    # build user prompt with manual notes
    user_prompt_parts = []

    if meeting_title:
        user_prompt_parts.append(f"meeting: {meeting_title}")
    if attendees:
        user_prompt_parts.append(f"attendees: {', '.join(attendees)}")

    # add manual notes first if available
    if manual_notes_text:
        user_prompt_parts.append(f"manual notes:\n{manual_notes_text}")

    user_prompt_parts.append(f"transcript:\n{transcript_text}")
    user_prompt = "\n\n".join(user_prompt_parts)

    # ... rest of existing code ...
```

#### 3.2.2 Integration in `_process_transcript_to_summary`

Update the [`RecordingOrchestrator._process_transcript_to_summary()`](meetcap/cli.py:533) method to pass the manual notes path:

```python
def _process_transcript_to_summary(
    self,
    transcript_path: Path,
    base_path: Path,
    llm_path: str | None = None,
    seed: int | None = None,
) -> Path | None:
    # ... existing code ...

    # determine manual notes path
    manual_notes_path = base_path.with_name("notes.md")

    # read transcript text
    with open(transcript_path, encoding="utf-8") as f:
        transcript_text = f.read()

    # check if speaker information is available
    has_speaker_info = False
    json_path = base_path.with_suffix(".transcript.json")
    if json_path.exists():
        try:
            import json
            with open(json_path, encoding="utf-8") as f:
                transcript_data = json.load(f)
                has_speaker_info = transcript_data.get("diarization_enabled", False)
        except Exception:
            pass  # ignore errors reading JSON

    summary = llm_service.summarize(
        transcript_text,
        has_speaker_info=has_speaker_info,
        manual_notes_path=manual_notes_path  # Pass manual notes path
    )
    summary_path = save_summary(summary, base_path)

    # ... rest of existing code ...
```

### 3.3 CLI Interface Updates

#### 3.3.1 Enhanced Recording Output

Update the final output display in [`RecordingOrchestrator._process_recording()`](meetcap/cli.py:634) to include the notes file:

```python
# show final results with absolute paths for easy navigation
if is_recording_workflow:
    console.print(
        Panel(
            f"[green]✅ recording complete![/green]\n\n"
            f"[bold]artifacts:[/bold]\n"
            f"  folder: {final_dir_path.absolute()}\n"
            f"  audio: {audio_file.absolute()}\n"
            f"  transcript: {text_path.absolute()}\n"
            f"  json: {json_path.absolute()}\n"
            f"  summary: {summary_path.absolute()}\n"
            f"  notes: {notes_path.absolute()}",  # Add notes file
            title="📦 output files",
            expand=False,
        )
    )
else:
    # for standalone files, show simpler output with absolute paths
    console.print(
        Panel(
            f"[green]✅ processing complete![/green]\n\n"
            f"[bold]output files:[/bold]\n"
            f"  transcript: {text_path.absolute()}\n"
            f"  json: {json_path.absolute()}\n"
            f"  summary: {summary_path.absolute()}\n"
            f"  notes: {notes_path.absolute()}",  # Add notes file
            title="📦 results",
            expand=False,
        )
    )
```

### 3.4 Backup and Reprocessing Support

#### 3.4.1 Backup Manager Updates

Update the [`BackupManager`](meetcap/cli.py:64) to handle manual notes files:

```python
def create_backup(self, file_path: Path) -> Path | None:
    # ... existing code ...

    # backup notes files specifically
    if file_path.name == "notes.md":
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            self.backups.append(backup_path)
            return backup_path
        except Exception as e:
            logger.error(f"failed to create backup for {file_path}: {e}")
            return None

    # ... rest of existing code ...
```

#### 3.4.2 Reprocessing Integration

Update [`RecordingOrchestrator._reprocess_recording()`](meetcap/cli.py:805) to handle manual notes:

```python
def _reprocess_recording(
    self,
    recording_dir: Path,
    mode: str = "stt",
    stt_engine: str | None = None,
    llm_model: str | None = None,
    skip_confirm: bool = False,
) -> None:
    # ... existing code ...

    # check existing files
    notes_md = recording_dir / "notes.md"

    # backup notes.md for reprocessing
    if notes_md.exists():
        backup_manager.create_backup(notes_md)

    # ... rest of existing code ...

    # restore notes.md from backup on failure
    if notes_md.exists():
        backup_manager.create_backup(notes_md)

    # ... rest of existing code ...
```

## 4. User Experience

### 4.1 Default Behavior

When a user runs `meetcap record`, the workflow will now include:

```bash
$ meetcap record
🎙️ meeting recorder
✓ recording started: recording.wav
  device: Aggregate Device (index 0)
  format: 48000 hz, 2 channels
✓ notes file created: notes.md
  path: /Users/username/Recordings/meetcap/20250910-143045-temp/notes.md
recording 00:15:32 (⌃C or ⌘⇧S to stop)
```

Users can immediately click or copy the path to `notes.md` to start adding their notes.

### 4.2 Notes File Format

The `notes.md` file will be created with a basic template:

```markdown
# Meeting Notes

*Add your notes here during or after the meeting*

*This file will be included in the final summary*
```

### 4.3 Summary Integration

Manual notes will be included in the LLM prompt before the transcript:

```
System: You are an expert meeting note-taker...
User:
manual notes:
*User's notes content here*

transcript:
*AI-generated transcript content here*
```

## 5. Configuration Options

### 5.1 Notes File Configuration

Add configuration options for manual notes:

```toml
[notes]
# Enable manual notes feature
enable = true
# Template for new notes files
template = "# Meeting Notes\n\n*Add your notes here during or after the meeting*\n\n*This file will be included in the final summary*\n"
# Default notes file name
filename = "notes.md"
```

### 5.2 Environment Variables

Support manual notes configuration via environment variables:

```bash
MEETCAP_NOTES_ENABLE=true
MEETCAP_NOTES_TEMPLATE="# Meeting Notes\n\n*Add your notes here*\n"
MEETCAP_NOTES_FILENAME="notes.md"
```

## 6. Error Handling

### 6.1 Notes File Creation Errors

Handle cases where notes file creation fails:

```python
def create_notes_file(recording_dir: Path) -> Path | None:
    """Create notes.md file in recording directory."""
    notes_path = recording_dir / "notes.md"
    try:
        template = config.get("notes", "template", "# Meeting Notes\n\n*Add your notes here*\n")
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(template)
        return notes_path
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] could not create notes file: {e}")
        return None
```

### 6.2 Notes File Reading Errors

Handle cases where notes file cannot be read during summarization:

```python
def read_manual_notes(notes_path: Path) -> str:
    """Read manual notes file with error handling."""
    if not notes_path.exists():
        return ""

    try:
        with open(notes_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] could not read manual notes: {e}")
        return ""
```

## 7. Testing Strategy

### 7.1 Unit Tests

Implement unit tests for manual notes functionality:

```python
def test_notes_file_creation():
    """Test notes.md file creation during recording."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        recorder = AudioRecorder(output_dir=temp_path)

        # Mock the notes file creation
        def mock_create_notes(recording_dir):
            notes_path = recording_dir / "notes.md"
            with open(notes_path, "w") as f:
                f.write("# Meeting Notes\n\nTest content\n")
            return notes_path

        # Test that notes file is created
        notes_path = mock_create_notes(temp_path / "test-recording")
        assert notes_path.exists()
        assert notes_path.read_text() == "# Meeting Notes\n\nTest content\n"

def test_manual_notes_integration():
    """Test manual notes are included in summarization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        notes_path = temp_path / "notes.md"
        notes_path.write_text("# Meeting Notes\n\nImportant context about the meeting\n")

        transcript_path = temp_path / "transcript.txt"
        transcript_path.write_text("Hello everyone, let's discuss the project timeline.")

        # Test summarization includes manual notes
        mock_service = MockSummarizationService()
        summary = mock_service.summarize(
            transcript_text="Hello everyone, let's discuss the project timeline.",
            manual_notes_path=notes_path
        )

        assert "Important context about the meeting" in summary
```

### 7.2 Integration Tests

Test the complete workflow with manual notes:

```python
def test_recording_with_manual_notes():
    """Test complete recording workflow with manual notes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Simulate recording workflow
        recorder = AudioRecorder(output_dir=temp_path)
        recording_dir = recorder.start_recording(device_index=0, device_name="Test Device")

        # Verify notes file was created
        notes_path = recording_dir / "notes.md"
        assert notes_path.exists()

        # Add manual notes
        notes_path.write_text("# Meeting Notes\n\nKey decisions made:\n- Decision 1\n- Decision 2\n")

        # Complete recording and processing
        final_dir = recorder.stop_recording()
        assert final_dir is not None

        # Verify notes file is preserved
        final_notes_path = final_dir / "notes.md"
        assert final_notes_path.exists()
        assert "Key decisions made" in final_notes_path.read_text()
```

## 8. Migration Path

### 8.1 Phase 1: Core Implementation

1. **Notes File Creation**: Modify `AudioRecorder.start_recording()` to create `notes.md`
2. **Path Display**: Update `_show_recording_progress()` to display notes file path
3. **Basic Integration**: Modify `SummarizationService.summarize()` to read manual notes

### 8.2 Phase 2: CLI Integration

1. **Output Updates**: Modify final output panels to include notes file path
2. **Error Handling**: Add robust error handling for notes file operations
3. **Configuration**: Add notes configuration options to config system

### 8.3 Phase 3: Backup and Reprocessing

1. **Backup Support**: Update `BackupManager` to handle notes files
2. **Reprocessing Integration**: Ensure notes files are preserved during reprocessing
3. **Testing**: Add comprehensive tests for notes file preservation

### 8.4 Phase 4: Documentation and Polish

1. **User Documentation**: Update user guide with manual notes feature
2. **Error Messages**: Improve error messages for notes file operations
3. **Performance**: Ensure notes file reading doesn't impact performance

## 9. Success Criteria

The implementation will be considered successful when:

1. **Notes File Creation**: `notes.md` is automatically created when recording starts
2. **Path Display**: Full path to `notes.md` is displayed in terminal for easy access
3. **Manual Notes Integration**: Manual notes are included in the final summary
4. **File Preservation**: `notes.md` is never deleted and remains with recording files
5. **Error Handling**: Graceful handling of notes file creation/read errors
6. **Backward Compatibility**: Existing recording workflows continue to work unchanged
7. **Testing**: Comprehensive test coverage for all manual notes functionality

## 10. Implementation Considerations

### 10.1 File System Operations

- Use proper error handling for file creation and reading
- Ensure UTF-8 encoding for notes files
- Handle concurrent access scenarios gracefully
- Provide clear error messages when operations fail

### 10.2 User Experience

- Make notes file path easily accessible and copyable
- Provide clear instructions for adding manual notes
- Ensure notes file doesn't interfere with existing workflows
- Consider adding a timestamp to notes file for tracking

### 10.3 Performance Impact

- Notes file reading should be fast and non-blocking
- Large notes files should be handled gracefully
- Memory usage should not be significantly impacted
- File operations should not slow down the recording process

### 10.4 Security and Privacy

- Notes files contain user content and should be handled securely
- Ensure notes files are not accidentally shared or exposed
- Consider adding encryption options for sensitive notes
- Respect user privacy for manual notes content

---

**Next Steps**: Implement Phase 1 by modifying the AudioRecorder and SummarizationService to support manual notes file creation and integration.
