---
paths:
  - "tests/**"
---

# Testing Rules

## Organization

- Tests in `tests/` mirror source structure
- Shared fixtures in `tests/conftest.py`
- Mock models in `tests/mock_models.py` (avoids large model downloads)
- Coverage requirement: **74% minimum**

## Running Tests

```bash
hatch run test                              # full suite with coverage
hatch run pytest tests/test_cli.py -v       # specific file
hatch run pytest tests/test_cli.py::test_record_command -v  # specific test
```

## Key Patterns

```python
# use temp directories for file operations
def test_with_files(tmp_path):
    file = tmp_path / "test.txt"

# mock external commands (ffmpeg, model loading)
def test_recording(mock_subprocess_run):
    mock_subprocess_run.return_value.returncode = 0

# reset environment variables (autouse fixture in conftest.py)
# automatically cleans MEETCAP_* env vars between tests
```

## Guidelines

- Mock external dependencies (ffmpeg, model files, audio devices)
- Never require actual model files or audio hardware in tests
- Use `pytest.mark.timeout` for tests that could hang
- Test both success and failure paths for service fallbacks
- When adding new functionality, add corresponding tests
