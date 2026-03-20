# Development Workflow

## uv Requirement

This project uses uv for dependency management. **All Python commands must be prefixed with `uv run`** to ensure the correct virtual environment is activated. Without it, you'll get `ModuleNotFoundError: No module named 'meetcap'`.

```bash
# correct
uv run python script.py
uv run python -c "import meetcap; print(meetcap.__version__)"
uv run pytest tests/test_cli.py -v

# wrong — will fail
python script.py
pytest tests/test_cli.py
```

## Configuration Hierarchy

Priority from highest to lowest:

1. Command-line arguments
2. Environment variables (`MEETCAP_*`)
3. Config file (`~/.meetcap/config.toml`)
4. Default values

### Key Environment Variables

```bash
MEETCAP_DEVICE              # audio device name
MEETCAP_STT_ENGINE          # stt engine: faster-whisper, mlx, vosk
MEETCAP_STT_MODEL           # faster-whisper model path
MEETCAP_MLX_STT_MODEL       # mlx-whisper model name
MEETCAP_VOSK_MODEL          # vosk model name
MEETCAP_LLM_MODEL           # huggingface repo id for llm (e.g., mlx-community/Qwen3.5-4B-MLX-4bit)
MEETCAP_OUT_DIR             # output directory
MEETCAP_ENABLE_DIARIZATION  # enable speaker identification (true/false)
MEETCAP_PARAKEET_MODEL      # parakeet model name (default: mlx-community/parakeet-tdt-0.6b-v3)
MEETCAP_DIARIZATION_BACKEND # diarization backend: sherpa or vosk
MEETCAP_SHERPA_NUM_SPEAKERS # expected speaker count (-1 for auto)
MEETCAP_SHERPA_THRESHOLD    # clustering threshold (0.0-1.0)
```

## macOS Permissions

- **Microphone Access**: System Preferences > Privacy & Security > Microphone
- **Input Monitoring**: System Preferences > Privacy & Security > Input Monitoring (for hotkeys)

## Version Management

Uses `bump-my-version` (maintained successor to `bump2version`) for versioning. Configuration lives in `pyproject.toml` under `[tool.bumpversion]`. The single source of truth for the version string is `meetcap/__init__.py`.

```bash
uv run bump-my-version bump patch   # 2.0.0 → 2.0.1
uv run bump-my-version bump minor   # 2.0.0 → 2.1.0
uv run bump-my-version bump major   # 2.0.0 → 3.0.0
uv run bump-my-version show current_version  # show current version
```

**NEVER edit `__version__` manually.** Always use `bump-my-version` so that `pyproject.toml` (`[tool.bumpversion] current_version`), `meetcap/__init__.py`, and git tags stay in sync. Manual edits cause version drift that breaks future bumps.
