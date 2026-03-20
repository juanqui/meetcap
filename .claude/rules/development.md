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

Uses `bump2version` for versioning. Version defined in `meetcap/__init__.py`.

```bash
uv run bump2version patch   # 1.3.1 → 1.3.2
uv run bump2version minor   # 1.3.1 → 1.4.0
uv run bump2version major   # 1.3.1 → 2.0.0
```
