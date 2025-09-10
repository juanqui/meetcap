"""shared test fixtures and configuration for meetcap tests"""

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import mock models
from tests.mock_models import mock_faster_whisper, mock_llama_cpp

# Install mock modules globally for tests
sys.modules["faster_whisper"] = mock_faster_whisper
sys.modules["llama_cpp"] = mock_llama_cpp


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_dir(temp_dir: Path) -> Path:
    """create a temporary config directory"""
    config_dir = temp_dir / ".meetcap"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def temp_audio_file(temp_dir: Path) -> Path:
    """create a temporary audio file"""
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"fake audio data")
    return audio_file


@pytest.fixture
def mock_subprocess_run():
    """mock subprocess.run for testing external commands"""
    with patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_ffmpeg_devices():
    """mock ffmpeg device listing output"""
    return """
[AVFoundation indev @ 0x7f8b0c704f40] AVFoundation video devices:
[AVFoundation indev @ 0x7f8b0c704f40] [0] FaceTime HD Camera
[AVFoundation indev @ 0x7f8b0c704f40] AVFoundation audio devices:
[AVFoundation indev @ 0x7f8b0c704f40] [0] Built-in Microphone
[AVFoundation indev @ 0x7f8b0c704f40] [1] BlackHole 2ch
[AVFoundation indev @ 0x7f8b0c704f40] [2] Aggregate Device (BlackHole + Mic)
"""


@pytest.fixture
def mock_config_data():
    """sample config data for testing"""
    return {
        "audio": {
            "preferred_device": "Aggregate Device",
            "sample_rate": 48000,
            "channels": 2,
        },
        "models": {
            "stt_model": "large-v3",
            "stt_model_path": "~/.meetcap/models/whisper",
            "llm_model": "qwen3-4b-thinking",
            "llm_gguf_path": "~/.meetcap/models/llm/qwen3-4b.gguf",
        },
        "paths": {
            "out_dir": "~/Recordings",
            "models_dir": "~/.meetcap/models",
        },
        "hotkey": {
            "stop": "cmd+shift+s",
        },
    }


@pytest.fixture
def mock_console():
    """mock rich console for testing output"""
    with patch("meetcap.core.devices.console") as mock:
        yield mock


@pytest.fixture
def mock_pynput_keyboard():
    """mock pynput keyboard for hotkey testing"""
    with patch("pynput.keyboard") as mock_kb:
        yield mock_kb


@pytest.fixture(autouse=True)
def reset_environment():
    """reset environment variables for each test"""
    env_vars = [
        "MEETCAP_DEVICE",
        "MEETCAP_SAMPLE_RATE",
        "MEETCAP_CHANNELS",
        "MEETCAP_HOTKEY",
        "MEETCAP_STT_ENGINE",
        "MEETCAP_STT_MODEL",
        "MEETCAP_MLX_STT_MODEL",
        "MEETCAP_VOSK_MODEL",
        "MEETCAP_VOSK_MODEL_PATH",
        "MEETCAP_VOSK_SPK_MODEL",
        "MEETCAP_ENABLE_DIARIZATION",
        "MEETCAP_LLM_MODEL",
        "MEETCAP_OUT_DIR",
        "MEETCAP_N_CTX",
        "MEETCAP_N_THREADS",
        "MEETCAP_N_GPU_LAYERS",
        # Memory management env vars
        "MEETCAP_MEMORY_AGGRESSIVE_GC",
        "MEETCAP_MEMORY_MONITORING",
        "MEETCAP_MEMORY_REPORT",
        "MEETCAP_MEMORY_WARNING_THRESHOLD",
        "MEETCAP_MEMORY_AUTO_FALLBACK",
    ]

    # store original environment
    original_env = {var: os.environ.get(var) for var in env_vars}

    # clear all MEETCAP environment variables before test
    for var in env_vars:
        os.environ.pop(var, None)

    yield

    # restore original environment after test
    for var, value in original_env.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


@pytest.fixture
def mock_whisper_model():
    """mock whisper model for transcription testing"""
    mock_model = Mock()
    mock_model.transcribe = Mock(
        return_value=(
            [{"text": "Test transcription", "start": 0.0, "end": 1.0}],
            {"text": "Test transcription", "language": "en"},
        )
    )
    return mock_model


@pytest.fixture
def mock_llm_model():
    """mock llm model for summarization testing"""
    mock_model = Mock()
    mock_model.return_value = "## Summary\n\nTest summary content"
    return mock_model


@pytest.fixture
def mock_faster_whisper():
    """mock faster-whisper module for testing"""
    mock_module = Mock()
    mock_whisper_model = Mock()
    mock_module.WhisperModel = mock_whisper_model

    with patch.dict("sys.modules", {"faster_whisper": mock_module}):
        yield mock_module, mock_whisper_model


@pytest.fixture
def mock_llama_cpp():
    """mock llama-cpp-python module for testing"""
    mock_module = Mock()
    mock_llama = Mock()
    mock_module.Llama = mock_llama

    with patch.dict("sys.modules", {"llama_cpp": mock_module}):
        yield mock_module, mock_llama


@pytest.fixture
def config(temp_config_dir):
    """create a config instance for testing"""
    from meetcap.utils.config import Config

    config_path = temp_config_dir / "config.toml"
    return Config(config_path)
