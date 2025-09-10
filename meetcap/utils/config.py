"""configuration management for meetcap"""

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from rich.console import Console

console = Console()


class Config:
    """manages application configuration"""

    DEFAULT_CONFIG = {
        "audio": {
            "preferred_device": "Aggregate Device",
            "sample_rate": 48000,
            "channels": 2,
        },
        "recording": {
            "default_auto_stop": 0,  # Default scheduled stop time in minutes (0 = no auto stop)
        },
        "hotkey": {
            "stop": "<cmd>+<shift>+s",
        },
        "models": {
            "stt_engine": "faster-whisper",  # stt engine: faster-whisper, mlx-whisper, or vosk
            "stt_model_name": "large-v3",  # whisper model name for auto-download
            "stt_model_path": "~/.meetcap/models/whisper-large-v3",  # will be created automatically
            "mlx_stt_model_name": "mlx-community/whisper-large-v3-turbo",  # mlx whisper model
            "mlx_stt_model_path": "~/.meetcap/models/mlx-whisper",  # mlx models directory
            "vosk_model_name": "vosk-model-en-us-0.22",  # vosk model name
            "vosk_model_path": "~/.meetcap/models/vosk/vosk-model-en-us-0.22",  # vosk model directory
            "vosk_spk_model_path": "~/.meetcap/models/vosk/vosk-model-spk-0.4",  # speaker model directory
            "enable_speaker_diarization": False,  # enable speaker identification with vosk
            "llm_model_name": "Qwen3-4B-Thinking-2507-Q8_K_XL.gguf",  # qwen model name
            "llm_gguf_path": "~/.meetcap/models/Qwen3-4B-Thinking-2507-Q8_K_XL.gguf",  # auto-download path
        },
        "paths": {
            "out_dir": "~/Recordings/meetcap",
            "models_dir": "~/.meetcap/models",  # directory for auto-downloaded models
        },
        "llm": {
            "n_ctx": 32768,  # default context size (32k tokens)
            "n_threads": 6,
            "n_gpu_layers": 35,
            "n_batch": 1024,
            "temperature": 0.4,
            "max_tokens": 4096,  # increased for detailed summaries
        },
        "telemetry": {
            "disable": True,
        },
        "memory": {
            "aggressive_gc": True,  # enable aggressive garbage collection between models
            "enable_monitoring": False,  # enable memory monitoring and reporting
            "memory_report": False,  # print detailed memory report after processing
            "warning_threshold": 80,  # memory pressure warning threshold (percentage)
            "critical_threshold": 90,  # memory pressure critical threshold (percentage)
            "auto_fallback": True,  # automatic model fallback when memory is constrained
            "explicit_lifecycle": True,  # force explicit model loading/unloading
        },
        "notes": {
            "enable": True,  # enable manual notes feature
            "template": "# Meeting Notes\n\n*Add your notes here during or after the meeting*\n\n*This file will be included in the final summary*\n",  # template for new notes files
            "filename": "notes.md",  # default notes file name
        },
    }

    def __init__(self, config_path: Path | None = None):
        """
        initialize config.

        args:
            config_path: path to config file (default: ~/.meetcap/config.toml)
        """
        if config_path is None:
            config_path = Path.home() / ".meetcap" / "config.toml"

        self.config_path = config_path
        # deep copy to ensure test isolation
        import copy

        self.config = copy.deepcopy(self.DEFAULT_CONFIG)

        # load from file if exists
        if self.config_path.exists():
            self._load_from_file()
            self._migrate_config()  # apply any necessary migrations

        # apply environment variable overrides
        self._apply_env_overrides()

    def _load_from_file(self) -> None:
        """load configuration from toml file."""
        try:
            with open(self.config_path, "rb") as f:
                file_config = tomllib.load(f)

            # merge with defaults
            self._deep_merge(self.config, file_config)

        except Exception as e:
            console.print(f"[yellow]warning: failed to load config: {e}[/yellow]")

    def _apply_env_overrides(self) -> None:
        """apply environment variable overrides."""
        env_mapping = {
            "MEETCAP_DEVICE": ("audio", "preferred_device"),
            "MEETCAP_SAMPLE_RATE": ("audio", "sample_rate", int),
            "MEETCAP_CHANNELS": ("audio", "channels", int),
            "MEETCAP_HOTKEY": ("hotkey", "stop"),
            "MEETCAP_STT_ENGINE": ("models", "stt_engine"),
            "MEETCAP_STT_MODEL": ("models", "stt_model_path"),
            "MEETCAP_VOSK_MODEL": ("models", "vosk_model_name"),
            "MEETCAP_VOSK_MODEL_PATH": ("models", "vosk_model_path"),
            "MEETCAP_VOSK_SPK_MODEL": ("models", "vosk_spk_model_path"),
            "MEETCAP_ENABLE_DIARIZATION": ("models", "enable_speaker_diarization", bool),
            "MEETCAP_MLX_STT_MODEL": ("models", "mlx_stt_model_name"),
            "MEETCAP_LLM_MODEL": ("models", "llm_gguf_path"),
            "MEETCAP_OUT_DIR": ("paths", "out_dir"),
            "MEETCAP_N_CTX": ("llm", "n_ctx", int),
            "MEETCAP_N_THREADS": ("llm", "n_threads", int),
            "MEETCAP_N_GPU_LAYERS": ("llm", "n_gpu_layers", int),
            # memory management settings
            "MEETCAP_MEMORY_AGGRESSIVE_GC": (
                "memory",
                "aggressive_gc",
                lambda x: x.lower() == "true",
            ),
            "MEETCAP_MEMORY_MONITORING": (
                "memory",
                "enable_monitoring",
                lambda x: x.lower() == "true",
            ),
            "MEETCAP_MEMORY_REPORT": ("memory", "memory_report", lambda x: x.lower() == "true"),
            "MEETCAP_MEMORY_WARNING_THRESHOLD": ("memory", "warning_threshold", int),
            "MEETCAP_MEMORY_AUTO_FALLBACK": (
                "memory",
                "auto_fallback",
                lambda x: x.lower() == "true",
            ),
            # manual notes settings
            "MEETCAP_NOTES_ENABLE": (
                "notes",
                "enable",
                lambda x: x.lower() == "true",
            ),
            "MEETCAP_NOTES_TEMPLATE": ("notes", "template"),
            "MEETCAP_NOTES_FILENAME": ("notes", "filename"),
        }

        for env_var, path_spec in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # parse type if specified
                if len(path_spec) == 3:
                    section, key, type_func = path_spec
                    try:
                        value = type_func(value)
                    except ValueError:
                        continue
                else:
                    section, key = path_spec

                # set value
                if section in self.config:
                    self.config[section][key] = value

    def _deep_merge(self, base: dict, update: dict) -> None:
        """recursively merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _migrate_config(self) -> None:
        """apply config migrations for backward compatibility."""
        # migrate n_ctx from 8192 to 32768 if it's still at the old default
        if self.config.get("llm", {}).get("n_ctx") == 8192:
            console.print("[dim]migrating context size from 8k to 32k...[/dim]")
            self.config["llm"]["n_ctx"] = 32768
            # save the migration immediately
            self.save()

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        get configuration value.

        args:
            section: config section name
            key: config key name
            default: default value if not found

        returns:
            configuration value or default
        """
        return self.config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> dict[str, Any]:
        """
        get entire configuration section.

        args:
            section: section name

        returns:
            section dict or empty dict
        """
        return self.config.get(section, {})

    def save(self) -> None:
        """save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # convert to toml format
        import toml

        with open(self.config_path, "w") as f:
            toml.dump(self.config, f)

        console.print(f"[green]✓[/green] config saved to {self.config_path}")

    def create_default_config(self) -> None:
        """create default config file if it doesn't exist."""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.save()
            console.print(f"[green]✓[/green] created default config: {self.config_path}")
            console.print("[yellow]edit this file to customize your settings[/yellow]")

    def expand_path(self, path_str: str) -> Path:
        """
        expand path with ~ and environment variables.

        args:
            path_str: path string possibly with ~ or env vars

        returns:
            expanded path object
        """
        return Path(os.path.expanduser(os.path.expandvars(path_str)))
