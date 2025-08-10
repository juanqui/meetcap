"""configuration management for meetcap"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
        "hotkey": {
            "stop": "<cmd>+<shift>+s",
        },
        "models": {
            "stt_model_name": "large-v3",  # whisper model name for auto-download
            "stt_model_path": "~/.meetcap/models/whisper-large-v3",  # will be created automatically
            "llm_model_name": "Qwen3-4B-Thinking-2507-Q8_K_XL.gguf",  # qwen model name
            "llm_gguf_path": "~/.meetcap/models/Qwen3-4B-Thinking-2507-Q8_K_XL.gguf",  # auto-download path
        },
        "paths": {
            "out_dir": "~/Recordings/meetcap",
            "models_dir": "~/.meetcap/models",  # directory for auto-downloaded models
        },
        "llm": {
            "n_ctx": 8192,
            "n_threads": 6,
            "n_gpu_layers": 35,
            "n_batch": 1024,
            "temperature": 0.4,
            "max_tokens": 4096,  # increased for detailed summaries
        },
        "telemetry": {
            "disable": True,
        },
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        initialize config.
        
        args:
            config_path: path to config file (default: ~/.meetcap/config.toml)
        """
        if config_path is None:
            config_path = Path.home() / ".meetcap" / "config.toml"
        
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        # load from file if exists
        if self.config_path.exists():
            self._load_from_file()
        
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
            "MEETCAP_STT_MODEL": ("models", "stt_model_path"),
            "MEETCAP_LLM_MODEL": ("models", "llm_gguf_path"),
            "MEETCAP_OUT_DIR": ("paths", "out_dir"),
            "MEETCAP_N_CTX": ("llm", "n_ctx", int),
            "MEETCAP_N_THREADS": ("llm", "n_threads", int),
            "MEETCAP_N_GPU_LAYERS": ("llm", "n_gpu_layers", int),
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
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """recursively merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
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
    
    def get_section(self, section: str) -> Dict[str, Any]:
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