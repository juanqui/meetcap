"""logging utilities for meetcap"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()


class MeetcapLogger:
    """centralized logging for meetcap"""

    _instance: Optional["MeetcapLogger"] = None
    _logger: logging.Logger | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """initialize logger (singleton)."""
        if self._logger is None:
            self._logger = logging.getLogger("meetcap")
            self._logger.setLevel(logging.INFO)

            # console handler with rich formatting
            console_handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=True,
            )
            console_handler.setLevel(logging.INFO)
            self._logger.addHandler(console_handler)

    def add_file_handler(self, log_path: Path, level: int = logging.DEBUG) -> None:
        """
        add file handler for logging to file.

        args:
            log_path: path to log file
            level: logging level for file
        """
        # ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # create file handler
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)

        # format for file logs
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self.info(f"logging to file: {log_path}")

    def set_level(self, level: int) -> None:
        """set logging level."""
        self._logger.setLevel(level)

    def debug(self, message: str) -> None:
        """log debug message."""
        self._logger.debug(message)

    def info(self, message: str) -> None:
        """log info message."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """log warning message."""
        self._logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """log error message."""
        self._logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False) -> None:
        """log critical message."""
        self._logger.critical(message, exc_info=exc_info)


# global logger instance
logger = MeetcapLogger()


class ErrorHandler:
    """centralized error handling with user-friendly messages"""

    EXIT_SUCCESS = 0
    EXIT_GENERAL_ERROR = 1
    EXIT_CONFIG_ERROR = 2
    EXIT_PERMISSION_ERROR = 3
    EXIT_RUNTIME_ERROR = 4

    @staticmethod
    def handle_config_error(error: Exception) -> None:
        """
        handle configuration errors.

        args:
            error: exception that occurred
        """
        console.print(f"[red]configuration error:[/red] {error}")
        console.print("\n[yellow]suggestions:[/yellow]")
        console.print("  • check your config file: ~/.meetcap/config.toml")
        console.print("  • run 'meetcap verify' to check your setup")
        console.print("  • ensure model files exist at configured paths")
        sys.exit(ErrorHandler.EXIT_CONFIG_ERROR)

    @staticmethod
    def handle_permission_error(error: Exception) -> None:
        """
        handle permission errors.

        args:
            error: exception that occurred
        """
        console.print(f"[red]permission error:[/red] {error}")
        console.print("\n[yellow]macos permissions required:[/yellow]")
        console.print("\n1. [cyan]microphone access:[/cyan]")
        console.print("   system preferences > privacy & security > microphone")
        console.print("   → enable for your terminal app")
        console.print("\n2. [cyan]input monitoring (for hotkeys):[/cyan]")
        console.print("   system preferences > privacy & security > input monitoring")
        console.print("   → enable for your terminal app")
        sys.exit(ErrorHandler.EXIT_PERMISSION_ERROR)

    @staticmethod
    def handle_runtime_error(error: Exception) -> None:
        """
        handle runtime errors.

        args:
            error: exception that occurred
        """
        console.print(f"[red]runtime error:[/red] {error}")

        error_str = str(error).lower()

        if "ffmpeg" in error_str:
            console.print("\n[yellow]ffmpeg issue detected:[/yellow]")
            console.print("  • install ffmpeg: brew install ffmpeg")
            console.print("  • check device permissions")
            console.print("  • verify audio devices with 'meetcap devices'")
        elif "model" in error_str or "llama" in error_str or "whisper" in error_str:
            console.print("\n[yellow]model loading issue:[/yellow]")
            console.print("  • verify model files exist at configured paths")
            console.print("  • check available disk space and ram")
            console.print("  • ensure models are in correct format (gguf for llm)")
        elif "disk" in error_str or "space" in error_str:
            console.print("\n[yellow]storage issue:[/yellow]")
            console.print("  • check available disk space")
            console.print("  • verify output directory is writable")

        sys.exit(ErrorHandler.EXIT_RUNTIME_ERROR)

    @staticmethod
    def handle_general_error(error: Exception) -> None:
        """
        handle general/unknown errors.

        args:
            error: exception that occurred
        """
        console.print(f"[red]unexpected error:[/red] {error}")
        console.print("\n[yellow]please report this issue with the full error message[/yellow]")
        logger.error("uncaught exception", exc_info=True)
        sys.exit(ErrorHandler.EXIT_GENERAL_ERROR)
