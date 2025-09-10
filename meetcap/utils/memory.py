"""memory monitoring and management utilities for meetcap"""

import gc
import os
import time

from rich.console import Console

console = Console()


def get_memory_usage() -> dict[str, float]:
    """
    get current process memory usage in MB.

    returns:
        dict with rss_mb, vms_mb, and percent fields
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Physical memory
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual memory
            "percent": process.memory_percent(),
        }
    except ImportError:
        # psutil not available, return zeros
        return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}


def check_memory_pressure(threshold_percent: float = 85) -> bool:
    """
    check if system memory pressure is high.

    args:
        threshold_percent: memory usage threshold to consider as high pressure

    returns:
        True if memory usage exceeds threshold, False otherwise
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        if memory.percent > threshold_percent:
            console.print(f"[yellow]warning: high memory usage ({memory.percent:.1f}%)[/yellow]")
            return True
        return False
    except ImportError:
        return False


def estimate_model_memory(model_type: str, model_size: str) -> int:
    """
    estimate memory requirements for a model.

    args:
        model_type: type of model (stt or llm)
        model_size: model size/name

    returns:
        estimated memory in MB
    """
    # rough estimates in MB
    estimates = {
        # STT models
        "whisper-large-v3": 1500,
        "whisper-large-v3-turbo": 1500,
        "whisper-small": 500,
        "whisper-tiny": 100,
        "mlx-whisper-large-v3-turbo": 1500,
        "mlx-community/whisper-large-v3-turbo": 1500,
        "vosk-small": 500,
        "vosk-standard": 1800,
        # LLM models
        "qwen2.5-3b": 3000,
        "qwen2.5-7b": 7000,
        "qwen2.5-14b": 14000,
    }

    # check for partial matches
    for key, value in estimates.items():
        if key in model_size.lower() or model_size.lower() in key:
            return value

    # default estimates based on model type
    if model_type == "stt":
        return 1500  # default 1.5GB for STT
    else:
        return 4000  # default 4GB for LLM


def safe_model_loading(load_func, model_name: str):
    """
    safely load model with memory monitoring.

    args:
        load_func: function to load the model
        model_name: name of model being loaded for logging

    raises:
        MemoryError: if insufficient memory to load model
    """
    initial_memory = get_memory_usage()

    try:
        load_func()
        final_memory = get_memory_usage()
        memory_delta = final_memory["rss_mb"] - initial_memory["rss_mb"]
        console.print(f"[dim]{model_name} loaded: +{memory_delta:.1f} MB[/dim]")

    except Exception as e:
        # check if memory-related
        error_str = str(e).lower()
        if "memory" in error_str or "allocation" in error_str or "malloc" in error_str:
            console.print(f"[red]memory exhaustion loading {model_name}[/red]")
            # trigger aggressive cleanup
            gc.collect()
            raise MemoryError(f"insufficient memory to load {model_name}") from e
        else:
            raise


def get_fallback_model(original_model: str, available_memory_mb: float) -> str | None:
    """
    get smaller model if memory is constrained.

    args:
        original_model: original model name
        available_memory_mb: available memory in MB

    returns:
        fallback model name or None if no fallback available
    """
    fallbacks = {
        "whisper-large-v3": "whisper-small",
        "whisper-large-v3-turbo": "whisper-small",
        "mlx-community/whisper-large-v3-turbo": "whisper-small",
        "vosk-standard": "vosk-small",
        "qwen2.5-7b": "qwen2.5-3b",
        "qwen2.5-14b": "qwen2.5-7b",
    }

    # check for partial matches
    for key, fallback in fallbacks.items():
        if key in original_model.lower() or original_model.lower() in key:
            if available_memory_mb < estimate_model_memory("", key):
                console.print(
                    f"[yellow]using smaller model {fallback} due to memory constraints[/yellow]"
                )
                return fallback

    return None


class MemoryMonitor:
    """monitor and track memory usage throughout processing pipeline."""

    def __init__(self):
        """initialize memory monitor."""
        self.checkpoints: dict[str, dict[str, float]] = {}
        self.start_time = time.time()

    def checkpoint(self, name: str, verbose: bool = True) -> dict[str, float]:
        """
        record memory usage at a specific point.

        args:
            name: name for this checkpoint
            verbose: whether to print checkpoint info

        returns:
            memory usage dict at this checkpoint
        """
        usage = get_memory_usage()
        self.checkpoints[name] = usage

        if verbose:
            console.print(f"[dim]memory checkpoint '{name}': {usage['rss_mb']:.1f} MB RSS[/dim]")

        return usage

    def get_delta(self, from_checkpoint: str, to_checkpoint: str) -> float:
        """
        get memory difference between two checkpoints.

        args:
            from_checkpoint: starting checkpoint name
            to_checkpoint: ending checkpoint name

        returns:
            memory difference in MB (positive means increase)
        """
        if from_checkpoint not in self.checkpoints or to_checkpoint not in self.checkpoints:
            return 0.0

        from_mem = self.checkpoints[from_checkpoint]["rss_mb"]
        to_mem = self.checkpoints[to_checkpoint]["rss_mb"]
        return to_mem - from_mem

    def report(self, detailed: bool = False):
        """
        print memory usage report.

        args:
            detailed: whether to include detailed breakdown
        """
        if not self.checkpoints:
            return

        console.print("\n[bold]Memory Usage Report[/bold]")

        # basic report - all checkpoints
        for name, usage in self.checkpoints.items():
            console.print(f"  {name}: {usage['rss_mb']:.1f} MB RSS, {usage['percent']:.1f}%")

        if detailed and len(self.checkpoints) > 1:
            # calculate deltas between major transitions
            console.print("\n[bold]Memory Deltas[/bold]")

            checkpoint_names = list(self.checkpoints.keys())
            for i in range(1, len(checkpoint_names)):
                prev_name = checkpoint_names[i - 1]
                curr_name = checkpoint_names[i]
                delta = self.get_delta(prev_name, curr_name)

                if delta > 0:
                    symbol = "↑"
                    color = "yellow" if delta > 500 else "dim"
                elif delta < 0:
                    symbol = "↓"
                    color = "green"
                else:
                    symbol = "→"
                    color = "dim"

                console.print(
                    f"  [{color}]{prev_name} → {curr_name}: {symbol} {abs(delta):.1f} MB[/{color}]"
                )

        # summary statistics
        all_rss = [usage["rss_mb"] for usage in self.checkpoints.values()]
        peak_memory = max(all_rss)
        min_memory = min(all_rss)

        console.print("\n[bold]Summary[/bold]")
        console.print(f"  Peak memory: {peak_memory:.1f} MB")
        console.print(f"  Min memory: {min_memory:.1f} MB")
        console.print(f"  Total variation: {peak_memory - min_memory:.1f} MB")

        elapsed = time.time() - self.start_time
        console.print(f"  Total time: {elapsed:.1f}s")


class MemoryError(Exception):
    """exception raised when memory constraints are encountered."""

    pass
