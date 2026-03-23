"""memory monitoring and management utilities for meetcap"""

import gc
import os
import time

from rich.console import Console

console = Console()

# memory estimates in MB — measured on Apple Silicon with unified memory
MODEL_MEMORY_ESTIMATES: dict[str, int] = {
    # STT models
    "parakeet-tdt-0.6b": 800,
    "parakeet-tdt-0.6b-v3": 800,
    "mlx-community/parakeet-tdt-0.6b-v3": 800,
    "whisper-large-v3": 1500,
    "whisper-large-v3-turbo": 1500,
    "whisper-small": 500,
    "whisper-tiny": 100,
    "mlx-whisper-large-v3-turbo": 1500,
    "mlx-community/whisper-large-v3-turbo": 1500,
    "vosk-small": 500,
    "vosk-standard": 1800,
    # LLM models (4-bit quantized sizes)
    "qwen3.5-4b-mlx-4bit": 3200,
    "mlx-community/qwen3.5-4b-mlx-4bit": 3200,
    "qwen3.5-9b-mlx-4bit": 6000,
    "mlx-community/qwen3.5-9b-mlx-4bit": 6000,
    "qwen2.5-3b": 3000,
    "qwen2.5-7b": 7000,
    "qwen2.5-14b": 14000,
    # diarization
    "sherpa-diarization": 250,
}

# minimum headroom (MB) to keep free after loading a model
MEMORY_HEADROOM_MB = 512


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


def get_available_memory_mb() -> float:
    """
    get available system memory in MB.

    returns:
        available memory in MB, or 0.0 if psutil is not available
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.available / 1024 / 1024
    except ImportError:
        return 0.0


def get_total_memory_mb() -> float:
    """
    get total system memory in MB.

    returns:
        total memory in MB, or 0.0 if psutil is not available
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.total / 1024 / 1024
    except ImportError:
        return 0.0


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
    # check for exact match first
    key_lower = model_size.lower()
    if key_lower in MODEL_MEMORY_ESTIMATES:
        return MODEL_MEMORY_ESTIMATES[key_lower]

    # check for partial matches
    for key, value in MODEL_MEMORY_ESTIMATES.items():
        if key in key_lower or key_lower in key:
            return value

    # default estimates based on model type
    if model_type == "stt":
        return 1500  # default 1.5GB for STT
    elif model_type == "diarization":
        return 250
    else:
        return 4000  # default 4GB for LLM


def check_memory_for_model(model_type: str, model_name: str) -> tuple[bool, float, float, str]:
    """
    check if there is enough available memory to load a model.

    args:
        model_type: type of model (stt, llm, diarization)
        model_name: model name/path

    returns:
        tuple of (sufficient, available_mb, required_mb, message)
    """
    available_mb = get_available_memory_mb()
    if available_mb == 0.0:
        # psutil not available, assume sufficient
        return (True, 0.0, 0.0, "")

    required_mb = estimate_model_memory(model_type, model_name) + MEMORY_HEADROOM_MB
    sufficient = available_mb >= required_mb

    if sufficient:
        message = ""
    else:
        deficit = required_mb - available_mb
        total_mb = get_total_memory_mb()
        message = (
            f"insufficient memory for {model_name}: "
            f"need ~{required_mb / 1024:.1f} GB but only "
            f"{available_mb / 1024:.1f} GB available "
            f"(total: {total_mb / 1024:.0f} GB). "
            f"free ~{deficit / 1024:.1f} GB or close other apps to proceed"
        )

    return (sufficient, available_mb, required_mb, message)


def preflight_memory_check(
    stt_model: str,
    llm_model: str,
    enable_diarization: bool = True,
) -> tuple[bool, str]:
    """
    pre-flight check: can the full pipeline (STT → diarization → LLM) run
    sequentially with available memory?

    models are loaded one at a time, so we only need enough memory for the
    largest single model, not all combined.

    args:
        stt_model: STT model name
        llm_model: LLM model name
        enable_diarization: whether diarization will run

    returns:
        tuple of (can_proceed, warning_message).
        can_proceed is False only when there is truly not enough memory.
        warning_message is non-empty when memory is tight.
    """
    available_mb = get_available_memory_mb()
    if available_mb == 0.0:
        return (True, "")

    stt_need = estimate_model_memory("stt", stt_model)
    llm_need = estimate_model_memory("llm", llm_model)
    diar_need = (
        estimate_model_memory("diarization", "sherpa-diarization") if enable_diarization else 0
    )

    # the pipeline loads models sequentially, so peak = max(single model) + headroom
    peak_need = max(stt_need, llm_need, diar_need) + MEMORY_HEADROOM_MB

    total_mb = get_total_memory_mb()

    if available_mb >= peak_need:
        return (True, "")

    deficit = peak_need - available_mb
    largest = "LLM" if llm_need >= stt_need else "STT"
    largest_gb = max(stt_need, llm_need) / 1024

    message = (
        f"low memory warning: the {largest} model needs ~{largest_gb:.1f} GB "
        f"but only {available_mb / 1024:.1f} GB is available "
        f"(total: {total_mb / 1024:.0f} GB). "
        f"free ~{deficit / 1024:.1f} GB or close other apps. "
        f"the process may hang or be killed by the OS if memory runs out"
    )

    # hard block if less than half of what's needed
    can_proceed = available_mb >= (peak_need * 0.5)

    return (can_proceed, message)


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
