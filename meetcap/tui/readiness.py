"""readiness checks for meetcap TUI — verifies engines and models are available."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field


@dataclass
class ReadinessIssue:
    """a single readiness problem."""

    component: str  # "stt", "diarization", "llm", "ffmpeg"
    severity: str  # "error" (blocks recording), "warning" (degraded)
    message: str
    fix_hint: str


@dataclass
class ReadinessResult:
    """result of a full readiness check."""

    issues: list[ReadinessIssue] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """true if no errors (warnings are ok)."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ReadinessIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ReadinessIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        """one-line summary for display."""
        if self.ready and not self.warnings:
            return "All systems ready"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)


def check_readiness() -> ReadinessResult:
    """check if all configured components are ready to use.

    checks: ffmpeg, stt engine package + model, llm model,
    diarization models (if enabled).
    """
    from meetcap.utils.config import Config

    config = Config()
    result = ReadinessResult()

    # 1. ffmpeg
    _check_ffmpeg(result)

    # 2. stt engine
    stt_engine = config.get("models", "stt_engine", "parakeet")
    _check_stt(result, config, stt_engine)

    # 3. llm model
    _check_llm(result, config)

    # 4. diarization (if enabled)
    enable_diar = config.get("models", "enable_speaker_diarization", True)
    diar_backend = config.get("models", "diarization_backend", "sherpa")
    if enable_diar and diar_backend == "sherpa":
        _check_diarization(result, config)

    return result


def _check_ffmpeg(result: ReadinessResult) -> None:
    """check if ffmpeg is installed."""
    import subprocess

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result.issues.append(
            ReadinessIssue(
                component="ffmpeg",
                severity="error",
                message="FFmpeg not found",
                fix_hint="Install with: brew install ffmpeg",
            )
        )


def _check_stt(result: ReadinessResult, config: object, engine: str) -> None:
    """check if the configured STT engine is ready."""
    models_dir = config.expand_path(  # type: ignore[union-attr]
        config.get("paths", "models_dir", "~/.meetcap/models")  # type: ignore[union-attr]
    )

    if engine == "parakeet":
        if not _is_package_installed("parakeet_mlx"):
            result.issues.append(
                ReadinessIssue(
                    component="stt",
                    severity="error",
                    message="parakeet-mlx not installed",
                    fix_hint="Install with: pip install parakeet-mlx",
                )
            )
            return
        # parakeet downloads on first use from HuggingFace, so just check package
    elif engine in ("faster-whisper", "fwhisper"):
        if not _is_package_installed("faster_whisper"):
            result.issues.append(
                ReadinessIssue(
                    component="stt",
                    severity="error",
                    message="faster-whisper not installed",
                    fix_hint="Install with: pip install faster-whisper",
                )
            )
            return
        from meetcap.services.model_download import verify_whisper_model

        model_name = config.get("models", "stt_model_name", "large-v3")  # type: ignore[union-attr]
        if not verify_whisper_model(model_name, models_dir):
            result.issues.append(
                ReadinessIssue(
                    component="stt",
                    severity="warning",
                    message=f"Whisper model '{model_name}' not downloaded",
                    fix_hint="Run: meetcap setup",
                )
            )
    elif engine in ("mlx-whisper", "mlx"):
        if not _is_package_installed("mlx_whisper"):
            result.issues.append(
                ReadinessIssue(
                    component="stt",
                    severity="error",
                    message="mlx-whisper not installed",
                    fix_hint="Install with: pip install mlx-whisper",
                )
            )
            return
        from meetcap.services.model_download import verify_mlx_whisper_model

        model_name = config.get(  # type: ignore[union-attr]
            "models", "mlx_stt_model_name", "mlx-community/whisper-large-v3-turbo"
        )
        if not verify_mlx_whisper_model(model_name, models_dir):
            result.issues.append(
                ReadinessIssue(
                    component="stt",
                    severity="warning",
                    message=f"MLX Whisper model '{model_name}' not downloaded",
                    fix_hint="Run: meetcap setup",
                )
            )
    elif engine == "vosk":
        if not _is_package_installed("vosk"):
            result.issues.append(
                ReadinessIssue(
                    component="stt",
                    severity="error",
                    message="vosk not installed",
                    fix_hint="Install with: pip install vosk",
                )
            )


def _check_llm(result: ReadinessResult, config: object) -> None:
    """check if the configured LLM model is available."""
    from meetcap.services.model_download import verify_mlx_llm_model

    model_name = config.get(  # type: ignore[union-attr]
        "models", "llm_model_name", "mlx-community/Qwen3.5-4B-MLX-4bit"
    )
    if not verify_mlx_llm_model(model_name):
        result.issues.append(
            ReadinessIssue(
                component="llm",
                severity="warning",
                message=f"LLM model '{model_name.split('/')[-1]}' not downloaded",
                fix_hint="Run: meetcap setup",
            )
        )


def _check_diarization(result: ReadinessResult, config: object) -> None:
    """check if sherpa-onnx diarization models are available."""
    if not _is_package_installed("sherpa_onnx"):
        result.issues.append(
            ReadinessIssue(
                component="diarization",
                severity="warning",
                message="sherpa-onnx not installed (diarization disabled)",
                fix_hint="Install with: pip install sherpa-onnx",
            )
        )
        return

    models_dir = config.expand_path(  # type: ignore[union-attr]
        config.get("paths", "models_dir", "~/.meetcap/models")  # type: ignore[union-attr]
    )
    seg_model = models_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    emb_model = models_dir / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

    if not seg_model.exists() or not emb_model.exists():
        result.issues.append(
            ReadinessIssue(
                component="diarization",
                severity="warning",
                message="Diarization models not downloaded",
                fix_hint="Run: meetcap setup",
            )
        )


def _is_package_installed(package_name: str) -> bool:
    """check if a python package is importable."""
    return importlib.util.find_spec(package_name) is not None
