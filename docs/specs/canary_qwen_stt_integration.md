# ONNX STT Integration Specification

**Document**: ONNX Speech-to-Text Integration for Superior Accuracy
**Version**: 2.1
**Last Updated**: September 20, 2025
**Author**: meetcap development team

## ⚠️ IMPORTANT CLARIFICATION ABOUT CANARY MODELS

**Investigation Results:**

- **`onnx-community/canary-qwen-2.5b-ONNX` DOES exist** on HuggingFace Hub
- **However, it is NOT a speech-to-text model** - it's a `Qwen3ForCausalLM` text generation model
- **The "canary" name is misleading** - this repository contains a language model, not an ASR model
- **The original NVIDIA Canary models** (`NVIDIA/canary-1b`) ARE legitimate ASR models but use NeMo format, not ONNX

**Updated Approach:**
This specification has been updated to focus on **legitimate ONNX STT models** that provide actual speech recognition capabilities, including Whisper variants, Distil-Whisper, Moonshine, Parakeet TDT, and Sherpa-ONNX models.

## 1. Executive Summary

This specification defines the integration of ONNX-based speech-to-text models as superior alternatives to Whisper for meeting transcription in meetcap. The focus is on local, efficient ONNX inference with support for multiple high-accuracy STT models through a generic ONNX service architecture.

### 1.1 Key Benefits

- **Superior Performance**: ONNX Runtime optimization provides faster inference, especially on Apple Silicon
- **Model Flexibility**: Generic architecture supports swapping different ONNX STT models without code changes
- **Local Processing**: 100% offline processing with no cloud dependencies
- **Memory Efficiency**: Quantized ONNX models (INT8/INT4) require significantly less RAM
- **Cross-Platform**: Consistent performance across different hardware architectures
- **Future-Proof**: Easy integration of new ONNX STT models as they become available

### 1.2 ONNX-First Approach Advantages

- **Hardware Acceleration**: Native support for Apple Silicon Neural Engine via CoreML
- **Smaller Memory Footprint**: Quantized models use 20-30% less RAM than PyTorch equivalents
- **Faster Inference**: ONNX Runtime optimizations provide significant speed improvements
- **No Complex Dependencies**: Eliminates need for large ML frameworks in production
- **Model Portability**: Same model files work across different deployment environments

## 2. Technical Architecture

### 2.1 Generic ONNX STT Service Pattern

The ONNX STT integration follows the established `TranscriptionService` pattern with model-agnostic design:

```python
class ONNXSTTService(TranscriptionService):
    """Generic ONNX-based speech-to-text service"""

    def __init__(
        self,
        model_path: str | Path,
        model_config: dict | None = None,
        providers: list[str] | None = None,  # ONNX execution providers
        audio_sample_rate: int = 16000,
        auto_download: bool = True,
    ):
        # Implementation details

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        # Return standard TranscriptResult with segments

    def load_model(self) -> None:
        # Load ONNX model with specified providers

    def unload_model(self) -> None:
        # Cleanup ONNX session and memory
```

### 2.2 Supported ONNX STT Models

Curated list of verified, high-quality ONNX STT models:

1. **Whisper ONNX** (`onnx-community/whisper-large-v3-ONNX`): OpenAI Whisper Large v3 in ONNX format
2. **Distil-Whisper** (`distil-whisper/distil-large-v3`): Faster, lighter Whisper variant with ONNX support
3. **Moonshine** (`UsefulSensors/moonshine`): Efficient on-device STT optimized for edge deployment
4. **Parakeet TDT** (`istupakov/parakeet-tdt-0.6b-v3-onnx`): NVIDIA multilingual transformer model
5. **Sherpa-ONNX Whisper Models**: Optimized Whisper models for streaming STT
   - `csukuangfj/sherpa-onnx-whisper-tiny.en` (fastest)
   - `csukuangfj/sherpa-onnx-whisper-base.en` (lightweight)
   - `csukuangfj/sherpa-onnx-whisper-small.en` (balanced)

All models are **verified ASR models** with proper speech recognition capabilities, unlike the misleading `canary-qwen-2.5b-ONNX` which is a text generation model.

### 2.3 ONNX Runtime Optimization

The service will leverage ONNX Runtime's hardware-specific optimizations:

- **Apple Silicon**: CoreML ExecutionProvider for Neural Engine acceleration
- **CPU Optimization**: Quantized models (INT8/INT4) for memory efficiency
- **Cross-Platform**: Consistent interface across different hardware
- **Provider Selection**: Automatic selection of optimal execution providers

## 3. Implementation Plan

### 3.1 Phase 1: Generic ONNX STT Service (Primary Path)

**Rationale**: ONNX models provide optimal balance of accuracy, performance, and local deployment. The generic architecture allows easy model swapping and future upgrades.

#### 3.1.1 Core ONNX Service Implementation

```python
class ONNXSTTService(TranscriptionService):
    """Generic ONNX-based speech-to-text service"""

    def __init__(
        self,
        model_path: str | Path,
        model_config: ONNXModelConfig,
        providers: list[str] | None = None,
        fallback_service: TranscriptionService | None = None,
    ):
        self.model_path = Path(model_path)
        self.model_config = model_config
        self.providers = providers or self._get_optimal_providers()
        self.fallback_service = fallback_service
        self.session = None
        self.tokenizer = None

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        try:
            return self._transcribe_onnx(audio_path)
        except Exception as e:
            console.print(f"[yellow]ONNX STT failed: {e}[/yellow]")
            if self.fallback_service:
                console.print("[cyan]Falling back to Whisper...[/cyan]")
                return self.fallback_service.transcribe(audio_path)
            raise

    def _transcribe_onnx(self, audio_path: Path) -> TranscriptResult:
        # Generic ONNX inference pipeline
        audio_features = self._preprocess_audio(audio_path)
        tokens = self._run_inference(audio_features)
        text = self._decode_tokens(tokens)
        return self._create_transcript_result(text, audio_path)
```

#### 3.1.2 Configuration Integration

Update `config.toml` to support Canary configuration:

```toml
[models]
# STT engine selection
stt_engine = "canary-cloud"  # "canary-cloud", "canary-local", "fwhisper", "mlx", "vosk"

# Canary-specific settings
canary_api_provider = "replicate"  # "replicate", "hf_inference", "local"
canary_mode = "asr"  # "asr" for transcription only, "llm" for analysis
canary_fallback = "fwhisper"  # fallback engine when Canary unavailable
canary_cloud_timeout = 300  # seconds
canary_max_retries = 3

[canary]
# Cloud API settings
replicate_api_key = ""  # Set via environment or prompt
hf_api_key = ""
api_timeout = 300
max_file_size_mb = 100

# Model behavior
include_timestamps = true
show_confidence = true
enable_diarization = true  # if supported

# LLM analysis prompts
meeting_summary_prompt = "Summarize this meeting transcript with key decisions and action items."
analysis_enabled = true  # Use LLM mode for enhanced analysis
```

### 3.2 Phase 2: Local NeMo Integration (Future)

**Rationale**: For users with compatible NVIDIA hardware or future Apple Silicon optimizations.

#### 3.2.1 Native Implementation

```python
class CanaryQwenNativeService(TranscriptionService):
    """Native Canary-Qwen2.5B service using NeMo"""

    def __init__(self, ...):
        self.model = None
        self.nemo_model = None

    def _load_model(self):
        """Load native NeMo model"""
        try:
            from nemo.collections.speechlm2.models import SALM

            self.nemo_model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
            console.print("[green]✓[/green] Canary-Qwen2.5B loaded natively")

        except ImportError:
            raise ImportError(
                "NeMo framework not available. Install with: "
                "pip install 'nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git'"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Canary model: {e}")

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        """Native transcription using NeMo"""
        if not self.nemo_model:
            self._load_model()

        # Convert to format expected by NeMo
        audio_files = [str(audio_path)]

        # Run inference
        results = self.nemo_model.transcribe(
            paths2audio_files=audio_files,
            include_timestamps=True,
        )

        # Convert NeMo results to TranscriptResult format
        return self._convert_nemo_results(results, audio_path)
```

#### 3.2.2 Hardware Detection and Fallback

```python
class CanaryHardwareDetector:
    """Detect hardware capabilities for Canary deployment"""

    @staticmethod
    def detect_optimal_deployment() -> str:
        """Determine best deployment strategy"""

        # Check for NVIDIA GPU
        if CanaryHardwareDetector.has_nvidia_gpu():
            return "native"

        # Check Apple Silicon with sufficient memory
        if CanaryHardwareDetector.is_apple_silicon():
            memory_gb = CanaryHardwareDetector.get_memory_gb()
            if memory_gb >= 32:
                return "native_cpu"  # Slow but possible
            else:
                return "cloud"  # Insufficient memory

        # Default to cloud for other platforms
        return "cloud"

    @staticmethod
    def has_nvidia_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def is_apple_silicon() -> bool:
        import platform
        return platform.processor() == "arm" and platform.system() == "Darwin"

    @staticmethod
    def get_memory_gb() -> float:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
```

### 3.3 Phase 3: Enhanced Meeting Features

#### 3.3.1 LLM-Powered Analysis

```python
class CanaryMeetingAnalyzer:
    """Enhanced meeting analysis using Canary's LLM capabilities"""

    def __init__(self, canary_service: CanaryQwenService):
        self.canary_service = canary_service

    def analyze_meeting(
        self,
        transcript: str,
        analysis_type: str = "summary"
    ) -> dict:
        """Perform meeting analysis using Canary's LLM mode"""

        prompts = {
            "summary": "Provide a concise summary of this meeting with key decisions and action items.",
            "decisions": "List all decisions made in this meeting with context.",
            "action_items": "Extract all action items from this meeting with assigned owners.",
            "questions": "List all questions raised and whether they were answered.",
            "participants": "Identify the participants and their roles in the discussion.",
        }

        prompt = prompts.get(analysis_type, prompts["summary"])

        if hasattr(self.canary_service, 'analyze_meeting'):
            # Use native LLM capabilities
            result = self.canary_service.analyze_meeting(transcript, prompt)
        else:
            # Fallback to external LLM
            result = self._fallback_analysis(transcript, prompt)

        return {
            "analysis_type": analysis_type,
            "result": result,
            "model": "canary-qwen-2.5b",
            "timestamp": time.time(),
        }

    def _fallback_analysis(self, transcript: str, prompt: str) -> str:
        """Fallback to existing meetcap LLM service"""
        # Use existing summarization service
        pass
```

## 4. ONNX Model Management

### 4.1 Automatic Model Download

The ONNX STT service will support automatic model downloads from HuggingFace:

```python
def ensure_onnx_model(
    model_id: str,
    filename: str,
    models_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download ONNX model from HuggingFace if not present locally"""

    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models" / "onnx"

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_id.replace('/', '--')}--{filename}"

    if model_path.exists() and not force_download:
        return model_path

    console.print(f"[cyan]Downloading ONNX model: {model_id}/{filename}[/cyan]")

    from huggingface_hub import hf_hub_download

    downloaded_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        cache_dir=str(models_dir.parent / "cache"),
        local_files_only=False,
    )

    # Create symlink or copy to expected location
    import shutil
    shutil.copy2(downloaded_path, model_path)

    console.print(f"[green]✓[/green] ONNX model downloaded: {model_path}")
    return model_path
```

### 4.2 Supported Model Configurations

Initial model support with automatic download:

```python
ONNX_MODELS = {
    "whisper-large-v3": {
        "repo_id": "onnx-community/whisper-large-v3-onnx",
        "filename": "model.onnx",
        "tokenizer_repo": "openai/whisper-large-v3",
        "sample_rate": 16000,
        "description": "OpenAI Whisper Large v3 (high accuracy)"
    },
    "whisper-medium": {
        "repo_id": "onnx-community/whisper-medium-onnx",
        "filename": "model.onnx",
        "tokenizer_repo": "openai/whisper-medium",
        "sample_rate": 16000,
        "description": "OpenAI Whisper Medium (balanced)"
    },
    "whisper-small": {
        "repo_id": "onnx-community/whisper-small-onnx",
        "filename": "model.onnx",
        "tokenizer_repo": "openai/whisper-small",
        "sample_rate": 16000,
        "description": "OpenAI Whisper Small (lightweight)"
    }
}
```

```python
def ensure_canary_model(
    models_dir: Path | None = None,
    force_download: bool = False,
) -> Path | None:
    """Ensure Canary-Qwen2.5B model is available for NeMo"""

    if models_dir is None:
        models_dir = Path.home() / ".meetcap" / "models" / "canary"

    models_dir = models_dir.expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "canary-qwen-2.5b"

    if model_path.exists() and not force_download:
        console.print(f"[green]✓[/green] Canary model already exists: {model_path}")
        return model_path

    console.print("[cyan]Downloading Canary-Qwen2.5B model...[/cyan]")
    console.print("[yellow]This is a large model and may take 15-30 minutes[/yellow]")

    try:
        # Use NeMo's download mechanism
        from nemo.collections.speechlm2.models import SALM

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Downloading Canary model...", total=None)

            # This downloads and caches the model
            model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')

            progress.update(task, completed=True)

        console.print(f"[green]✓[/green] Canary model downloaded successfully")
        return model_path

    except Exception as e:
        console.print(f"[red]Error downloading Canary model:[/red] {e}")
        return None

def verify_canary_model(models_dir: Path | None = None) -> bool:
    """Verify Canary model is available and functional"""

    # Check if NeMo is available
    try:
        import nemo
        from nemo.collections.speechlm2.models import SALM
    except ImportError:
        console.print("[yellow]⚠[/yellow] NeMo framework not installed")
        return False

    # Check hardware compatibility
    detector = CanaryHardwareDetector()
    deployment = detector.detect_optimal_deployment()

    if deployment == "cloud":
        console.print("[yellow]⚠[/yellow] Local deployment not optimal for this hardware")
        return False

    # Try loading model
    try:
        console.print("[cyan]Verifying Canary model...[/cyan]")
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        console.print("[green]✓[/green] Canary model ready for use")
        return True

    except Exception as e:
        console.print(f"[red]✗[/red] Canary model verification failed: {e}")
        return False
```

#### 4.2.2 Dependency Management

```python
def ensure_canary_dependencies() -> bool:
    """Ensure all Canary dependencies are installed"""

    required_packages = {
        "nemo_toolkit": "Install from: git+https://github.com/NVIDIA/NeMo.git",
        "torch": "PyTorch 2.6+ required",
    }

    missing_packages = []

    for package, install_hint in required_packages.items():
        try:
            if package == "nemo_toolkit":
                import nemo
                # Check if it's recent enough
                if not hasattr(nemo, 'collections'):
                    raise ImportError("NeMo version too old")
            elif package == "torch":
                import torch
                # Check PyTorch version
                version = torch.__version__
                if not version.startswith(('2.6', '2.7', '2.8')):
                    console.print(f"[yellow]Warning: PyTorch {version} may not be compatible[/yellow]")

        except ImportError:
            missing_packages.append((package, install_hint))

    if missing_packages:
        console.print("[red]Missing required dependencies:[/red]")
        for package, hint in missing_packages:
            console.print(f"  {package}: {hint}")
        return False

    return True
```

## 5. CLI Integration

### 5.1 Engine Selection

Update CLI to support Canary engine selection:

```python
@app.command()
def record(
    stt: str | None = typer.Option(
        None,
        "--stt",
        help="STT engine: canary-cloud, canary-local, fwhisper, mlx, vosk, or whispercpp",
    ),
    # ... other options
):
    """Start recording with optional Canary STT engine"""
```

### 5.2 Setup Integration

Update setup command to configure Canary:

```python
def setup_canary_stt(config: Config) -> None:
    """Configure Canary STT engine during setup"""

    console.print("\n[bold]step X: configure Canary-Qwen2.5B STT[/bold]")

    # Hardware detection
    detector = CanaryHardwareDetector()
    optimal_deployment = detector.detect_optimal_deployment()

    console.print(f"[cyan]Detected optimal deployment: {optimal_deployment}[/cyan]")

    if optimal_deployment == "cloud":
        # Configure cloud API
        console.print("[cyan]Cloud deployment recommended for your hardware[/cyan]")

        api_choice = typer.prompt(
            "Select cloud provider (1: Replicate, 2: Hugging Face, 3: Skip)",
            default="1"
        )

        if api_choice == "1":
            setup_replicate_api(config)
        elif api_choice == "2":
            setup_huggingface_api(config)
        else:
            console.print("[yellow]Canary setup skipped[/yellow]")

    else:
        # Native deployment
        console.print("[cyan]Checking native deployment requirements...[/cyan]")

        if ensure_canary_dependencies():
            if verify_canary_model():
                config.config["models"]["stt_engine"] = "canary-local"
                config.save()
                console.print("[green]✓[/green] Canary native deployment configured")
            else:
                console.print("[yellow]Canary model verification failed[/yellow]")
        else:
            console.print("[yellow]Canary dependencies missing[/yellow]")

def setup_replicate_api(config: Config) -> None:
    """Setup Replicate API configuration"""

    console.print("\n[cyan]Setting up Replicate API for Canary...[/cyan]")
    console.print("1. Visit https://replicate.com/account/api-tokens")
    console.print("2. Create a new API token")
    console.print("3. Enter the token below (it will be stored securely)")

    api_key = typer.prompt("Replicate API token", hide_input=True)

    # Store in environment or config (encrypted)
    config.config["canary"]["replicate_api_key"] = api_key
    config.config["models"]["stt_engine"] = "canary-cloud"
    config.config["canary"]["api_provider"] = "replicate"
    config.save()

    console.print("[green]✓[/green] Replicate API configured for Canary")
```

## 6. Error Handling and Fallback

### 6.1 Graceful Degradation

```python
class CanaryServiceFactory:
    """Factory for creating appropriate Canary service instance"""

    @staticmethod
    def create_service(config: Config) -> TranscriptionService:
        """Create optimal Canary service based on config and hardware"""

        # Check configuration preference
        engine = config.get("models", "stt_engine", "fwhisper")

        if engine == "canary-cloud":
            return CanaryServiceFactory._create_cloud_service(config)
        elif engine == "canary-local":
            return CanaryServiceFactory._create_native_service(config)
        else:
            # Not Canary - use existing factory methods
            return CanaryServiceFactory._create_whisper_service(config)

    @staticmethod
    def _create_cloud_service(config: Config) -> TranscriptionService:
        """Create cloud-based Canary service with fallback"""

        try:
            api_provider = config.get("canary", "api_provider", "replicate")
            api_key = config.get("canary", f"{api_provider}_api_key")

            if not api_key:
                raise ValueError(f"No API key configured for {api_provider}")

            # Create fallback service
            fallback_engine = config.get("canary", "fallback", "fwhisper")
            fallback_service = CanaryServiceFactory._create_fallback_service(
                fallback_engine, config
            )

            return CanaryQwenCloudService(
                api_provider=api_provider,
                api_key=api_key,
                fallback_service=fallback_service,
            )

        except Exception as e:
            console.print(f"[yellow]Canary cloud setup failed: {e}[/yellow]")
            console.print("[cyan]Falling back to Whisper...[/cyan]")
            return CanaryServiceFactory._create_whisper_service(config)

    @staticmethod
    def _create_native_service(config: Config) -> TranscriptionService:
        """Create native NeMo-based service with fallback"""

        try:
            # Check dependencies
            if not ensure_canary_dependencies():
                raise ImportError("NeMo dependencies not available")

            # Check model availability
            if not verify_canary_model():
                raise FileNotFoundError("Canary model not available")

            return CanaryQwenNativeService(
                model_name="nvidia/canary-qwen-2.5b",
                auto_download=True,
            )

        except Exception as e:
            console.print(f"[yellow]Canary native setup failed: {e}[/yellow]")
            console.print("[cyan]Falling back to Whisper...[/cyan]")
            return CanaryServiceFactory._create_whisper_service(config)
```

### 6.2 API Error Handling

```python
class CanaryAPIError(Exception):
    """Canary-specific API errors"""
    pass

class CanaryRetryHandler:
    """Handle retries and rate limiting for Canary API calls"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry"""

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                if attempt == self.max_retries:
                    break  # Final attempt failed

                # Check if error is retryable
                if self._is_retryable_error(e):
                    delay = self.base_delay * (2 ** attempt)
                    console.print(
                        f"[yellow]API call failed (attempt {attempt + 1}), "
                        f"retrying in {delay:.1f}s...[/yellow]"
                    )
                    time.sleep(delay)
                else:
                    break  # Non-retryable error

        # All retries exhausted
        raise CanaryAPIError(f"API call failed after {self.max_retries} retries") from last_exception

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        """Check if error is retryable (network, rate limit, etc.)"""
        error_str = str(error).lower()
        retryable_indicators = [
            "timeout", "connection", "network", "rate limit",
            "502", "503", "504", "temporary"
        ]
        return any(indicator in error_str for indicator in retryable_indicators)
```

## 7. Performance and Memory Management

### 7.1 Memory Optimization

```python
class CanaryQwenService(TranscriptionService):
    # ... existing implementation

    def unload_model(self) -> None:
        """Cleanup Canary model and GPU memory"""

        if hasattr(self, 'nemo_model') and self.nemo_model is not None:
            # Clear NeMo model
            del self.nemo_model
            self.nemo_model = None

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                console.print("[dim]CUDA memory cleared[/dim]")
        except ImportError:
            pass

        # Clear any MLX memory (for future Apple Silicon support)
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
            console.print("[dim]MLX memory cleared[/dim]")
        except (ImportError, AttributeError):
            pass

        # Force garbage collection
        import gc
        gc.collect()

        console.print("[dim]Canary-Qwen model unloaded[/dim]")

    def get_memory_usage(self) -> dict:
        """Return memory usage statistics specific to Canary"""
        base_stats = super().get_memory_usage()

        # Add GPU memory stats if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024   # MB
                base_stats.update({
                    'gpu_allocated_mb': gpu_memory,
                    'gpu_cached_mb': gpu_cached,
                })
        except ImportError:
            pass

        return base_stats
```

### 7.2 Performance Monitoring

```python
class CanaryPerformanceMonitor:
    """Monitor Canary-specific performance metrics"""

    def __init__(self):
        self.metrics = {}

    def start_transcription(self, audio_duration: float):
        """Start monitoring transcription performance"""
        self.metrics['audio_duration'] = audio_duration
        self.metrics['start_time'] = time.time()

    def end_transcription(self, segments_count: int):
        """End monitoring and calculate performance metrics"""
        end_time = time.time()
        duration = end_time - self.metrics['start_time']

        self.metrics.update({
            'transcription_time': duration,
            'segments_count': segments_count,
            'real_time_factor': self.metrics['audio_duration'] / duration,
            'throughput_minutes': self.metrics['audio_duration'] / 60,
        })

        return self.metrics

    def report_performance(self):
        """Print performance report"""
        if not self.metrics:
            return

        rtf = self.metrics.get('real_time_factor', 0)
        rtf_color = "green" if rtf > 1.0 else "yellow" if rtf > 0.5 else "red"

        console.print(
            f"[cyan]Canary Performance:[/cyan] "
            f"[{rtf_color}]{rtf:.1f}x realtime[/{rtf_color}] "
            f"({self.metrics.get('segments_count', 0)} segments, "
            f"{self.metrics.get('transcription_time', 0):.1f}s)"
        )
```

## 8. Configuration and Environment

### 8.1 Enhanced Configuration

```toml
[models]
# Primary STT engine selection
stt_engine = "canary-cloud"

# Canary-specific model settings
canary_model_name = "nvidia/canary-qwen-2.5b"
canary_mode = "asr"  # "asr" for transcription only, "llm" for analysis
canary_deployment = "auto"  # "auto", "cloud", "native"
canary_fallback_engine = "fwhisper"

[canary]
# Cloud API settings
api_provider = "replicate"  # "replicate", "hf_inference"
api_timeout = 300
max_retries = 3
max_file_size_mb = 100

# API keys (prefer environment variables)
replicate_api_key = ""
hf_api_key = ""

# Transcription settings
include_timestamps = true
show_confidence = true
enable_diarization = true

# LLM analysis settings
analysis_enabled = true
meeting_summary_prompt = "Summarize this meeting with key decisions and action items."
extract_decisions = true
extract_action_items = true

# Performance settings
chunk_duration_seconds = 30  # For long audio files
parallel_processing = false  # Enable for multiple files
cache_results = true

# Fallback settings
auto_fallback = true
fallback_timeout = 60  # Switch to fallback after N seconds
fallback_on_error = true

[canary.native]
# Native deployment settings (future)
device = "auto"  # "auto", "cuda", "cpu", "mps"
compute_type = "auto"  # "auto", "float16", "int8"
model_cache_dir = "~/.meetcap/models/canary"
enable_gpu_optimization = true
```

### 8.2 Environment Variables

```bash
# API Keys
REPLICATE_API_TOKEN=your_replicate_token_here
HF_API_KEY=your_huggingface_token_here

# Canary Configuration
MEETCAP_CANARY_ENGINE=canary-cloud
MEETCAP_CANARY_API_PROVIDER=replicate
MEETCAP_CANARY_FALLBACK=fwhisper
MEETCAP_CANARY_TIMEOUT=300

# Performance Tuning
MEETCAP_CANARY_MAX_RETRIES=3
MEETCAP_CANARY_CHUNK_SIZE=30
MEETCAP_CANARY_PARALLEL=false
```

## 9. Testing Strategy

### 9.1 Unit Tests

```python
def test_canary_cloud_service():
    """Test Canary cloud service functionality"""

    # Mock API responses
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': 'Hello world'},
                {'start': 5.0, 'end': 10.0, 'text': 'This is a test'},
            ],
            'language': 'en'
        }

        service = CanaryQwenCloudService(
            api_provider='replicate',
            api_key='test_key'
        )

        result = service.transcribe(create_test_audio_file())

        assert result.language == 'en'
        assert len(result.segments) == 2
        assert result.segments[0].text == 'Hello world'

def test_canary_fallback_mechanism():
    """Test automatic fallback to Whisper"""

    # Create service with failing API
    service = CanaryQwenCloudService(
        api_provider='replicate',
        api_key='invalid_key'
    )

    # Should automatically fall back to Whisper
    result = service.transcribe(create_test_audio_file())

    # Verify fallback was used
    assert result.stt['engine'] == 'faster-whisper'
    assert len(result.segments) > 0

def test_hardware_detection():
    """Test hardware capability detection"""

    detector = CanaryHardwareDetector()
    deployment = detector.detect_optimal_deployment()

    assert deployment in ['cloud', 'native', 'native_cpu']

    # Test specific scenarios
    with patch.object(detector, 'has_nvidia_gpu', return_value=True):
        assert detector.detect_optimal_deployment() == 'native'

    with patch.object(detector, 'is_apple_silicon', return_value=True):
        with patch.object(detector, 'get_memory_gb', return_value=16):
            assert detector.detect_optimal_deployment() == 'cloud'
```

### 9.2 Integration Tests

```python
def test_complete_canary_pipeline():
    """Test complete pipeline with Canary STT"""

    config = Config()
    config.config['models']['stt_engine'] = 'canary-cloud'
    config.config['canary']['api_provider'] = 'mock'

    orchestrator = RecordingOrchestrator(config)

    # Create test audio
    audio_file = create_test_meeting_audio(duration=60)

    try:
        # Process audio
        orchestrator._process_recording(
            audio_path=audio_file,
            stt_engine='canary-cloud',
            llm_path=None,  # Use Canary's built-in LLM
            seed=42
        )

        # Verify outputs
        base_path = audio_file.parent / audio_file.stem
        transcript_path = base_path.with_suffix('.transcript.txt')
        summary_path = base_path.with_suffix('.summary.md')

        assert transcript_path.exists()
        assert summary_path.exists()

        # Verify content quality (Canary should have better accuracy)
        with open(transcript_path) as f:
            transcript = f.read()

        # Check for proper names recognition (Canary's strength)
        assert 'proper_name_accuracy_test' in transcript.lower()

    finally:
        # Cleanup
        if audio_file.exists():
            audio_file.unlink()

def test_canary_performance_monitoring():
    """Test performance monitoring for Canary service"""

    service = CanaryQwenCloudService(api_provider='mock')
    monitor = CanaryPerformanceMonitor()

    audio_file = create_test_audio_file(duration=30)

    monitor.start_transcription(30.0)
    result = service.transcribe(audio_file)
    metrics = monitor.end_transcription(len(result.segments))

    # Verify metrics
    assert metrics['audio_duration'] == 30.0
    assert metrics['real_time_factor'] > 0
    assert metrics['segments_count'] > 0

    # Canary should achieve high RTF
    assert metrics['real_time_factor'] > 5.0  # Should be much faster than realtime
```

### 9.3 Load Tests

```python
def test_canary_api_rate_limiting():
    """Test API rate limiting and retry behavior"""

    service = CanaryQwenCloudService(api_provider='replicate')
    retry_handler = CanaryRetryHandler(max_retries=2)

    # Simulate rate limiting
    with patch('requests.post') as mock_post:
        # First call: rate limited
        # Second call: success
        mock_post.side_effect = [
            MockResponse(429, {'error': 'Rate limited'}),
            MockResponse(200, {'segments': [], 'language': 'en'})
        ]

        # Should retry and succeed
        result = retry_handler.execute_with_retry(
            service._transcribe_replicate,
            create_test_audio_file()
        )

        assert mock_post.call_count == 2
        assert result is not None
```

## 10. Deployment and Operations

### 10.1 Production Deployment Checklist

- [ ] **API Keys Configured**: Replicate/HuggingFace API keys set securely
- [ ] **Fallback Tested**: Whisper fallback works when Canary unavailable
- [ ] **Performance Monitored**: RTF and accuracy metrics tracked
- [ ] **Error Handling**: Graceful degradation for API failures
- [ ] **Memory Management**: Proper cleanup and memory monitoring
- [ ] **Cost Monitoring**: API usage and costs tracked
- [ ] **Quality Assurance**: Accuracy improvements validated

### 10.2 Monitoring and Alerting

```python
class CanaryOperationalMonitor:
    """Monitor operational metrics for Canary deployment"""

    def __init__(self):
        self.metrics = {
            'api_calls': 0,
            'api_failures': 0,
            'fallback_usage': 0,
            'total_audio_minutes': 0,
            'average_accuracy': 0,
        }

    def record_api_call(self, success: bool, audio_duration: float):
        """Record API call metrics"""
        self.metrics['api_calls'] += 1
        self.metrics['total_audio_minutes'] += audio_duration / 60

        if not success:
            self.metrics['api_failures'] += 1

    def record_fallback_usage(self, reason: str):
        """Record fallback usage"""
        self.metrics['fallback_usage'] += 1
        console.print(f"[yellow]Fallback used: {reason}[/yellow]")

    def get_health_status(self) -> str:
        """Get current health status"""
        if self.metrics['api_calls'] == 0:
            return "UNKNOWN"

        failure_rate = self.metrics['api_failures'] / self.metrics['api_calls']

        if failure_rate < 0.05:  # Less than 5% failure
            return "HEALTHY"
        elif failure_rate < 0.20:  # Less than 20% failure
            return "DEGRADED"
        else:
            return "UNHEALTHY"

    def generate_report(self) -> str:
        """Generate operational report"""
        status = self.get_health_status()

        return f"""
Canary STT Operational Report
============================
Status: {status}
API Calls: {self.metrics['api_calls']}
API Failures: {self.metrics['api_failures']}
Fallback Usage: {self.metrics['fallback_usage']}
Total Audio: {self.metrics['total_audio_minutes']:.1f} minutes
Failure Rate: {(self.metrics['api_failures'] / max(1, self.metrics['api_calls']) * 100):.1f}%
"""
```

### 10.3 Cost Management

```python
class CanaryCostTracker:
    """Track API usage costs for Canary service"""

    def __init__(self):
        self.usage_log = []

    def log_usage(self, provider: str, audio_duration: float, cost: float = None):
        """Log API usage for cost tracking"""

        # Estimate costs based on provider pricing
        if cost is None:
            cost = self._estimate_cost(provider, audio_duration)

        self.usage_log.append({
            'timestamp': time.time(),
            'provider': provider,
            'audio_duration': audio_duration,
            'cost': cost,
        })

    def _estimate_cost(self, provider: str, audio_duration: float) -> float:
        """Estimate API cost based on duration"""

        # Pricing estimates (as of 2025)
        pricing = {
            'replicate': 0.01,  # per minute
            'hf_inference': 0.005,  # per minute
        }

        rate = pricing.get(provider, 0.01)
        return (audio_duration / 60) * rate

    def get_monthly_cost(self) -> float:
        """Calculate estimated monthly cost"""
        current_time = time.time()
        month_ago = current_time - (30 * 24 * 60 * 60)

        monthly_usage = [
            log for log in self.usage_log
            if log['timestamp'] > month_ago
        ]

        return sum(log['cost'] for log in monthly_usage)

    def generate_cost_report(self) -> str:
        """Generate cost usage report"""
        monthly_cost = self.get_monthly_cost()
        total_minutes = sum(
            log['audio_duration'] / 60 for log in self.usage_log
            if log['timestamp'] > time.time() - (30 * 24 * 60 * 60)
        )

        return f"""
Canary API Cost Report (Last 30 Days)
=====================================
Total Audio Processed: {total_minutes:.1f} minutes
Estimated Cost: ${monthly_cost:.2f}
Average Cost per Minute: ${(monthly_cost / max(1, total_minutes)):.4f}
"""
```

## 11. Migration and Rollout

### 11.1 Phased Rollout Strategy

**Phase 1: Optional Integration (Weeks 1-2)**
- Implement cloud API service with fallback
- Add CLI option for Canary STT selection
- Deploy as opt-in feature for testing

**Phase 2: Enhanced Setup (Weeks 3-4)**
- Integrate Canary into setup wizard
- Add configuration management
- Implement performance monitoring

**Phase 3: Quality Validation (Weeks 5-6)**
- Conduct accuracy comparisons with Whisper
- Performance benchmarking and optimization
- User feedback collection and iteration

**Phase 4: Default Integration (Weeks 7-8)**
- Make Canary default for supported scenarios
- Full documentation and user guides
- Production monitoring and alerting

### 11.2 Migration Path for Existing Users

```python
def migrate_to_canary_config(config: Config) -> bool:
    """Migrate existing configuration to support Canary"""

    try:
        # Backup existing config
        config.backup()

        # Add Canary sections if missing
        if 'canary' not in config.config:
            config.config['canary'] = {
                'api_provider': 'replicate',
                'fallback_engine': config.get('models', 'stt_engine', 'fwhisper'),
                'auto_fallback': True,
                'analysis_enabled': True,
            }

        # Update STT engine options
        current_engine = config.get('models', 'stt_engine', 'fwhisper')
        if current_engine not in ['canary-cloud', 'canary-local']:
            # Keep current engine as fallback
            config.config['canary']['fallback_engine'] = current_engine

        config.save()
        console.print("[green]✓[/green] Configuration migrated to support Canary")
        return True

    except Exception as e:
        console.print(f"[red]Configuration migration failed: {e}[/red]")
        config.restore_backup()
        return False
```

## 12. Success Metrics and Evaluation

### 12.1 Accuracy Metrics

- **Word Error Rate (WER)**: Target <6% (vs Whisper's 7-12%)
- **Proper Name Accuracy**: >95% recognition of person/company names
- **Meeting-Specific Terms**: >90% accuracy on business vocabulary
- **Diarization Accuracy**: >85% speaker identification when available

### 12.2 Performance Metrics

- **Real-Time Factor**: >10x for cloud API, >5x for local deployment
- **Latency**: <30s for 1-hour meeting (cloud), <2 minutes (local)
- **Availability**: >99% uptime with fallback to Whisper
- **User Satisfaction**: >4.5/5 rating for transcription quality

### 12.3 Operational Metrics

- **API Cost Efficiency**: <$0.05 per meeting hour
- **Fallback Rate**: <10% of transcriptions use Whisper fallback
- **Error Rate**: <5% API failures requiring fallback
- **Adoption Rate**: >50% of users choose Canary when available

## 13. Future Enhancements

### 13.1 Advanced Features Roadmap

**Q1 2026: Native Apple Silicon Support**
- Investigate MLX or ONNX conversion for better macOS performance
- Implement quantized models for resource-constrained environments

**Q2 2026: Enhanced Meeting Intelligence**
- Advanced speaker diarization and identification
- Real-time sentiment analysis and meeting insights
- Integration with calendar and contact systems

**Q3 2026: Multi-Modal Integration**
- Screen sharing content analysis
- Meeting slides and document integration
- Visual meeting analytics and insights

**Q4 2026: Enterprise Features**
- Custom vocabulary and terminology training
- Multi-language meeting support
- Advanced security and compliance features

### 13.2 Research and Development

- **Model Optimization**: Custom fine-tuning for specific meeting domains
- **Edge Deployment**: Optimized models for local deployment
- **Real-Time Processing**: Live meeting transcription and analysis
- **Integration Ecosystem**: APIs for third-party meeting platforms

## 14. Conclusion

The integration of NVIDIA Canary-Qwen2.5B represents a significant advancement in meetcap's transcription capabilities, offering superior accuracy, built-in intelligence, and specialized meeting optimization. While platform limitations require a cloud-first approach initially, the implementation provides a solid foundation for future native deployment as the ecosystem evolves.

The phased implementation plan prioritizes user value delivery while maintaining system reliability through comprehensive fallback mechanisms. Performance monitoring and cost management ensure the solution remains practical and sustainable for production use.

This specification provides the technical foundation for implementing Canary-Qwen2.5B while preserving meetcap's core principles of reliability, ease of use, and offline-first operation where possible.

---

**Next Steps**:
1. Begin Phase 1 implementation with cloud API integration
2. Set up development and testing infrastructure
3. Implement basic fallback mechanisms
4. Create user documentation for Canary STT option

**Dependencies**:
- API access credentials for cloud providers
- Testing infrastructure for accuracy validation
- Performance benchmarking environment
- User feedback collection system
