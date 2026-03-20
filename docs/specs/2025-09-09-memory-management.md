# Memory Management Specification

**Document**: Memory Management for Model Lifecycle
**Version**: 1.0
**Last Updated**: January 10, 2025
**Author**: meetcap development team

## 1. Overview

This specification defines the memory management strategy for STT (Speech-to-Text) and LLM (Large Language Model) models in meetcap. The primary goal is to ensure optimal memory usage during the audio processing pipeline by properly unloading models when they are no longer needed.

### 1.1 Current Problem

The meetcap application currently exhibits suboptimal memory management:

1. **STT models remain in memory** after transcription is complete
2. **LLM models are loaded while STT models are still in memory**, causing unnecessary memory pressure
3. **No explicit cleanup** of model resources between processing stages
4. **Potential memory exhaustion** on systems with limited RAM when processing large files or using large models

### 1.2 Goals

- **Minimize peak memory usage** during audio processing pipeline
- **Ensure complete model unloading** before loading the next model
- **Provide deterministic cleanup** with explicit garbage collection
- **Maintain processing performance** while optimizing memory usage
- **Support graceful fallback** when memory constraints are encountered

## 2. Architecture Overview

### 2.1 Processing Pipeline

The current audio processing pipeline follows this sequence:

```
Audio Recording → STT Model Loading → Transcription → LLM Model Loading → Summarization
```

The improved pipeline should enforce proper memory management:

```
Audio Recording → STT Model Loading → Transcription → STT Model Unloading → LLM Model Loading → Summarization → LLM Model Unloading
```

### 2.2 Memory Lifecycle States

Each model service should support these lifecycle states:

1. **Unloaded**: No model in memory, minimal memory footprint
2. **Loading**: Model being loaded into memory
3. **Ready**: Model loaded and ready for inference
4. **Processing**: Model actively performing inference
5. **Unloading**: Model being explicitly removed from memory

## 3. Implementation Requirements

### 3.1 Service Interface Requirements

All transcription and summarization services must implement:

```python
class ModelService:
    def load_model(self) -> None:
        """Explicitly load model into memory"""

    def unload_model(self) -> None:
        """Explicitly unload model from memory with cleanup"""

    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory"""

    def get_memory_usage(self) -> dict:
        """Return current memory usage statistics"""
```

### 3.2 STT Service Requirements

#### 3.2.1 FasterWhisperService

```python
class FasterWhisperService(TranscriptionService):
    def unload_model(self) -> None:
        """Unload faster-whisper model and cleanup GPU/CPU resources"""
        if self.model is not None:
            # Release model reference
            del self.model
            self.model = None

            # Force garbage collection
            import gc
            gc.collect()

            # Clear GPU cache if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
```

#### 3.2.2 MlxWhisperService

```python
class MlxWhisperService(TranscriptionService):
    def unload_model(self) -> None:
        """Unload MLX-whisper model and cleanup Metal resources"""
        if self.model is not None:
            # MLX models don't have explicit cleanup, but we can clear references
            del self.model
            self.model = None

            # Clear MLX memory if available
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except (ImportError, AttributeError):
                pass

            # Force garbage collection
            import gc
            gc.collect()
```

#### 3.2.3 VoskTranscriptionService

```python
class VoskTranscriptionService(TranscriptionService):
    def unload_model(self) -> None:
        """Unload Vosk models and cleanup resources"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.spk_model is not None:
            del self.spk_model
            self.spk_model = None

        # Force garbage collection
        import gc
        gc.collect()
```

### 3.3 LLM Service Requirements

#### 3.3.1 SummarizationService

```python
class SummarizationService:
    def unload_model(self) -> None:
        """Unload llama-cpp-python model and cleanup resources"""
        if self.llm is not None:
            # llama-cpp-python handles cleanup in destructor
            del self.llm
            self.llm = None

            # Clear Metal/GPU cache if available
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except (ImportError, AttributeError):
                pass

            # Force garbage collection
            import gc
            gc.collect()
```

### 3.4 Orchestrator Requirements

The `RecordingOrchestrator` must be updated to enforce proper memory management:

```python
class RecordingOrchestrator:
    def _process_recording(self, ...):
        stt_service = None
        llm_service = None

        try:
            # STT Phase
            stt_service = self._create_stt_service(stt_engine)
            stt_service.load_model()  # Explicit loading

            transcript_result = stt_service.transcribe(audio_file)
            text_path, json_path = save_transcript(transcript_result, base_path)

            # Critical: Unload STT model before LLM
            stt_service.unload_model()
            stt_service = None  # Clear reference

            # Force garbage collection between models
            import gc
            gc.collect()

            # LLM Phase
            llm_service = self._create_llm_service(llm_path, seed)
            llm_service.load_model()  # Explicit loading

            summary = llm_service.summarize(transcript_text)
            summary_path = save_summary(summary, base_path)

        finally:
            # Cleanup in reverse order
            if llm_service:
                llm_service.unload_model()
            if stt_service:
                stt_service.unload_model()

            # Final garbage collection
            import gc
            gc.collect()
```

## 4. Memory Monitoring and Diagnostics

### 4.1 Memory Usage Tracking

Implement memory usage tracking throughout the pipeline:

```python
import psutil
import os

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Physical memory
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory
        'percent': process.memory_percent()
    }

class MemoryMonitor:
    def __init__(self):
        self.checkpoints = {}

    def checkpoint(self, name: str):
        """Record memory usage at a specific point"""
        self.checkpoints[name] = get_memory_usage()
        console.print(f"[dim]memory checkpoint '{name}': {self.checkpoints[name]['rss_mb']:.1f} MB[/dim]")

    def report(self):
        """Print memory usage report"""
        console.print("\n[bold]Memory Usage Report[/bold]")
        for name, usage in self.checkpoints.items():
            console.print(f"  {name}: {usage['rss_mb']:.1f} MB RSS, {usage['percent']:.1f}%")
```

### 4.2 Memory Pressure Detection

Add memory pressure detection to prevent OOM conditions:

```python
def check_memory_pressure(threshold_percent=85):
    """Check if system memory pressure is high"""
    memory = psutil.virtual_memory()
    if memory.percent > threshold_percent:
        console.print(f"[yellow]warning: high memory usage ({memory.percent:.1f}%)[/yellow]")
        return True
    return False

def estimate_model_memory(model_type, model_size):
    """Estimate memory requirements for a model"""
    # Rough estimates in MB
    estimates = {
        'whisper-large-v3': 1500,
        'whisper-small': 500,
        'mlx-whisper-large-v3-turbo': 1500,
        'qwen3-4b': 4000,
        'vosk-small': 500,
        'vosk-standard': 1800,
    }
    return estimates.get(model_size, 2000)  # Default 2GB
```

## 5. Error Handling and Recovery

### 5.1 Memory Exhaustion Recovery

Implement graceful handling of memory exhaustion:

```python
class MemoryError(Exception):
    pass

def safe_model_loading(load_func, model_name):
    """Safely load model with memory monitoring"""
    initial_memory = get_memory_usage()

    try:
        load_func()
        final_memory = get_memory_usage()
        memory_delta = final_memory['rss_mb'] - initial_memory['rss_mb']
        console.print(f"[dim]{model_name} loaded: +{memory_delta:.1f} MB[/dim]")

    except Exception as e:
        # Check if memory-related
        if 'memory' in str(e).lower() or 'allocation' in str(e).lower():
            console.print(f"[red]memory exhaustion loading {model_name}[/red]")
            # Trigger aggressive cleanup
            import gc
            gc.collect()
            raise MemoryError(f"Insufficient memory to load {model_name}") from e
        else:
            raise
```

### 5.2 Fallback Strategies

Implement fallback strategies for memory-constrained environments:

```python
def get_fallback_model(original_model, available_memory_mb):
    """Get smaller model if memory is constrained"""
    fallbacks = {
        'whisper-large-v3': 'whisper-small',
        'qwen3-4b': None,  # No smaller alternative
        'vosk-standard': 'vosk-small',
    }

    if available_memory_mb < 2000 and original_model in fallbacks:
        fallback = fallbacks[original_model]
        if fallback:
            console.print(f"[yellow]using smaller model {fallback} due to memory constraints[/yellow]")
            return fallback

    return original_model
```

## 6. Configuration Options

### 6.1 Memory Management Configuration

Add memory management options to `config.toml`:

```toml
[memory]
# Enable aggressive garbage collection between models
aggressive_gc = true

# Memory monitoring and reporting
enable_monitoring = true
memory_report = false  # Print detailed memory report

# Memory pressure thresholds (percentage)
warning_threshold = 80
critical_threshold = 90

# Automatic model fallback when memory is constrained
auto_fallback = true

# Force explicit model loading/unloading (vs lazy loading)
explicit_lifecycle = true
```

### 6.2 Environment Variables

Support memory management configuration via environment variables:

```bash
MEETCAP_MEMORY_AGGRESSIVE_GC=true
MEETCAP_MEMORY_MONITORING=true
MEETCAP_MEMORY_WARNING_THRESHOLD=80
MEETCAP_MEMORY_AUTO_FALLBACK=true
```

## 7. Testing Strategy

### 7.1 Memory Leak Detection

Implement tests to detect memory leaks:

```python
def test_stt_memory_cleanup():
    """Test that STT service properly cleans up memory"""
    initial_memory = get_memory_usage()

    service = FasterWhisperService(model_name="small", auto_download=True)
    service.load_model()

    loaded_memory = get_memory_usage()
    assert loaded_memory['rss_mb'] > initial_memory['rss_mb']  # Model loaded

    service.unload_model()
    import gc
    gc.collect()

    final_memory = get_memory_usage()
    # Allow for some memory overhead, but should be close to initial
    memory_delta = final_memory['rss_mb'] - initial_memory['rss_mb']
    assert memory_delta < 100, f"Memory not properly cleaned up: +{memory_delta:.1f} MB"

def test_model_transition_memory():
    """Test memory usage during STT->LLM transition"""
    memory_monitor = MemoryMonitor()

    memory_monitor.checkpoint('start')

    # Load STT
    stt_service = FasterWhisperService(model_name="small")
    stt_service.load_model()
    memory_monitor.checkpoint('stt_loaded')

    # Unload STT
    stt_service.unload_model()
    import gc
    gc.collect()
    memory_monitor.checkpoint('stt_unloaded')

    # Load LLM
    llm_service = SummarizationService(model_path="test_model.gguf")
    llm_service.load_model()
    memory_monitor.checkpoint('llm_loaded')

    # Verify STT memory was released before LLM loading
    stt_memory = memory_monitor.checkpoints['stt_loaded']['rss_mb']
    transition_memory = memory_monitor.checkpoints['stt_unloaded']['rss_mb']
    llm_memory = memory_monitor.checkpoints['llm_loaded']['rss_mb']

    # STT memory should be significantly reduced after unloading
    assert transition_memory < stt_memory - 200, "STT model not properly unloaded"

    llm_service.unload_model()
```

### 7.2 Integration Tests

Test the complete pipeline with memory monitoring:

```python
def test_complete_pipeline_memory():
    """Test memory usage through complete audio processing pipeline"""
    memory_monitor = MemoryMonitor()
    audio_file = create_test_audio_file()

    orchestrator = RecordingOrchestrator(config)

    # Enable memory monitoring
    orchestrator.enable_memory_monitoring = True

    try:
        orchestrator._process_recording(
            audio_path=audio_file,
            stt_engine='fwhisper',
            llm_path='test_model.gguf',
            seed=42
        )

        # Verify final memory usage is reasonable
        final_memory = get_memory_usage()
        assert final_memory['rss_mb'] < 1000, "Memory usage too high after processing"

    finally:
        if audio_file.exists():
            audio_file.unlink()
```

## 8. Performance Considerations

### 8.1 Loading/Unloading Overhead

- **Model loading time**: Each explicit load/unload cycle adds overhead
- **Cold start penalty**: First model loading is typically slower
- **GC pause**: Garbage collection may cause brief processing pauses

### 8.2 Optimization Strategies

1. **Async cleanup**: Perform model unloading in background where possible
2. **Memory pools**: Reuse memory allocations between models
3. **Smart caching**: Cache frequently used models while respecting memory limits
4. **Progressive loading**: Load model components on-demand

### 8.3 Benchmarking

Track memory management performance metrics:

```python
class MemoryBenchmark:
    def benchmark_model_lifecycle(self, model_class, model_args):
        """Benchmark model load/unload performance"""
        times = []

        for i in range(5):  # Multiple runs for average
            start_time = time.time()

            service = model_class(**model_args)
            service.load_model()
            load_time = time.time() - start_time

            start_unload = time.time()
            service.unload_model()
            unload_time = time.time() - start_unload

            times.append({
                'load_time': load_time,
                'unload_time': unload_time,
                'total_time': load_time + unload_time
            })

        return times
```

## 9. Migration Path

### 9.1 Phase 1: Add Unload Methods

1. Add `unload_model()` methods to all service classes
2. Implement basic cleanup logic (del model, gc.collect())
3. Add memory monitoring infrastructure
4. Update tests to verify basic cleanup

### 9.2 Phase 2: Orchestrator Integration

1. Update `RecordingOrchestrator` to call unload methods
2. Add explicit model lifecycle management
3. Implement memory pressure detection
4. Add configuration options for memory management

### 9.3 Phase 3: Advanced Features

1. Implement fallback strategies for memory constraints
2. Add detailed memory reporting and diagnostics
3. Optimize loading/unloading performance
4. Add integration tests for complete pipeline

### 9.4 Phase 4: Production Hardening

1. Add error recovery and graceful degradation
2. Implement memory leak detection in CI
3. Performance optimization and benchmarking
4. Documentation and user guidance

## 10. Success Criteria

The implementation will be considered successful when:

1. **Memory usage is minimized**: Peak memory usage reduced by at least 30% during processing
2. **Models are properly cleaned up**: Memory usage returns to baseline after each model unloading
3. **No memory leaks**: Extended testing shows stable memory usage over multiple processing cycles
4. **Graceful handling**: System handles memory pressure gracefully with appropriate fallbacks
5. **Performance maintained**: Model loading/unloading overhead is less than 10% of total processing time

## 11. References

- [llama-cpp-python memory management](https://github.com/abetlen/llama-cpp-python)
- [faster-whisper GPU memory handling](https://github.com/SYSTRAN/faster-whisper)
- [MLX memory management best practices](https://github.com/ml-explore/mlx)
- [Python garbage collection documentation](https://docs.python.org/3/library/gc.html)
- [psutil memory monitoring](https://psutil.readthedocs.io/en/latest/#memory)

---

**Next Steps**: Implement Phase 1 of the migration path by adding unload methods to all service classes and basic memory monitoring infrastructure.
