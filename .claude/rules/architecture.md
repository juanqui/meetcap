# Architecture & Design

## Core Principle: Offline-First

All processing happens locally. No network calls, no telemetry, no external APIs. Models prefer `local_files_only=True` when available locally, with download fallback on first setup. This is a hard constraint at runtime.

## Audio Pipeline

```
Audio Input → FFmpeg (AVFoundation) → WAV → STT Service → Diarization → Transcript → LLM Service → Summary
```

- FFmpeg with AVFoundation for macOS audio capture
- Prefers Aggregate Device (BlackHole + Mic) for perfect sync
- 48kHz stereo WAV output
- Graceful shutdown via stdin `q` command (not SIGTERM)

## STT Engines (in order of recommendation)

1. **Parakeet TDT** (default, Apple Silicon) — 16x faster than Whisper, MLX-accelerated, native timestamps
2. **MLX-Whisper** (Apple Silicon) — Metal-accelerated, proven fallback
3. **Faster-Whisper** — CTranslate2 optimization, universal
4. **Vosk** — legacy speaker diarization support
5. **Whisper.cpp** — CLI fallback

Engine selection is explicit via config (`stt_engine`) or CLI (`--stt`). Parakeet is the default.

## Speaker Diarization

Diarization is enabled by default using sherpa-onnx (decoupled from STT engine):
- **Segmentation**: pyannote-3.0 ONNX model (~5MB)
- **Embedding**: 3dspeaker eres2net (~38MB)
- **Clustering**: agglomerative with configurable threshold (default 0.85)
- Works with any STT engine, runs as a post-STT step
- Legacy Vosk built-in diarization still available as fallback

## LLM Summarization

- Qwen3.5-4B via mlx-vlm with Metal GPU acceleration (Apple Silicon native)
- Default model: `mlx-community/Qwen3.5-4B-MLX-4bit` (~2.9 GB, 4-bit quantized)
- Optional: `mlx-community/Qwen3.5-9B-MLX-4bit` (~5.6 GB) for higher quality
- 262K native context window — most meetings fit in a single pass
- Automatic context batching for very long transcripts (>500K chars)
- Structured markdown output with full transcript appended
- `<think>` tag removal for thinking models

## Critical Implementation Details

- **Subprocess management**: FFmpeg runs as subprocess with careful stdin/stdout/stderr handling
- **Hotkey debouncing**: 0.5s debounce to prevent double-triggers
- **Model lazy loading**: models load on first use to reduce startup time
- **Transcript chunking**: automatic chunking when transcripts exceed LLM context window
- **Backup safety**: reprocessing creates `.backup` files before modification, restores on failure
- **Path resolution**: fuzzy matching for recording directories in reprocess command

## Output Structure

```
~/Recordings/meetcap/
├── 2025_Jan_15_TeamStandup/
│   ├── recording.opus             # audio (OPUS format)
│   ├── recording.transcript.txt   # plain text transcript
│   ├── recording.transcript.json  # transcript with timestamps
│   └── recording.summary.md      # AI-generated summary
```

## Models Directory

All models stored in `~/.meetcap/models/`. Downloaded automatically during `meetcap setup`.
