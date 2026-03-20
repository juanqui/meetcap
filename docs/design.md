# meetcap — Formal Design Document (v2)

## 0. Document Control

* **project**: meetcap — offline meeting recorder & summarizer (macOS)
* **version**: 2.0 (refreshed 2026-03-19 to reflect current architecture)
* **owner**: juan (engineering)
* **status**: living architecture document
* **scope**: detailed architecture for meetcap CLI (v1.3.x)

---

## 1. Overview

meetcap is a python cli for macos that captures both system audio (speaker) and microphone simultaneously, writes audio in OPUS (default), WAV, or FLAC format, transcribes locally with Parakeet TDT (default), MLX-Whisper, Faster-Whisper, Vosk, or Whisper.cpp (no network), optionally identifies speakers via sherpa-onnx diarization, and summarizes with Qwen3.5-4B via mlx-vlm (Metal GPU). users trigger recording from the terminal, stop with a global hotkey, and receive a transcript (with optional speaker labels) and AI-generated summary.

**key properties**

* 100% offline (no network) — local models, local processing
* multiple STT engines — Parakeet TDT (16x faster), MLX-Whisper, Faster-Whisper, Vosk, Whisper.cpp
* speaker diarization — sherpa-onnx (pyannote segmentation + 3dspeaker embeddings)
* compressed audio — OPUS (default, ~14 MB/hr), FLAC, WAV
* simple flow — one command to start, hotkey to stop, automatic stt + diarize + summarize
* mac audio routing via blackhole + aggregate device (user pre-setup)

---

## 2. System Context & Assumptions

**environment**

* macos 13+ on apple silicon (m-series)
* user has admin rights to install blackhole and brew ffmpeg

**audio routing** (performed once by user)

* multi-output device: built-in output/headphones + blackhole (user can hear audio)
* aggregate input device: blackhole + microphone with drift correction

**privacy & connectivity**

* all processing is local. no http calls and no auto-downloads

---

## 3. Requirements Summary (traceability)

* capture system+mic concurrently (aggregate device preferred)
* cli with `record`, `devices`, `verify`, `setup`, `summarize`, `reprocess`
* hotkey to stop (⌘+⇧+s default), timer controls (Ctrl+A prefix)
* output: audio (OPUS/FLAC/WAV) + transcript (.txt/.json) + markdown summary + notes.md
* stt: 5 engines — Parakeet TDT (default), MLX-Whisper, Faster-Whisper, Vosk, Whisper.cpp
* speaker diarization: sherpa-onnx (pyannote-3.0 segmentation + 3dspeaker embedding), works with any STT engine
* summarization: Qwen3.5-4B via mlx-vlm (Metal GPU, 262K context, 4-bit quantization)
* scheduled stop timer with hotkey extension/cancellation
* reprocessing: regenerate transcripts/summaries with different models
* setup wizard: interactive model selection and download
* offline by construction, reproducible via hatch env

---

## 4. Architecture

### 4.1 Component Diagram (logical)

* **cli** — typer-based command dispatch (record, devices, verify, setup, summarize, reprocess)
* **orchestrator** — recording lifecycle state machine (idle → recording → processing → done)
* **device discovery** — ffmpeg avfoundation listing & parsers
* **recorder** — subprocess wrapper around ffmpeg; handles graceful stop; OPUS/FLAC/WAV encoding
* **hotkey service** — global hotkeys (pynput) with prefix key (Ctrl+A) for timer operations
* **transcription service** — Parakeet TDT (default), MLX-Whisper, Faster-Whisper, Vosk, Whisper.cpp
* **diarization service** — sherpa-onnx speaker identification (post-STT step, engine-agnostic)
* **summarization service** — mlx-vlm with Qwen3.5-4B safetensors (Metal GPU)
* **model download manager** — huggingface_hub snapshot_download, model verification
* **config manager** — read/write `~/.meetcap/config.toml`, env overrides, config migration
* **memory monitor** — psutil-based memory tracking with checkpoints and reporting
* **timer service** — scheduled stop with hotkey-based extension and cancellation
* **logging** — console (rich) and optional file logs (operational only)
* **storage** — artifact naming, output directory, json schemas

### 4.2 Sequence: Record → Stop → Process

```
user          cli           orchestrator    recorder(ffmpeg)   hotkey    stt svc          llm svc          storage
 |  meetcap record  |              |                |            |          |                |                |
 |----------------->|              |                |            |          |                |                |
 |                  |  load cfg    |                |            |          |                |                |
 |                  |------------->|                |            |          |                |                |
 |                  |  start rec   |  spawn ffmpeg  |            |          |                |                |
 |                  |------------->|--------------->|            |          |                |                |
 |                  |  show timer  |                |            |          |                |                |
 |                  |<-------------|                |            |          |                |                |
 | press ⌘⇧S        |              | hotkey event   |            |          |                |                |
 |----------------->|--------------|--------------->| send 'q'   |          |                |                |
 |                  |              |                |<-----------|          |                |                |
 |                  | wait exit    |<---------------| exit       |          |                |                |
 |                  | transcribe   |--------------->|            |          |  run stt       |                |
 |                  |                              |            |          |--------------->|                |
 |                  | summarize    |                                               done     |  run llm       |
 |                  |--------------------------------------------------------------------->|---------------->|
 |                  | print paths  |                                                                      done|
 |                  |<-------------|
```

### 4.3 States (orchestrator)

* **idle** → **recording** (ffmpeg started) → **processing** (stt then llm) → **done** (artifacts ready)
* error transitions: any state → **error** with actionable message

---

## 5. External Dependencies

* **ffmpeg** (brew) — avfoundation capture, encoding (OPUS/FLAC/WAV), and mixing
* **blackhole** (installer) — virtual audio device for system audio capture
* **core python packages**: `pynput`, `rich`, `typer`, `tomli`/`tomllib`, `toml`, `urllib3`, `mlx-vlm[torch]`, `psutil`
* **optional STT extras**: `[stt]` faster-whisper, `[mlx-stt]` mlx-whisper, `[vosk-stt]` vosk+soundfile+scikit-learn
* **diarization** (implicit): `sherpa-onnx`, `librosa`, `soundfile`
* **hatch** — env and packaging

---

## 6. macOS Audio Routing Design

### 6.1 Target Setup

* **multi-output device**: real speakers/headphones + blackhole (monitoring preserved)
* **aggregate input device**: microphone (clock source) + blackhole (drift correction enabled)

### 6.2 Device Selection Heuristics

* prefer a single **aggregate** input for perfect sync
* fallback: capture blackhole and mic separately and mix with `amix`

### 6.3 Device Enumeration

* command: `ffmpeg -f avfoundation -list_devices true -i ""`
* parse lines for `[AVFoundation input device @]` and record `(index, name)` tuples
* allow selection by name substring or numeric index

---

## 7. Audio Capture Module

### 7.1 Command Templates

**single-input (aggregate) recommended**

```bash
# capture aggregate input as stereo 48k wav
ffmpeg -hide_banner -nostdin \
  -f avfoundation -i ":${AGG_INDEX}" \
  -ac 2 -ar 48000 -c:a pcm_s16le "${OUT}.wav"
```

**dual-input with amix (fallback)**

```bash
# capture blackhole and mic separately, then mix
ffmpeg -hide_banner -nostdin \
  -f avfoundation -i ":${BH_INDEX}" \
  -f avfoundation -i ":${MIC_INDEX}" \
  -filter_complex "amix=inputs=2:duration=longest:normalize=0" \
  -ar 48000 -c:a pcm_s16le "${OUT}.wav"
```

### 7.2 Process Control

* start via `subprocess.Popen` with pipes; print elapsed time
* **graceful stop**: send `'q'` to ffmpeg stdin; fallback to `terminate()` then `kill()` with timeouts
* ensure wav header is finalized (graceful path preferred)
* capture stderr tail for diagnostics on failure

### 7.3 File Naming & Paths

* recording directory: `~/Recordings/meetcap/YYYY_MMM_DD_MeetingTitle/`
* files: `recording.opus` (or `.flac`/`.wav`), `recording.transcript.txt`, `recording.transcript.json`, `recording.summary.md`, `notes.md`
* output directory default: `~/Recordings/meetcap` (configurable)

---

## 8. Hotkey Service

* library: `pynput.keyboard`
* default: `⌘+⇧+S` (configurable)
* permissions: input monitoring + accessibility; on error show guidance
* debounce: ignore repeated triggers within 500 ms

---

## 9. Transcription Service (STT)

### 9.1 Parakeet TDT (default, Apple Silicon)

* **model**: `mlx-community/parakeet-tdt-0.6b-v3` via MLX
* 16x faster than MLX-Whisper (~1.2s for 40s audio, 32.7x realtime)
* native word-level timestamps, confidence scores
* automatic fallback to MLX-Whisper on failure

### 9.2 MLX-Whisper (Apple Silicon)

* Metal-accelerated Whisper via `mlx-whisper` library
* models: large-v3-turbo, large-v3-mlx, small-mlx
* auto-downloads models on first use

### 9.3 Faster-Whisper (universal)

* CTranslate2-optimized Whisper, works on all platforms
* load model from local directory path; compute type `int8_float16` or `float16`
* api pattern: create model → `transcribe(audio)` → iterate segments

### 9.4 Vosk (legacy, with built-in diarization)

* Kaldi-based offline recognition with optional speaker model
* deprecated in favor of Parakeet + sherpa-onnx diarization

### 9.5 Whisper.cpp (CLI fallback)

* run cli with local ggml model path; request srt or json output
* parse into normalized segment schema

### 9.6 Speaker Diarization (post-STT step)

* **engine-agnostic**: runs after any STT engine, not coupled to transcription
* **backend**: sherpa-onnx with pyannote-3.0 segmentation ONNX model (~5MB) + 3dspeaker eres2net embedding (~38MB)
* **clustering**: agglomerative with configurable threshold (default 0.85)
* speaker assignment: max-overlap algorithm maps diarization segments to transcript segments
* legacy Vosk built-in diarization still available as fallback

### 9.7 Transcript JSON Schema

```json
{
  "audio_path": "string",
  "sample_rate": 48000,
  "language": "en",
  "segments": [
    { "id": 0, "start": 0.00, "end": 3.42, "text": "...", "speaker_id": 0, "confidence": 0.95 }
  ],
  "duration": 1234.56,
  "stt": { "engine": "parakeet-tdt", "model": "...", "version": "..." },
  "speakers": [{ "id": 0, "label": "Speaker 0" }],
  "diarization_enabled": true
}
```

### 9.8 Language Handling

* default to auto-detect; allow config override `language = "en"`

### 9.9 Performance

* Parakeet TDT: 32.7x realtime (40s audio on Apple Silicon)
* sherpa-onnx diarization: ~4x realtime
* combined STT + diarization: ~3.6x realtime

---

## 10. Summarization Service (LLM)

### 10.1 Runtime

* `mlx-vlm` with Metal GPU acceleration on Apple Silicon
* default model: **mlx-community/Qwen3.5-4B-MLX-4bit** (~2.9GB, 34.8 tok/s)
* alternative: **mlx-community/Qwen3.5-9B-MLX-4bit** (~5.6GB, 19.0 tok/s)
* models downloaded via `huggingface_hub.snapshot_download()` on first use
* 262K native context window — no chunking needed for most meetings

### 10.2 Prompting

* **system**: structured meeting notes with sections: title, summary, participants, discussion points, decisions, action items, notable quotes, meeting tone
* **user**: transcript text with speaker labels (if diarization enabled)
* temperature 0.4; max_tokens 4096
* thinking mode enabled — Qwen3.5 uses `<think>` tags internally, cleaned before output

### 10.3 Chunking Strategy

* threshold: 500K chars (~125K tokens) triggers chunking
* chunk size: 400K chars per chunk
* stitch partial summaries: for >1 chunk, create a final pass that merges and deduplicates

### 10.4 Output Format (Markdown)

```
## Meeting Title
...

## Summary
- ...

## Participants
- Speaker 0: ...

## Discussion Points
- ...

## Decisions
- ...

## Action Items
- [ ] owner — task

## Notable Quotes
- ...

## Meeting Tone
- ...

---
[full transcript appended below separator]
```

### 10.5 Meeting Title Extraction

* LLM generates a concise PascalCase title from the transcript
* used to rename the recording directory (e.g., `2025_Jan_15_TeamStandup/`)

---

## 11. Configuration

**file**: `~/.meetcap/config.toml`

```toml
[audio]
preferred_device = "Aggregate Device"
sample_rate = 48000
channels = 2
format = "opus"              # opus (default), flac, wav
opus_bitrate = 32            # kbps (6-510)
flac_compression_level = 5   # 0-8

[hotkey]
stop = "<cmd>+<shift>+s"

[models]
stt_engine = "parakeet"      # parakeet, mlx, fwhisper, vosk, whispercpp
stt_model_path = "~/.meetcap/models/whisper-large-v3"
parakeet_model_name = "mlx-community/parakeet-tdt-0.6b-v3"
llm_model_name = "mlx-community/Qwen3.5-4B-MLX-4bit"
enable_diarization = true
diarization_backend = "sherpa"  # sherpa or vosk

[paths]
out_dir = "~/Recordings/meetcap"

[llm]
temperature = 0.4
max_tokens = 4096

[memory]
aggressive_gc = true
warning_threshold = 80       # percent
explicit_lifecycle = true

[diarization]
sherpa_num_speakers = -1     # -1 for auto-detect
sherpa_cluster_threshold = 0.85  # 0.0-1.0
```

**overrides**: env vars `MEETCAP_*` take precedence; cli flags override both

**key env vars**: `MEETCAP_DEVICE`, `MEETCAP_STT_ENGINE`, `MEETCAP_LLM_MODEL`, `MEETCAP_OUT_DIR`, `MEETCAP_ENABLE_DIARIZATION`, `MEETCAP_PARAKEET_MODEL`, `MEETCAP_DIARIZATION_BACKEND`, `MEETCAP_SHERPA_NUM_SPEAKERS`, `MEETCAP_SHERPA_THRESHOLD`

---

## 12. CLI Design

* `meetcap record [--out DIR] [--device NAME] [--format {opus,flac,wav}] [--stt {parakeet,mlx,fwhisper,vosk,whispercpp}] [--llm MODEL] [--auto-stop MINUTES] [--diarize] [--num-speakers N]`
* `meetcap summarize FILE [--out DIR] [--stt ENGINE] [--llm MODEL] [--diarize]` — process existing audio file
* `meetcap reprocess DIR [--mode {stt,summary,both}] [--stt ENGINE] [--llm MODEL] [--yes]` — regenerate transcripts/summaries
* `meetcap devices` — lists avfoundation inputs with indices and highlights likely aggregate devices
* `meetcap verify` — checks ffmpeg availability, permissions, model file presence, output directory
* `meetcap setup` — interactive wizard: model selection, download, permission checks

**console ux**

* show a banner with selected device + sample rate
* progress for recording time
* progress bars: stt decoding and llm summary generation
* at end, print absolute paths to all artifacts

---

## 13. Error Handling & Exit Codes

**categories**

* configuration error (missing models, invalid device) → exit 2
* permission error (microphone/accessibility) → exit 3
* runtime error (ffmpeg failure, disk full) → exit 4

**strategy**

* human-readable message + remediation tip
* print last 30 lines of ffmpeg stderr on capture failures
* never partially overwrite existing files; append numeric suffix if conflict

---

## 14. Logging

* console: info-level operational logs; timestamps
* optional file log via `--log-file` (no transcript content by default)
* rotate per run; include versions of models and libraries in headers

---

## 15. Performance Plan

* target real-time (≈1.0x or better) transcription for 48 kHz stereo on m-series with medium/large-v3 depending on quant/engine
* ensure non-blocking ui while ffmpeg runs
* parallelize: stt waits for recording to stop in v1 (no streaming); consider streaming in v2
* cache: none in v1; models kept memory-resident only during run

---

## 16. Testing Strategy

**unit tests**

* parsers for ffmpeg device list
* json schema validation for transcript output
* config precedence (file, env, cli)

**integration tests**

* 10s synthetic audio (two tones) to validate amix and aggregate paths
* run stt on short clip and verify non-empty transcript
* run llm with stub transcript (short text) and verify markdown sections exist

**manual qa**

* permissions denial scenarios
* device hot-switch while recording
* very long meeting (e.g., 2h) — functional pass

---

## 17. Packaging & Reproducibility

* `pyproject.toml` managed by hatch; pinned minor versions
* hatch scripts for `meetcap:record`, `meetcap:devices`, `meetcap:verify`
* installation doc covers: brew ffmpeg, blackhole setup, model placement

---

## 18. Observability (Local Only)

* print elapsed time, stt duration, llm duration
* optional `--emit-meta` to write a small `YYYYmmdd-HHMMSS.meta.json` with timings and versions

---

## 19. Security & Privacy

* fully offline; no network sockets
* files written only to configured output dir
* no telemetry; logs exclude transcript content unless user passes `--log-content` (post‑v1 option)

---

## 20. Risks & Mitigations

* **audio drift**: prefer aggregate with drift correction; as fallback, amix
* **permissions friction**: add `verify` to catch early and provide step‑by‑step guidance
* **model memory**: offer smaller quant; document requirements; stream transcript chunks for llm
* **performance variability**: allow cpu fallback with clear notice; expose llm tuning knobs

---

## 21. Future Enhancements

* optional multi-track export (separate mic/system wavs)
* ~~diarization/speaker tags (offline)~~ — **implemented** (sherpa-onnx, 2026-03-19)
* session metadata capture (attendees) and calendar integration (still offline)
* tui and/or small menu bar helper app
* ScreenCaptureKit integration to replace BlackHole dependency (pending PyObjC maturity)
* migration from Hatch to uv for faster dependency management
* `parakeet-mlx` library evaluation as drop-in STT backend

---

## 22. Implementation Notes (for Agent)

* keep subprocess and hotkey handlers simple and robust; avoid complex threading where possible
* ensure that any code examples and inline comments in the repo use lower-case comments per user preference
* avoid importing libraries that try to fetch models by name; always point to local paths
* write clear errors with remediation (e.g., how to create aggregate device)
