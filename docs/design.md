# meetcap — Formal Design Document (v1)

## 0. Document Control

* **project**: meetcap — offline meeting recorder & summarizer (macOS)
* **version**: 1.0 (initial design)
* **owner**: juan (engineering)
* **status**: design baseline for implementation handoff
* **scope**: detailed architecture and implementation plan for v1 cli

---

## 1. Overview

meetcap is a python cli for macos that captures both system audio (speaker) and microphone simultaneously, writes a single wav, transcribes locally with whisper (no network), and summarizes with a local qwen3 4b model via llama.cpp. users trigger recording from the terminal, stop with a global hotkey, and receive a transcript and summary.

**key properties**

* 100% offline (no network) — local models, local processing
* simple v1 flow — one command to start, hotkey to stop, automatic stt + summarize
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
* cli with `record`, `devices`, `verify`
* hotkey to stop (⌘+⇧+s default)
* output: wav + transcript (.txt/.json) + markdown summary
* stt: whisper (faster-whisper default; whisper.cpp optional)
* summarization: qwen/qwen3-4b-thinking-2507 via llama-cpp-python (metal) with local gguf
* offline by construction, reproducible via hatch env

---

## 4. Architecture

### 4.1 Component Diagram (logical)

* **cli** — argparse / command dispatch
* **orchestrator** — recording lifecycle state machine (idle → recording → processing → done)
* **device discovery** — ffmpeg avfoundation listing & parsers
* **recorder** — subprocess wrapper around ffmpeg; handles graceful stop
* **hotkey service** — global hotkeys (pynput)
* **transcription service** — faster-whisper (primary) / whisper.cpp (alt)
* **summarization service** — llama-cpp-python with local qwen3 gguf
* **config manager** — read/write `~/.meetcap/config.toml`, env overrides
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

* **ffmpeg** (brew) — avfoundation capture and mixing
* **blackhole** (installer) — virtual audio device
* **python packages** (pinned): `pynput`, `rich`, `faster-whisper` (or `whispercpp`), `llama-cpp-python`, `tomli`/`tomllib` if py<3.11, `typer` or `argparse`
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

* base: `YYYYmmdd-HHMMSS` + suffixes (`.wav`, `.transcript.txt`, `.json`, `.summary.md`)
* output directory default: `~/Recordings/meetcap` (configurable)

---

## 8. Hotkey Service

* library: `pynput.keyboard`
* default: `⌘+⇧+S` (configurable)
* permissions: input monitoring + accessibility; on error show guidance
* debounce: ignore repeated triggers within 500 ms

---

## 9. Transcription Service (STT)

### 9.1 faster-whisper (primary)

* load model from **local directory path** (no auto-download). compute type `int8_float16` or `float16` when available
* api pattern: create model → `transcribe(audio, vad_filter=False, word_timestamps=False)` (v1: segment-level timestamps)
* iterate segments `{start, end, text}` and stream to `.json` & `.txt`

### 9.2 whisper.cpp (alternative)

* run cli with local ggml/gguf model path; request srt or json output
* parse into normalized segment schema

### 9.3 Transcript JSON Schema

```json
{
  "audio_path": "string",
  "sample_rate": 48000,
  "language": "en",
  "segments": [
    { "id": 0, "start": 0.00, "end": 3.42, "text": "..." }
  ],
  "duration": 1234.56,
  "stt": { "engine": "faster-whisper", "model_path": "...", "version": "..." }
}
```

### 9.4 Language Handling

* default to auto-detect; allow config override `language = "en"`

### 9.5 Performance Considerations

* chunk input for very long files if needed (streaming decode) to bound memory

---

## 10. Summarization Service (LLM)

### 10.1 Runtime

* `llama-cpp-python` with metal enabled (install-time flag or prebuilt wheel)
* load **qwen3-4b-thinking-2507.gguf** in quant such as `Q5_K_M` (local path from config)
* config: `n_ctx`, `n_threads`, `n_batch`, `n_gpu_layers` tuned for m-series

### 10.2 Prompting

* **system**: concise meeting notes with sections: summary, decisions, action items (owner, due)
* **user**: transcript text (chunked to fit context); prepend meeting title/time if provided
* temperature 0.2–0.6; max\_new\_tokens sized for \~1–2 pages

### 10.3 Chunking Strategy

* estimate tokens from chars (≈4 chars/token heuristic) to select chunk size
* stitch partial summaries: for >1 chunk, create a final pass that merges bullet lists and deduplicates action items

### 10.4 Output Format (Markdown)

```
## summary
- ...

## decisions
- ...

## action items
- [ ] owner — task (due: yyyy-mm-dd)
```

### 10.5 Determinism

* set `seed` if supported; expose `--seed` flag for repeatability

---

## 11. Configuration

**file**: `~/.meetcap/config.toml`

```toml
[audio]
preferred_device = "Aggregate Device"
sample_rate = 48000
channels = 2

[hotkey]
stop = "cmd+shift+s"

[models]
stt_model_path = "/models/whisper/medium.en"
llm_gguf_path = "/models/qwen/Qwen3-4B-Thinking-2507.Q5_K_M.gguf"

[paths]
out_dir = "~/Recordings/meetcap"

[llm]
n_ctx = 8192
n_threads = 6
n_gpu_layers = 35
n_batch = 1024

'telemetry'
disable = true
```

**overrides**: env vars `MEETCAP_*` take precedence; cli flags override both

---

## 12. CLI Design

* `meetcap record [--out DIR] [--device NAME_OR_INDEX] [--rate 48000] [--channels 2] [--stt {fwhisper,whispercpp}] [--llm PATH] [--seed N]`
* `meetcap devices` — lists avfoundation inputs with indices and highlights likely aggregate devices
* `meetcap verify` — checks ffmpeg availability, permissions, model file presence and readability, output directory

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
* diarization/speaker tags (offline)
* session metadata capture (title, attendees) and calendar integration (still offline)
* tui and/or small menu bar helper app

---

## 22. Implementation Notes (for Agent)

* keep subprocess and hotkey handlers simple and robust; avoid complex threading where possible
* ensure that any code examples and inline comments in the repo use lower-case comments per user preference
* avoid importing libraries that try to fetch models by name; always point to local paths
* write clear errors with remediation (e.g., how to create aggregate device)
