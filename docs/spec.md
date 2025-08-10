# Offline Meeting Recorder & Summarizer (macOS) — High‑Level Spec

## 1) Title

**meetcap** — an offline macOS CLI that records system (speaker) + microphone audio, transcribes locally, and generates a meeting summary with a local LLM.

---

## 2) Summary

meetcap is a Python-based command-line tool for macOS that captures both system audio and microphone simultaneously via BlackHole routing, saves a single mixed WAV, performs offline speech‑to‑text (Whisper family), and produces an offline LLM summary. The app never connects to the internet: models are local, and runtime network access is not used. The design favors a minimal v1 with a single command to record, hotkey to stop, then auto‑transcribe and summarize.

---

## 3) Goals

* One‑key workflow: start recording → stop with a global hotkey → get transcript + summary.
* 100% offline operation on Apple Silicon Macs.
* Accurate, timestamped transcription and useful, concise summaries (decisions, action items).
* Simple installation and repeatable environment via **hatch**.

### Non‑Goals (v1)

* GUI, diarization, speaker separation, or multi‑track mixing beyond simple mic+system mix.
* Cloud APIs, calendar integration, or auto‑meeting detection.
* Editing tools for transcripts or summaries.

---

## 4) Key Requirements

### Functional

1. Record **both** system output and the active microphone concurrently.
2. Use **BlackHole** to route system audio while still monitoring through speakers/headphones.
3. Provide a **single CLI** command to start; present a **global hotkey** (e.g., ⌘+⇧+S) to stop.
4. Save audio as **WAV (PCM 16‑bit)** at 48 kHz stereo.
5. Transcribe locally using **Whisper** (default: *faster‑whisper*/CTranslate2; alt: *whisper.cpp*).
6. Summarize locally using **Qwen/Qwen3‑4B‑Thinking‑2507** via **llama.cpp/llama‑cpp‑python** (Metal), with quantized **GGUF** model supplied by user.
7. Emit artifacts per session: `YYYYmmdd‑HHMMSS.wav`, `.transcript.txt`, `.transcript.json` (segments + timestamps), `.summary.md`.
8. Device discovery: list available AVFoundation input devices; allow selecting an **Aggregate** input (BlackHole + mic) **or** capture two inputs and mix.
9. Provide clear errors and guidance for **macOS permissions** (Microphone, Input Monitoring/Accessibility).

### Non‑Functional

1. **Offline‑only**: never perform network requests; do not import code paths that auto‑download.
2. **Performance**: on M-series Macs, target transcription ≤ real‑time for 1‑hour audio with medium Whisper; summary generation ≤ 30–60 s on 4B quant (guideline, not enforced).
3. **Robustness**: handle device changes gracefully; fail with actionable messages.
4. **Reproducibility**: pinned versions and isolation via **hatch**.
5. **Security/Privacy**: data remains on device; no telemetry, no logs containing content unless user opts in (v1: logs contain only operational info).

### Constraints & Assumptions

* User will pre‑install **BlackHole** and create a **Multi‑Output** (monitoring) and an **Aggregate** input (BlackHole + mic w/ drift correction). The CLI will help verify.
* **ffmpeg** available via Homebrew; **Python 3.10+** recommended.
* Models (Whisper and Qwen GGUF) are provided via **local file paths**.
* Apple Silicon (M1+). Intel may work but is not optimized in v1.

---

## 5) Interfaces (CLI)

* `meetcap record --out <dir> [--device <name_or_index>] [--rate 48000] [--channels 2] [--stt {fwhisper,whispercpp}] [--llm <gguf_path>]`

  * Starts capture, prints elapsed time/device info; shows "press ⌘+⇧+S to stop".
  * On stop: runs transcription → summarization → prints paths to artifacts.
* `meetcap devices`

  * Lists AVFoundation input devices (indices + names) and highlights likely Aggregate devices.
* `meetcap verify`

  * Quick checks: ffmpeg present, permissions granted, model files accessible.

---

## 6) Outputs & File Formats

* **Audio**: WAV, PCM s16le, 48 kHz, 2 channels.
* **Transcript (plain)**: UTF‑8 text.
* **Transcript (structured)**: JSON array of segments `{start, end, text}`.
* **Summary**: Markdown with sections: `## summary`, `## decisions`, `## action items` (owner, due date if present).

---

## 7) Dependencies & Runtime

* System: macOS 13+; **ffmpeg** (brew), **BlackHole** (manual install/setup by user).
* Python: `pynput`, `rich`, `llama-cpp-python` (Metal build), **either** `faster-whisper` **or** `whispercpp`.
* Packaging: **hatch** (`pyproject.toml` with pinned versions; scripts for CLI entry points).

---

## 8) Architecture (High Level)

* **capture**: ffmpeg/avfoundation → single WAV (preferred: Aggregate device) or dual‑input + `amix` filter.
* **control**: Python process supervises ffmpeg (Popen), handles hotkey (pynput) to stop gracefully.
* **stt**: wrapper over faster‑whisper (segments iterator) or whisper.cpp CLI; produces `.txt` + `.json`.
* **summarizer**: llama‑cpp‑python chat call w/ Qwen3 GGUF; prompt template enforces concise notes.
* **io & config**: `~/.meetcap/config.toml` for defaults (device, model paths, sample rate).
* **logging**: console (rich) + optional file log without transcript content.

---

## 9) Prompts (Summary)

* **system**: “you are a note‑taker. output concise meeting notes with sections: summary, decisions, action items (owner, due). do not include chain‑of‑thought.”
* **user**: transcript text (chunked if needed), optional context (title, attendees).
* Parameters: temperature 0.2–0.6; max tokens sized for \~1–2 pages.

---

## 10) Risks & Mitigations

* **desync/drift**: prefer Aggregate input with drift correction; otherwise mix with `amix`.
* **permissions**: guide user to enable Microphone and Input Monitoring; detect and warn.
* **model RAM/VRAM**: choose smaller quant (Q5\_K\_M) if 4B model is tight; stream transcript to reduce prompt length.
* **performance variance**: allow CPU fallback with clear warning; keep progress indicators.

---

## 11) Acceptance Criteria (v1)

* Can record a 5–10 min sample including both system audio and mic; audible monitoring works.
* Stopping via global hotkey reliably terminates recording and triggers local STT + summary.
* Artifacts exist and are readable: `.wav`, `.transcript.txt/.json`, `.summary.md`.
* Runs with network disabled; no attempted downloads; model paths are local.
* Works with Aggregate device **and** dual‑input `amix` configuration.

---

## 12) High‑Level Tasks (Implementation)

1. **Project bootstrap**

   * Initialize repo; add `pyproject.toml` with hatch; pin deps; pre‑commit config (lint/format).
2. **Device discovery & verification**

   * `meetcap devices`: shell out to `ffmpeg -f avfoundation -list_devices true -i ""`; parse indices; highlight Aggregate candidates.
   * `meetcap verify`: check ffmpeg, permissions, model file existence, output dir.
3. **Recorder module**

   * Popen wrapper for ffmpeg single‑input (Aggregate) pipeline; configurable sample rate/channels.
   * Optional dual‑input pipeline with `amix=inputs=2:duration=longest`.
   * Graceful stop: send `q` or terminate, ensure WAV headers finalize.
4. **Hotkey module**

   * GlobalHotKeys (pynput) for ⌘+⇧+S; permission prompts/help on failure.
5. **Transcription module (STT)**

   * Option A: faster‑whisper (local model dir) → segments iterator → write `.txt` + `.json`.
   * Option B: whisper.cpp CLI path support; parse output or SRT/VTT.
6. **Summarizer module (LLM)**

   * llama‑cpp‑python loader for local Qwen3‑4B‑Thinking GGUF; Metal enabled; chat template aware.
   * Chunk transcript if needed; generate `summary.md` with required sections.
7. **CLI glue**

   * `record`, `devices`, `verify` commands; rich status; timestamps; structured errors.
8. **Configuration & paths**

   * `~/.meetcap/config.toml`: default device name/index, model paths, hotkey, output dir.
9. **Logging**

   * Operational logs only; optional `--log-file` flag; avoid content logging by default.
10. **Packaging & distribution**

    * Hatch scripts; simple install doc; Homebrew/BlackHole setup notes.
11. **QA & tests**

    * Unit tests for parsers and path logic; smoke test that runs a 10s capture and validates artifacts.
12. **Benchmarks (manual)**

    * Measure real‑time factor for STT on sample audio; check summary latency.

---

## 13) Deliverables

* Source repo with working CLI (v1).
* Setup guide (README): BlackHole routing, Aggregate device creation, permissions, model placement.
* Example config file; sample run outputs; short demo script.

---

## 14) Future Enhancements (post‑v1)

* Optional multi‑track capture (separate mic/system WAVs).
* Basic speaker attribution or diarization (offline).
* Simple TUI or lightweight menu bar helper.
* Meeting title/attendee capture; ICS export of action items.
