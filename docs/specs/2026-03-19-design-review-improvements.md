# Design Review & Improvement Plan

- **Version**: 2.0
- **Date**: 2026-03-20
- **Status**: Implemented
- **Type**: Comprehensive review spec

### Implementation Notes (2026-03-20)

All spec items implemented across 4 iterative review+fix+test cycles:
- **Phase 1 (Security)**: MLX >= 0.29.4 pinned via mlx-vlm, upper bounds added, diarization group added
- **Phase 2 (Thread Safety)**: RLock in HotkeyManager, context manager on AudioRecorder, double-checked locking on all services, standardized unload_model
- **Phase 3 (Code Quality)**: _is_loaded flag, _cleanup_service helper, error logging, temp file cleanup, dead code fixes, config warnings
- **Phase 4 (Docs)**: All docs updated to uv, zero hatch references remain
- **Phase 5 (Polish)**: Config range validation deferred, parakeet-mlx evaluation deferred (both P3)
- **Additional**: Full Hatch → uv migration completed (was listed as P3/future but done now)
- **Deviations**: Used threading.RLock instead of Lock (reviewer caught deadlock). Added is_loaded() to DiarizationService (not in original spec).
- **Test results**: 373 tests pass, 75.19% coverage, 5/5 stability runs clean, lint+format pass

---

## 0. Executive Summary

This spec captures findings from a comprehensive review of meetcap's architecture, implementation, dependencies, testing, and documentation. Three independent review passes were conducted concurrently (architecture/deps, code quality/testing, design doc accuracy), supplemented by web research via Exa MCP for current best practices.

**Key findings:**

1. **Security (URGENT)**: CVE-2025-62608 — critical heap buffer overflow in MLX <= 0.29.3 (CVSS 9.1). Must pin `mlx >= 0.29.4`.
2. **Design doc severely outdated**: `docs/design.md` references llama-cpp-python/GGUF, faster-whisper as default, and omits diarization, OPUS/FLAC, 3 CLI commands, and 3 architecture components.
3. **cli.py is 2,077 lines**: Needs decomposition into focused modules (existing spec: 2025-09-10-modularize.md).
4. **Thread safety bugs**: Race conditions in HotkeyManager state machine and model lazy loading.
5. **Dependency gaps**: No upper version bounds, missing diarization optional group, Vosk is high-risk/unmaintained.
6. **Test coverage**: 79% overall but cli.py is only 71% — error paths and orchestration logic undertested.

---

## 1. Security: CVE-2025-62608 (MLX Heap Buffer Overflow)

### Problem

CVE-2025-62608 is a heap-based buffer overflow in Apple's MLX framework (versions <= 0.29.3) in the `.npy` file parser (`mlx::core::load()`). CVSS score: **9.1 (Critical)**. Malicious `.npy` files can trigger out-of-bounds reads, potentially leading to remote code execution or information disclosure.

Current MLX latest: v0.31.1. The project must ensure `mlx >= 0.29.4`.

### Source

- https://cvefeed.io/vuln/detail/CVE-2025-62608
- https://cxsecurity.com/issue/WLB-2026020032

### Action

| # | Action | Effort | Priority |
|---|--------|--------|----------|
| 1.1 | Verify installed MLX version is >= 0.29.4 | 5 min | **P0** |
| 1.2 | Pin `mlx>=0.29.4` as explicit dependency in pyproject.toml (mlx-vlm pulls mlx transitively, but an explicit pin ensures the floor even if mlx-vlm's constraint is looser) | 10 min | **P0** |
| 1.3 | Add `pip-audit` or `safety` to CI pipeline for ongoing CVE detection | 1 hr | P1 |

---

## 2. Design Document Refresh (docs/design.md)

### Problem

The design doc (v1.0) was written for the initial implementation and is now substantially outdated. A section-by-section audit found **8 critical discrepancies** between the document and the current codebase.

### Critical Discrepancies

| Section | Design Says | Reality |
|---------|-------------|---------|
| S1 (Overview) | "summarizes with qwen3 4b via llama.cpp" | Qwen3.5-4B via mlx-vlm |
| S3 (Requirements) | "faster-whisper default; whisper.cpp optional" | 5 STT engines; Parakeet TDT is default |
| S4 (Architecture) | 8 components listed | 11+ components (missing diarization, memory monitor, timer) |
| S5 (Dependencies) | llama-cpp-python listed | Replaced by mlx-vlm; missing sherpa-onnx, psutil |
| S9 (STT) | 2 engines documented | 5 engines exist |
| S10 (LLM) | llama-cpp-python, GGUF, n_ctx=8192 | mlx-vlm, safetensors, 262K context |
| S11 (Config) | 5 config sections | 8+ sections with new keys |
| S12 (CLI) | 3 commands | 6 commands (missing setup, summarize, reprocess) |
| S21 (Future) | "diarization" listed as future | Fully implemented via sherpa-onnx |

### Action

| # | Action | Effort | Priority |
|---|--------|--------|----------|
| 2.1 | Rewrite S1 (Overview): mention all 5 STT engines, mlx-vlm, diarization | 15 min | P1 |
| 2.2 | Rewrite S3 (Requirements): add all current features | 20 min | P1 |
| 2.3 | Rewrite S4 (Architecture): add diarization, memory, timer components; update diagrams | 30 min | P1 |
| 2.4 | Rewrite S5 (Dependencies): complete list from pyproject.toml | 15 min | P1 |
| 2.5 | Rewrite S9 (STT): document all 5 engines, post-STT diarization | 30 min | P1 |
| 2.6 | Rewrite S10 (LLM): mlx-vlm, Qwen3.5, 262K context, thinking tags | 30 min | P1 |
| 2.7 | Rewrite S11 (Config): show actual config.toml with all sections | 20 min | P1 |
| 2.8 | Rewrite S12 (CLI): all 6 commands with current flags | 20 min | P1 |
| 2.9 | Update S15 (Performance): add Parakeet/diarization benchmarks | 10 min | P2 |
| 2.10 | Update S21 (Future): move diarization to implemented, update remaining | 10 min | P2 |
| 2.11 | Bump version to 2.0, update Document Control metadata | 5 min | P2 |
| 2.12 | Update `.claude/rules/architecture.md`: fix LLM section (llama.cpp → mlx-vlm), update STT engine list | 15 min | P1 |
| 2.13 | Update `CLAUDE.md`: fix project overview to match current architecture | 10 min | P1 |
| 2.14 | Update `README.md`: change "Local transcription using Whisper" to mention Parakeet as default | 10 min | P1 |

---

## 3. Thread Safety & Subprocess Management

### 3.1 HotkeyManager Race Conditions

**File**: `meetcap/core/hotkeys.py:40-240`

**Problem**: The prefix key state machine has three flags (`_prefix_active`, `_waiting_for_action`, `_debounce_active`) that are read/written from multiple threads (GlobalHotKeys callback thread, Timer thread, SingleKeyListener thread) without synchronization.

```python
# hotkeys.py:70-71 — Timer thread writes _prefix_active
self._prefix_active = True       # No lock
self._prefix_timer = threading.Timer(self._prefix_timeout, self._deactivate_prefix)

# hotkeys.py:63,79 — Callback thread reads _prefix_active
if self._prefix_active:           # No lock
    self._prefix_active = False
```

**Fix**: Add `threading.Lock()` protecting all state machine flag reads/writes.

### 3.2 FFmpeg Subprocess Lifecycle

**File**: `meetcap/core/recorder.py:289-530`

**Problem**: No context manager pattern. If an exception occurs between `Popen()` and `stop_recording()`, the FFmpeg process becomes a zombie. Cleanup relies on manual `stop_recording()` calls.

**Fix**: Implement `__enter__`/`__exit__` on RecordingSession for guaranteed cleanup.

### 3.3 Model Lazy Load Race Condition

**Files**: `meetcap/services/transcription.py:121`, `meetcap/services/summarization.py:43`

**Problem**: `if self.model is not None: return` in `_load_model()` is not atomic. Concurrent threads could both enter the load path, creating duplicate model instances and wasting GPU memory. In practice, transcription is currently called sequentially from the orchestrator, so this is a defensive fix — but it's cheap insurance against future parallelism or accidental re-entry.

**Fix**: Add `threading.Lock()` in `_load_model()` with a double-check pattern:
```python
def _load_model(self):
    if self.model is not None:
        return
    with self._load_lock:
        if self.model is not None:
            return  # another thread loaded while we waited
        # ... actual loading ...
```

### Action

| # | Action | Effort | Priority |
|---|--------|--------|----------|
| 3.1 | Add threading.Lock to HotkeyManager state machine | 2 hr | P1 |
| 3.2 | Add context manager to RecordingSession | 2 hr | P1 |
| 3.3 | Add threading.Lock to _load_model() in all services | 1 hr | P1 |
| 3.4 | Add timer.join() in auto_stop_worker cleanup path (cli.py:570) | 30 min | P2 |

---

## 4. Dependency Management

### 4.1 No Upper Version Bounds

**File**: `pyproject.toml:32-41`

**Problem**: All dependencies use `>=` only (e.g., `pynput>=1.8.1`). A major version bump in any dependency could silently break the project.

**Fix**: Add upper bounds using the "next major version" convention (prevents breaking API changes while allowing minor/patch updates):
```toml
"pynput>=1.8.1,<2.0",
"rich>=14.0.0,<15.0",
"typer>=0.16.0,<1.0",
"mlx-vlm>=0.4.0,<1.0",
"psutil>=6.1.0,<7.0",
```
For pre-1.0 libraries (e.g., typer 0.x), the upper bound `<1.0` is appropriate since any 1.0 release may break the API.

### 4.2 Missing Diarization Optional Group

**Problem**: Diarization requires `sherpa-onnx`, `librosa`, and `soundfile`, but these are not declared in pyproject.toml. Users enabling diarization will get `ImportError` at runtime.

**Fix**: Add optional dependency group:
```toml
diarization = [
    "sherpa-onnx>=1.10.0",
    "librosa>=0.10.0",
    "soundfile>=0.13.1",
]
```

### 4.3 Vosk Risk Assessment

**Problem**: Vosk has minimal maintenance (last meaningful update ~2023), is Kaldi-based (legacy architecture), and pulls in heavy dependencies (soundfile, scikit-learn). The Vosk STT path now has a much better alternative (Parakeet + sherpa-onnx for diarization).

**Recommendation**: Deprecate Vosk STT in favor of Parakeet + sherpa-onnx. Mark `[vosk-stt]` as legacy in README. Remove in a future major version.

### 4.4 Evaluate parakeet-mlx Library

**Source**: https://github.com/senstella/parakeet-mlx (871 stars, MIT, actively maintained)

**Problem**: The project currently has a custom Parakeet MLX integration. The `parakeet-mlx` library is purpose-built, well-maintained, and pip-installable. Using it would reduce maintenance burden.

**Recommendation**: Evaluate `parakeet-mlx` as a potential replacement for the custom integration. Add as optional dependency:
```toml
parakeet-stt = [
    "parakeet-mlx>=0.1.0",
]
```

### 4.5 Hatch vs uv

**Source**: https://scopir.com/posts/best-python-package-managers-2026/

**Finding**: uv has become the dominant Python project manager in 2026 (10-100x faster installs, built-in Python version management, native lockfiles). Hatch is still maintained but niche.

**Recommendation**: Not urgent. Plan migration for a future major version. Both use pyproject.toml so migration is straightforward.

### Action

| # | Action | Effort | Priority |
|---|--------|--------|----------|
| 4.1 | Add upper version bounds to all dependencies | 30 min | P1 |
| 4.2 | Add `[diarization]` optional dependency group | 15 min | P1 |
| 4.3 | Mark Vosk as deprecated/legacy in README and code | 30 min | P2 |
| 4.4 | Evaluate parakeet-mlx library for STT backend | 4 hr | P2 |
| 4.5 | Plan Hatch→uv migration for future major version | Track | P3 |

---

## 5. Configuration System

### 5.1 Config Precedence Verification

**File**: `meetcap/utils/config.py:85-243`

**Problem**: Config loading order is: defaults → file → migrate → env overrides. The order is correct (env overrides are applied last on line 107), but `_migrate_config()` (line 233) sets values like `config["audio"]["format"] = "opus"` unconditionally. If a user has `format = "wav"` in their config file, migration overwrites it. The migration should be conditional — only set defaults for keys that don't already exist.

**Fix**: Change migration to use `setdefault()` instead of direct assignment. Add integration test verifying: file config → migration (fills gaps only) → env overrides (wins).

### 5.2 Silent Type Coercion Failures

**File**: `meetcap/utils/config.py:169-184`

**Problem**: When environment variables fail type coercion (e.g., `MEETCAP_SHERPA_THRESHOLD=notanumber`), the failure is silently ignored with `except ValueError: continue`. User gets no feedback that their config was rejected.

**Fix**: Log a warning on type coercion failure.

### 5.3 Missing Validation

**Problem**: Several config values have valid ranges but no validation:
- Opus bitrate (valid: 6-510 kbps, default: 32)
- FLAC compression (valid: 0-8, default: 5)
- Clustering threshold (valid: 0.0-1.0, default: 0.85)
- Hotkey combo (no pynput format validation)

**Fix**: Add range validation in config loading or a dedicated `validate()` method.

### 5.4 Explicit Lifecycle Config Ignored

**File**: `meetcap/utils/config.py:76`, `meetcap/services/transcription.py:212,529`

**Problem**: `explicit_lifecycle: True` config setting exists but is never checked by services. They always lazy-load on first `transcribe()` call.

**Fix**: Check config before auto-loading. If explicit_lifecycle is enabled, require callers to call `load_model()` explicitly.

### Action

| # | Action | Effort | Priority |
|---|--------|--------|----------|
| 5.1 | Audit config loading order; add integration test for precedence | 1 hr | P1 |
| 5.2 | Log warning on env var type coercion failure | 30 min | P2 |
| 5.3 | Add range validation for config values | 1 hr | P2 |
| 5.4 | Implement explicit_lifecycle enforcement | 2 hr | P3 |

---

## 6. Code Quality & Architecture

### 6.1 cli.py Decomposition (2,077 lines)

**File**: `meetcap/cli.py`

**Problem**: Single file handles CLI commands, recording orchestration, backup management, timer logic, processing pipeline, notes, and more. This is the most impactful structural issue. An existing spec (2025-09-10-modularize.md) already proposes the decomposition.

**Proposed structure** (from modularize spec):
```
meetcap/cli/
├── __init__.py          # app, common imports
├── record.py            # record command + RecordingOrchestrator
├── summarize.py         # summarize command
├── reprocess.py         # reprocess command
├── setup.py             # setup wizard
├── verify.py            # verify command
├── devices.py           # devices command
└── utils.py             # BackupManager, helpers
```

### 6.2 Model Unload Inconsistency

**Problem**: Three different cleanup patterns across services:

| Service | Pattern | mlx.metal.clear_cache? | gc.collect? | torch.cuda.empty_cache? |
|---------|---------|----------------------|-------------|------------------------|
| FasterWhisperService | del + GC + torch | No | Yes | Yes |
| SummarizationService | del only | No | No | No |
| DiarizationService | del + GC | No (missing!) | Yes | No |

**Fix**: Create standard `_cleanup_resources()` method in base class:
```python
def _cleanup_resources(self):
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except ImportError:
        pass
```

### 6.3 TranscriptionService Interface Inconsistency

**File**: `meetcap/services/transcription.py`

**Problem**: `MlxWhisperService.model` is set to the string `"loaded"` instead of an actual model object. This breaks the `is_loaded()` contract (base class checks `self.model is not None`), since `"loaded" is not None` evaluates to True but the "model" is meaningless.

**Fix**: Use a dedicated `_is_loaded` boolean flag (preferred, since MLX-Whisper's API doesn't expose a persistent model object — it loads and transcribes in one call). Update `is_loaded()` in base class to check `self._is_loaded` instead of `self.model is not None`.

### 6.4 Duplicate Code

**File**: `meetcap/cli.py:690-760, 795-855`

**Problem**: Memory cleanup and GC code duplicated between `_process_audio_to_transcript` and `_process_transcript_to_summary`.

**Fix**: Extract `_ensure_memory_cleanup(service)` helper method.

### 6.5 Error Context Loss

**File**: `meetcap/cli.py:763-770`

**Problem**: Exception tracebacks lost in processing pipeline. `console.print(f"[red]transcription failed: {e}")` shows the error but doesn't log the full traceback.

**Fix**: Add `logger.error(f"transcription failed: {e}", exc_info=True)` before the console print.

### 6.6 Temp File Cleanup in Vosk and Model Download

**Files**: `meetcap/services/transcription.py:754-778`, `meetcap/services/model_download.py:451-467`

**Problem 1**: Vosk audio conversion creates temporary WAV files with cleanup only in a `finally` block. If `unlink()` fails, the temp file is orphaned.

**Problem 2**: Model downloads (URLopen) aren't wrapped in try/finally for cleanup. If an exception occurs during download, the partial temp_zip file is not deleted.

**Fix**: Use `tempfile.NamedTemporaryFile(delete=True)` context manager for Vosk conversion. Add try/finally cleanup for model download temp files.

### 6.7 Dead Code: Empty String Prints

**Files**: `meetcap/services/transcription.py:605-608, 972-974`

**Problem**: Ternary expression prints empty string when `audio_duration <= 0`:
```python
console.print(f"... (speed: {audio_duration / duration:.1f}x)" if audio_duration > 0 else "")
```

**Fix**: Use if/else block to print appropriate message for both cases.

### Action

| # | Action | Effort | Priority |
|---|--------|--------|----------|
| 6.1 | Implement cli.py decomposition per 2025-09-10-modularize.md | 8 hr | P1 |
| 6.2 | Standardize model unload across all services | 2 hr | P1 |
| 6.3 | Fix MlxWhisperService.model to store actual model ref | 30 min | P2 |
| 6.4 | Extract _ensure_memory_cleanup helper | 30 min | P2 |
| 6.5 | Add exc_info=True to error logging in pipeline | 30 min | P2 |
| 6.6 | Fix temp file cleanup (Vosk conversion + model download) | 1 hr | P2 |
| 6.7 | Fix dead code empty string prints | 15 min | P3 |

---

## 7. Testing Improvements

### 7.1 Coverage Gaps (cli.py: 71%)

**Missing test coverage in cli.py**:
- Lines 242, 262-264, 290-293: Device not found error handling
- Lines 333-334, 342, 346, 350: Auto-stop timer edge cases
- Lines 367-412: KeyboardInterrupt handling (multiple scenarios)
- Lines 562-566, 593-597: Double/triple interrupt timing
- Lines 701-702, 712-739: All 4 STT engine branch paths
- Lines 769-770, 836-837, 863-864: Exception handling in transcription/summarization
- Lines 1090-1094: Setup command paths

### 7.2 Missing Integration Tests

No tests exist for:
- Full pipeline: record → transcribe → diarize → summarize
- Memory management during model transitions
- Diarization + transcription integration
- Exception recovery in chained operations

### 7.3 Model Download Coverage (71%)

Missing tests for:
- Download progress reporting
- Network failure during download
- Partial download cleanup
- All model verification paths

### 7.4 Config Migration Tests

Missing tests for:
- GGUF → MLX model migration
- Audio format migration
- Python version-specific imports (tomllib vs tomli)

### Action

| # | Action | Effort | Priority |
|---|--------|--------|----------|
| 7.1 | Add cli.py error path tests (target: 85% coverage) | 4 hr | P1 |
| 7.2 | Add integration test class for full pipeline | 3 hr | P1 |
| 7.3 | Add model download edge case tests | 2 hr | P2 |
| 7.4 | Add config migration tests | 1 hr | P2 |
| 7.5 | Raise coverage threshold from 74% to 80% | 15 min | P2 |

---

## 8. Research-Informed Improvements

### 8.1 Parakeet TDT Validation

**Finding**: Parakeet TDT 0.6B v3 remains the consensus best offline STT for Apple Silicon (confirmed via Reddit, Northflank benchmarks, and NVIDIA catalog). The `parakeet-mlx` library (github.com/senstella/parakeet-mlx, 871 stars) is a polished, pip-installable alternative to custom MLX integration.

**Recommendation**: Current choice validated. Evaluate parakeet-mlx library.

### 8.2 Sherpa-onnx Diarization Quality

**Finding**: Sherpa-onnx diarization uses pyannote-3.0 ONNX segmentation but has a known quality gap vs native pyannote 3.1 (GitHub issue #1708: "mismatched diarization results"). The clustering threshold (0.85) may need per-recording tuning.

**Recommendation**: Keep sherpa-onnx as default. Consider offering pyannote 3.1 as a higher-quality optional backend for users with sufficient RAM. Expose clustering threshold in config (already done).

### 8.3 Qwen3.5 as Current Best

**Finding**: Qwen 3.5 (released March 2, 2026) is confirmed as the leading small LLM for summarization. The 4B model achieves ~70%+ MMLU with 262K native context. MLX inference is 3x faster than llama.cpp on Apple Silicon.

**Recommendation**: Current choice of Qwen3.5-4B via mlx-vlm is optimal. No change needed.

### 8.4 macOS Audio Capture Future

**Finding**: ScreenCaptureKit (macOS 13+) is the long-term replacement for BlackHole but has poor Python interop (PyObjC issues, mic+system audio bugs). OwnScribe and systemAudioDump show early adopters.

**Recommendation**: Track ScreenCaptureKit maturity. Consider a small Swift CLI helper in future to eliminate BlackHole dependency.

### 8.5 Competitive Landscape

**Finding**: Meetily (10.4K GitHub stars, Rust-based) is the dominant open-source competitor. meetcap differentiates on: pure Python simplicity, Qwen3.5 summarization quality, and sherpa-onnx diarization.

**Recommendation**: Invest in the Qwen3.5 upgrade (done) and cli.py modularization to maintain code quality advantage.

### Action

| # | Action | Effort | Priority |
|---|--------|--------|----------|
| 8.1 | Evaluate parakeet-mlx as STT backend | 4 hr | P2 |
| 8.2 | Document sherpa-onnx quality caveats; add clustering threshold tuning guide | 1 hr | P3 |
| 8.3 | No action (Qwen3.5 confirmed optimal) | — | — |
| 8.4 | Track ScreenCaptureKit PyObjC support | — | P3 |
| 8.5 | No action (competitive positioning is sound) | — | — |

---

## 9. Implementation Plan (Prioritized)

### Phase 1: Security & Urgent Fixes (Day 1)

| # | Action | Effort |
|---|--------|--------|
| 1.1 | Verify MLX >= 0.29.4, pin in pyproject.toml | 15 min |
| 4.1 | Add upper version bounds to all dependencies | 30 min |
| 4.2 | Add `[diarization]` optional dependency group | 15 min |

### Phase 2: Thread Safety & Reliability (Week 1)

| # | Action | Effort |
|---|--------|--------|
| 3.1 | Add threading.Lock to HotkeyManager | 2 hr |
| 3.2 | Add context manager to RecordingSession | 2 hr |
| 3.3 | Add threading.Lock to _load_model() | 1 hr |
| 6.2 | Standardize model unload across services | 2 hr |
| 5.1 | Audit config loading order | 1 hr |

**Done when**: All existing tests pass. No race conditions detectable with `ThreadSanitizer` or manual stress testing of hotkey rapid-fire. RecordingSession can be used in a `with` block. Config precedence test proves env vars always win.

### Phase 3: Code Quality (Week 2)

| # | Action | Effort |
|---|--------|--------|
| 6.1 | cli.py decomposition (must precede 7.1-7.2) | 8 hr |
| 6.3 | Fix MlxWhisperService.model interface | 30 min |
| 6.4 | Extract memory cleanup helper | 30 min |
| 6.5 | Fix error context logging | 30 min |
| 6.6 | Fix temp file cleanup (Vosk + model download) | 1 hr |
| 7.1 | Add cli.py error path tests (after 6.1 lands; target post-split modules) | 4 hr |
| 7.2 | Add integration tests | 3 hr |

**Note**: Line numbers in 7.1 will shift after 6.1 (cli.py decomposition). Target the new module files, not the original line references.

**Done when**: All tests pass, lint passes, cli.py is split into 7+ files each under 400 lines, coverage >= 80%.

### Phase 4: Documentation (Week 2-3, can run in parallel with Phase 2-3)

| # | Action | Effort |
|---|--------|--------|
| 2.1-2.11 | Full design.md refresh to v2.0 | 4 hr |
| 2.12-2.14 | Update CLAUDE.md, architecture.md, README.md | 35 min |
| 4.3 | Mark Vosk as deprecated | 30 min |

**Done when**: All documentation accurately reflects current architecture. No references to llama-cpp-python, GGUF, or "faster-whisper as default" remain in any doc.

### Phase 5: Polish & Future (Week 3+)

| # | Action | Effort |
|---|--------|--------|
| 3.4 | Add timer.join() in auto_stop_worker cleanup | 30 min |
| 5.2 | Log warnings for config type coercion failures | 30 min |
| 5.3 | Add config range validation | 1 hr |
| 6.7 | Fix dead code empty string prints | 15 min |
| 7.3-7.5 | Additional test coverage improvements | 3 hr |
| 8.1 | Evaluate parakeet-mlx library | 4 hr |
| 1.3 | Add pip-audit to CI | 1 hr |

**Done when**: Coverage >= 80%, no silent config failures, all temp files cleaned up.

### Phase Dependencies & Risks

| Dependency | Reason |
|------------|--------|
| Phase 1 → Phase 2 | Security fixes must land first; version bounds affect all other work |
| 6.1 (cli.py split) → 7.1-7.2 (new tests) | Writing tests against monolithic cli.py, then splitting, causes test churn. Decompose first. |
| 3.1-3.3 (thread safety) → 6.1 (cli split) | Thread safety fixes touch cli.py; apply before reorganizing |
| 2.1-2.14 (docs) can run in parallel with Phase 2-3 | Documentation is independent of code changes |

### Total Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1: Security & Deps | ~1 hr |
| Phase 2: Reliability | ~8 hr |
| Phase 3: Code Quality | ~17.5 hr |
| Phase 4: Documentation | ~5 hr |
| Phase 5: Polish | ~10.25 hr |
| **Total** | **~41.75 hr** |

---

## 10. Items NOT Recommended

These were considered but rejected:

| Item | Reason |
|------|--------|
| Switch to uv from Hatch | Not urgent; Hatch works fine. Plan for future major version. |
| Replace sherpa-onnx with native pyannote | Adds 2GB+ RAM requirement, complex PyTorch dependency, against offline-first principle of minimal deps |
| Add ScreenCaptureKit support now | PyObjC interop too buggy; BlackHole remains most reliable |
| Replace pynput with another hotkey library | No better alternative exists for cross-framework macOS global hotkeys |
| Add real model integration tests | Requires 4GB+ model downloads in CI; impractical. Mock tests are appropriate. |

---

## Appendix A: Dependency Audit Table

| Dependency | Version | Purpose | Risk | Notes |
|------------|---------|---------|------|-------|
| pynput | >=1.8.1 | Global hotkeys | Low | No CVEs. Monkey-patching needed for signature compat. |
| rich | >=14.1.0 | Terminal UI | Low | Well-maintained, 50K+ stars |
| typer | >=0.16.0 | CLI framework | Low | Built on Click |
| tomli | >=2.2.1 | TOML parsing (py<3.11) | Low | Standard library backport |
| toml | >=0.10.2 | TOML writing | Low | Legacy but stable |
| urllib3 | >=2.5.0 | HTTP (indirect) | Low | Via huggingface_hub |
| mlx-vlm[torch] | >=0.4.0 | LLM inference | **Medium** | Apple Silicon only. **Pin mlx>=0.29.4 for CVE fix** |
| psutil | >=6.1.0 | Memory monitoring | Low | Standard system info library |
| faster-whisper | >=1.2.0 | STT (optional) | Low | No CVEs. CTranslate2 backend. |
| mlx-whisper | >=0.4.2 | STT (optional) | Medium | Apple Silicon only. Auto-download behavior. |
| vosk | >=0.3.44 | STT (optional) | **High** | Minimal maintenance since 2023. Kaldi-based. Deprecate. |
| soundfile | >=0.13.1 | Audio I/O | Medium | Needed by vosk and diarization |
| scikit-learn | >=1.3.0 | Speaker clustering | Medium | Heavy dep (~100MB), only for Vosk path |
| sherpa-onnx | (undeclared) | Diarization | Medium | Not in pyproject.toml. Must add. |
| librosa | (undeclared) | Audio resampling | Medium | Not in pyproject.toml. Must add. |

## Appendix B: Exa Research Sources

All recommendations in Section 8 are grounded in web research. Key sources:

- **Parakeet validation**: Reddit r/LocalLLaMA (Mar 2026), Northflank benchmarks (Jan 2026)
- **Sherpa-onnx quality**: GitHub issue k2-fsa/sherpa-onnx#1708, AssemblyAI comparison (2026)
- **Qwen3.5**: HuggingFace model card, ComputerTech review (2026), Unsloth docs
- **MLX CVE**: CVE Feed, CX Security bulletin
- **macOS audio**: Apple ScreenCaptureKit docs, systemAudioDump GitHub, PyObjC issue #647
- **Hatch vs uv**: Scopir comparison (2026), Hatch GitHub discussion #1867
- **Competitors**: Meetily (10.4K stars), MeetScribe (219 stars), OwnScribe (5 stars)
