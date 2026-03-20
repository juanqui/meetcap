# meetcap

Offline meeting recorder & summarizer for macOS. Captures system audio + microphone, transcribes locally with Parakeet TDT (default) or Whisper, identifies speakers via sherpa-onnx diarization, summarizes with a local LLM. 100% offline — no network calls.

## Essential Commands

All Python commands **must** use `uv run` prefix.

```bash
uv run pytest                 # run tests with coverage
uv run ruff format .          # format code
uv run ruff check --fix .     # lint and fix
uv run ruff check .           # lint code
uv run meetcap record         # start recording
uv run meetcap summarize      # process existing audio
uv run meetcap reprocess      # reprocess with different models
uv run meetcap devices        # list audio devices
uv run meetcap verify         # verify system setup
uv run meetcap setup          # interactive setup wizard
```

## Commit Style

Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`

Format: `type: brief description` (lowercase, no period). Do not mention Claude or AI assistance.

## Rules

Project rules are modularized in `.claude/rules/`. Key rules:

- `critical.md` — hard rules that always apply
- `development.md` — dev environment and workflow
- `code-style.md` — Python formatting and conventions
- `testing.md` — test patterns and requirements
- `architecture.md` — system design and key decisions
- `spec-workflow.md` — spec-driven development process
- `docs.md` — documentation conventions
