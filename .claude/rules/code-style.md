---
paths:
  - "**/*.py"
---

# Python Code Style

## Formatting

- **Black** formatter with 100-character line length
- **Ruff** for linting (rules: E, W, F, I, B, C4, UP)
- Target Python 3.10–3.12

## Conventions

- Type hints on all function signatures (enforced by mypy with `disallow_untyped_defs`)
- Lowercase inline comments
- Rich console for all user-facing output (`from rich.console import Console`)
- snake_case for variables, functions, modules
- PascalCase for classes
- SCREAMING_SNAKE_CASE for constants

## Import Order

1. Standard library
2. Third-party packages
3. Local imports (`from meetcap.core import ...`)

Ruff's `I` rule enforces import sorting automatically.

## Error Handling

- Graceful fallbacks for all services (STT, LLM)
- User-friendly error messages with rich formatting
- Automatic recovery attempts where possible
- Detailed logging to `~/.meetcap/debug.log`
