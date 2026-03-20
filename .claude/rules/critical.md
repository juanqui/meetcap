# Critical Rules

Hard rules with no exceptions. These apply to every task regardless of scope.

## Secret Protection

Never commit secrets, API keys, tokens, credentials, or `.env` files to source control. All secrets belong in environment variables or secret managers. Use placeholder values (`YOUR_API_KEY_HERE`) in examples. If you discover a committed secret, immediately flag it.

## Git Discipline

Never commit or push unless explicitly asked. Do not infer that you should commit after implementing a feature. Wait for explicit instruction ("commit this", "push this", "cut a PR").

Before pushing: always run `uv run pytest` and `uv run ruff check . && uv run ruff format --check .` to verify the build passes. Never push code that fails tests or linting.

## Assumption Grounding

Never assume how something works — always verify.

- **Third-party APIs/libraries**: use Context7 MCP to verify method signatures, parameters, return values
- **Best practices**: use Exa MCP to research current best practices, deprecations, security advisories
- **Existing codebase**: read the actual code; don't assume based on file names or patterns
- **Data shapes**: check actual schemas, API responses, and data formats

If you cannot verify something, explicitly flag it as an **unverified assumption**.

## Verification Before Completion

Work is not done until verified:

1. Lint: `uv run ruff check . && uv run ruff format --check .` passes
2. Tests: `uv run pytest` passes
3. Manual check when possible: `uv run meetcap verify`

### Debugging Discipline

When something fails:

1. Formulate a hypothesis about the root cause
2. Validate the hypothesis before acting — check logs, inspect state, run experiments
3. Confirm you found the actual problem, not a red herring
4. Only then fix it
5. Never make sweeping changes based on an unvalidated theory

## Expensive Command Efficiency

Never run builds, tests, or linters multiple times when once suffices.

```bash
# correct — capture output, inspect separately
uv run pytest 2>&1 | tee /tmp/test-output.log
grep -iE "(error|warning|failed)" /tmp/test-output.log

# wrong — runs the entire test suite twice
uv run pytest 2>&1 | tail -40
uv run pytest 2>&1 | grep -iE "error"
```

## Version Bumping

Never edit `__version__` in `meetcap/__init__.py` manually. Always use `uv run bump-my-version bump {patch|minor|major}`. Manual edits desync `pyproject.toml`'s `[tool.bumpversion] current_version`, the source file, and git tags — breaking all future version bumps.

## Cost-Sensitive Configuration

Never modify model configuration, LLM model selections, or any setting that affects resource consumption without explicit user permission. This includes changes to `config.toml`, model paths, or GGUF files.

## Iterative Review

All significant work products (specs, implementations, configurations) must be reviewed iteratively. Minimum 3 sequential review passes for specifications. Each pass must produce concrete improvements — no rubber-stamping. Apply edits from pass N before starting pass N+1.

## Spec-Driven vs Plan-Driven

Not every feature requires a full specification. Use judgment:

- **Full spec**: when the user explicitly asks, or the feature is architecturally significant, touches many files, or introduces new integrations
- **Session plan**: for narrow, well-understood changes (adding a field, fixing a bug, updating UI) — think through the approach, note assumptions, then implement
