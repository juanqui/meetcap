---
paths:
  - "docs/specs/**"
---

# Spec-Driven Development

Use this workflow when the user explicitly asks to write a spec, or when the feature is complex enough to warrant formal specification.

## Lifecycle

```
Idea → Research → Specification → Review (3+ passes) → [Experimentation] → Implementation → Verification → Spec Update
```

## Phase 1: Research

1. Read the issue/requirement thoroughly
2. Explore the affected codebase — understand existing patterns and tests
3. Map the blast radius — list every file that would change
4. Research externally with Exa MCP (best practices, pitfalls) and Context7 MCP (library APIs)
5. Document findings — every claim must be backed by evidence

## Phase 2: Specification

Create `docs/specs/YYYY-MM-DD-<slug>.md`. Every spec must include:

1. **Summary** — what and why (2-3 sentences)
2. **Problem Statement** — current state, goals, non-goals
3. **Design Overview** — approach, key decisions, alternatives rejected with reasoning
4. **Detailed Design** — data model, API, workflows; precise enough to implement without guessing
5. **Edge Cases & Error Handling** — failure modes, recovery strategies
6. **File Changes** — table of every file to create/modify
7. **Testing Strategy** — unit tests, manual validation, acceptance criteria

One file per spec. Never split across multiple files.

## Phase 3: Review (Minimum 3 Passes)

Reviews must be sequential — apply edits from pass N before starting pass N+1.

1. **Pass 1 — Assumption Validation**: identify and verify every assumption
2. **Pass 2 — Completeness & Consistency**: find gaps, contradictions, missing edge cases
3. **Pass 3 — Clarity & Actionability**: eliminate vague language, ensure implementability

Every pass must produce concrete improvements. No rubber-stamping.

## Phase 4: Implementation

1. Break work into discrete tasks
2. Follow the file changes table exactly
3. Run `hatch run test` and `hatch run lint` after each significant change
4. Self-review all changes against the spec

## Phase 5: Verification & Maintenance

- Verify implementation matches spec acceptance criteria
- Update spec status: `Draft` → `In Progress` → `Implemented`
- Document deviations with reasoning
