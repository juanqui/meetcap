---
paths:
  - "docs/**"
---

# Documentation Conventions

## Directory Structure

| Directory | Purpose | Naming |
|-----------|---------|--------|
| `docs/specs/` | Technical specifications | `YYYY-MM-DD-slug.md` |
| `docs/research/` | Research findings | `YYYY-MM-DD-slug.md` |
| `docs/designs/` | Architecture docs (living) | `slug.md` |

## Document Standards

- Start with a title (`# Title`)
- Include metadata: Version, Date, Status (at minimum)
- Use tables for comparisons and structured data (not prose lists)
- Use fenced code blocks with language tags for code, schemas, and CLI commands
- Use Mermaid diagrams for flows, sequences, state machines, and architecture — never ASCII art
- Link related documents when cross-referencing

## Status Values

`Idea` → `Draft` → `In Review` → `In Progress` → `Implemented` → `Superseded`
