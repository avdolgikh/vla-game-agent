# Agentic TDD Pipeline

Automates: **spec (human) → tests → test review → implement → validate → code review → done**.

One command. Walk away. Come back to green tests and reviewed code.
Interrupted runs auto-resume from the last completed stage.

## Architecture

```
Bash orchestrator (run_pipeline.sh)
  ├── claude -p --agent test-writer    # sonnet — writes tests from spec
  ├── claude -p --agent reviewer       # opus  — reviews tests, returns JSON
  ├── claude -p --agent implementer    # sonnet — implements against frozen tests
  ├── claude -p --agent implementer    # sonnet — validates end-to-end
  └── claude -p --agent reviewer       # opus  — reviews code, returns JSON
```

- **External script** owns the state machine, loop caps, stage transitions, and resume logic
- **Subagents** own role-specific behavior, tools, and model choice
- **Hooks** enforce guardrails (run tests after edits, block protected files)
- **AGENTS.md** encodes repo-wide rules all agents inherit
- Spec approval stays **human-in-the-loop** — everything after is automated

## State Machine

```
SPEC_APPROVED (human)
  → TESTS_GENERATED
  → TESTS_REVIEWED ←─╮
  → TESTS_REVISED ────╯  (max 4 iterations)
  → TESTS_FROZEN
  → CODE_IMPLEMENTED
  → CODE_VALIDATED
  → CODE_REVIEWED  ←─╮
  → CODE_REVISED  ────╯  (max 4 iterations)
  → DONE
```

## Auto-Resume

State is persisted to `.pipeline-state/<task-id>.json` after each completed stage:

```json
{ "task": "<task-id>", "stage": "TESTS_FROZEN", "iteration": 0 }
```

On startup, the orchestrator reads this file and skips all stages up to and including the saved stage. For review loops, the saved `iteration` determines where to re-enter the loop.

If no state file exists, the pipeline starts from scratch. On `DONE`, the state file is retained as a record.

Stages have a linear ordering used for resume decisions:

| Order | Stage | After |
|-------|-------|-------|
| 0 | SPEC_APPROVED | (entry) |
| 1 | TESTS_GENERATED | Stage 1 |
| 2 | TESTS_FROZEN | Stage 2 approves |
| 3 | CODE_IMPLEMENTED | Stage 3 |
| 4 | CODE_VALIDATED | Stage 4 |
| 5 | DONE | Stage 5 approves |

## Logging

All stage output is tee'd to `.pipeline-state/<task-id>.log`. This provides a full transcript for debugging failed or interrupted runs.

## Subagents (`.claude/agents/`)

### test-writer.md

```md
---
name: test-writer
description: Writes tests from an approved spec. Never touches production code.
tools: Read, Write, Edit, Grep, Glob, Bash
model: sonnet
---

Read the approved spec. Generate unit tests covering all acceptance criteria.
Run `uv run pytest` to confirm tests fail (red) for unimplemented code.
Do not write production code. Output a coverage summary.
```

### implementer.md

```md
---
name: implementer
description: Implements production code against frozen tests and approved spec.
tools: Read, Write, Edit, Grep, Glob, Bash
model: sonnet
---

Read the spec and frozen tests. Implement the minimal production code to make
all tests pass. Run `uv run pytest` after changes. Do not modify frozen tests.
Prefer the smallest correct diff. If blocked, explain the blocker precisely.
```

### reviewer.md

```md
---
name: reviewer
description: Reviews tests or code for correctness, coverage, and spec alignment.
tools: Read, Grep, Glob, Bash
model: opus
---

Review the provided artifacts against the approved spec.
Do not edit any files. Produce findings grouped by severity:
- blocking: must fix before proceeding
- important: should fix
- optional: suggestions

State whether the artifact is acceptable or needs revision.
```

## Hooks (`.claude/settings.json`)

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "uv run ruff format --quiet ."
          }
        ]
      }
    ]
  }
}
```

## Review Stage Contract

Review stages use `--output-format json --json-schema` for machine-readable decisions:

```json
{
  "type": "object",
  "properties": {
    "decision": { "type": "string", "enum": ["approve", "revise"] },
    "summary": { "type": "string" },
    "blocking": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["decision", "summary"]
}
```

Reviewer `blocking` items are extracted and passed verbatim to revision agents, so they know exactly what to fix.

## Guardrails

- **Revision caps**: 4 per loop (tests and code). Prevents drift.
- **Frozen tests**: after implementation, validation, and code revisions, `git diff -- tests/` is checked. Any modifications to test files abort the pipeline (exit 4).
- **Pytest gate**: after each code revision, `uv run pytest` must pass. Failures abort the pipeline (exit 5).
- **Reviewer is read-only**: no Write/Edit tools. Findings only.
- **Reviewer feedback forwarded**: blocking items from reviews are passed into revision prompts.
- **Spec is human-approved**: pipeline assumes spec already exists and is signed off.
- **Logging**: full transcript to `.pipeline-state/<task-id>.log` for post-mortem analysis.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Spec not found |
| 2 | Test revision cap reached |
| 3 | Code revision cap reached |
| 4 | Frozen tests modified |
| 5 | Tests broke after code revision |