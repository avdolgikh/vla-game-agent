# Agentic Spec-driven Development Pipeline

An automated spec-driven development pipeline that turns a human-approved spec
into tested, reviewed, validated code -- with zero human intervention after
the spec is approved.

## How It Works

You write a spec. Three AI agents do the rest:

1. **Test-writer** reads the spec, writes tests, confirms they fail
2. **Reviewer** checks the tests against the spec (up to 4 revision rounds)
3. **Implementer** writes code to make the frozen tests pass
4. **Reviewer** checks the code (up to 4 revision rounds)
5. *(Optional)* Pipeline runs commands from the spec to produce artifacts,
   validates outputs, and checks acceptance criteria

The agents cannot see each other's prompts. The reviewer cannot modify files.
Tests are frozen after review -- the implementer cannot change them. All of
this is enforced by file hashes, not trust.

## Quick Start

```bash
# Run the pipeline on a spec
uv run python scripts/run_pipeline.py <task-id> --provider codex

# Available providers
--provider codex    # OpenAI Codex CLI
--provider claude   # Claude Code CLI
--provider gemini   # Gemini CLI
```

The spec must exist at `specs/<task-id>-spec.md` before you run.

## Specs

A spec is a markdown file with:

- **Goal** -- what the feature does
- **Requirements** -- what to build
- **Acceptance Criteria** -- what the tests must verify

If the spec includes an `## Artifact Pipeline` section (YAML), the pipeline
will also run commands to produce artifacts, check output files, validate
results, and verify acceptance criteria automatically.

## State & Resume

Pipeline state is saved to `.pipeline-state/<task-id>.json` after every stage.
If the run is interrupted, re-run the same command -- it picks up where it
left off. To start fresh, delete the state file.

## Configuration

Override project layout via `--config pipeline.toml`:

```toml
specs_dir = "specifications"
tests_dir = "test_suite"
source_dirs = ["lib", "scripts"]
test_command = ["uv", "run", "python", "-m", "pytest"]
context_file = "AGENTS.md"
```

Target any repo with `--repo-root <path>`.

## Providers

Each provider wraps a CLI tool. Model selection uses two tiers:

| Tier | Used by | Purpose |
|------|---------|---------|
| Economy | test-writer, implementer | Bulk code generation |
| Premium | reviewer | Careful review |

Override models via environment variables:

```bash
# Codex
CODEX_MODEL_TEST_WRITER, CODEX_MODEL_IMPLEMENTER, CODEX_MODEL_REVIEWER

# Claude
CLAUDE_MODEL_ECONOMY, CLAUDE_MODEL_PREMIUM

# Gemini
GEMINI_MODEL_ECONOMY, GEMINI_MODEL_PREMIUM
```

## Guardrails

- **Frozen tests** -- SHA-256 hash locks tests after review; any modification
  halts the pipeline
- **Reviewer immutability** -- repository hash is checked before and after
  every review stage
- **Retry loop** -- artifact failures trigger an implementer fix, re-verify
  tests, then restart artifact production
- **Scope enforcement** -- agents receive only the approved spec; they cannot
  expand scope beyond it

## File Layout

```
src/vla_agent/pipeline/
  core.py              # State machine, orchestrator, prompt builder
  providers/
    base.py            # Provider protocol (run_role interface)
    codex.py           # OpenAI Codex CLI adapter
    claude.py          # Claude Code CLI adapter
    gemini.py          # Gemini CLI adapter
  prompts/
    test_writer.md     # Test-writer role prompt
    implementer.md     # Implementer role prompt
    reviewer.md        # Reviewer role prompt
```
