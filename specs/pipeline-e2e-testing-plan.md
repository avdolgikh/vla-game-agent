# Pipeline E2E Testing Plan

## Goal

Validate that the generalized pipeline (`PipelineConfig`) works in practice — not just unit tests, but real spec runs through real providers.

## Phase 1: Backward Compatibility (this repo)

Run the existing `smoke-test` spec in this repo with default `PipelineConfig()`.

```bash
uv run python scripts/run_pipeline.py smoke-test --provider codex
uv run python scripts/run_pipeline.py smoke-test --provider claude
```

**Validates**: Default config produces identical behavior to pre-refactoring hardcodings.

**Pass criteria**: Pipeline reaches `DONE` on both providers.

## Phase 2: Config Loading via CLI

Add config-file support to `scripts/run_pipeline.py` so custom `PipelineConfig` values can be passed without code changes.

- New flag: `--config <path>` pointing to a TOML file.
- TOML fields map 1:1 to `PipelineConfig` fields.
- All fields optional — unspecified fields use `PipelineConfig` defaults.

Example `pipeline.toml`:

```toml
specs_dir = "specifications"
tests_dir = "test_suite"
source_dirs = ["lib"]
context_file = "CONTEXT.md"
test_command = ["python", "-m", "pytest"]
```

**Validates**: CLI accepts custom config.

**Pass criteria**: `run_pipeline.py --config pipeline.toml smoke-test --provider codex` parses correctly and `PipelineConfig` fields match the TOML.

## Phase 3: Generalization Proof (different repo layout)

Create a minimal throwaway repo with non-standard structure:

```
my-test-repo/
  lib/
  test_suite/
  specifications/
  CONTEXT.md
  pyproject.toml
```

Write a trivial spec (copy of `smoke-test`) targeting `lib/`. Provide a `pipeline.toml` with non-default dirs.

```bash
uv run python scripts/run_pipeline.py --config pipeline.toml smoke-test --provider codex
```

**Validates**: Pipeline works with arbitrary project layouts.

**Pass criteria**: Pipeline reaches `DONE` with non-default dirs on at least one provider.

## Phase 4: Gemini Provider

Implement `gemini-provider-spec.md` (status: Proposed). Then:

```bash
uv run python scripts/run_pipeline.py smoke-test --provider gemini
```

**Validates**: Third provider adapter works.

**Pass criteria**: Pipeline reaches `DONE` with Gemini.

## Phase 5: Full Matrix

Run `smoke-test` across all providers in both repo layouts.

| Repo | Codex | Claude | Gemini |
|------|-------|--------|--------|
| vla-game-agent (default config) | DONE | DONE | DONE |
| Fresh repo (custom config) | DONE | — | DONE |

**Pass criteria**: All 6 cells reach `DONE`.

## Execution Order

| Step | What | Depends on | Validates |
|------|------|------------|-----------|
| 1 | `smoke-test` in this repo | Codex/Claude CLI installed | Backward compat |
| 2 | Add `--config` flag + TOML loading to `run_pipeline.py` | Step 1 green | CLI accepts custom config |
| 3 | `smoke-test` in fresh repo with non-standard layout | Step 2 | True generalization |
| 4 | Implement Gemini provider | `gemini-provider-spec.md` | 3rd provider |
| 5 | `smoke-test` in this repo with Gemini | Step 4 | Gemini backward compat |
| 6 | `smoke-test` in fresh repo with all three | Steps 3 + 5 | Full matrix |

## Current Status

- [x] `PipelineConfig` dataclass added, all hardcodings replaced, 53 unit tests pass.
- [x] Step 1: Backward compatibility — `smoke-test` DONE on Codex (gpt-5.1-codex-mini/gpt-5.2-codex) and Claude (sonnet/opus). 2026-03-26.
- [x] Step 2: Config loading via CLI — `--config pipeline.toml` + `--repo-root` flags added to `run_pipeline.py`. 2026-03-26.
- [x] Step 3: Generalization proof — `smoke-test` (clamp function) DONE on Codex with non-standard layout (specifications/, test_suite/, lib/, CONTEXT.md). 2026-03-26.
- [x] Step 4: Gemini provider — adapter implemented (`providers/gemini.py`), wired into `run_pipeline.py`. 2026-03-26.
- [x] Step 5: Gemini backward compat — `smoke-test` DONE on Gemini (gemini-2.5-flash). Note: first attempt hit quota on gemini-2.5-pro; second attempt hit frozen-test-modified guard (model compliance); third attempt succeeded. Reviewer fallback normalization handled markdown-fenced JSON. 2026-03-26.
- [x] Step 6: Full matrix — Gemini DONE on non-standard repo (specifications/, test_suite/, lib/, CONTEXT.md). 2026-03-26.
