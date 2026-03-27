# Roadmap (for Future Sessions)

This section is the authoritative roadmap for any coding agent working on this project. Follow in order.

## Prerequisites

All specs are written and ready:
- `specs/pipeline-training-stages-spec.md` -- Pipeline extension (stages 6-8)
- `specs/vla-quality-improvements-spec.md` -- 10 improvement directions with impact/effort analysis

## Step 1: Extend Pipeline (Pipeline-ext) -- DONE

Completed 2026-03-25. Stages 6-8 implemented in `core.py`, 29 tests in `test_pipeline_stages.py`. Specs now support `## Artifact Pipeline` section with YAML-formatted Training/Evaluation/Acceptance blocks.

## Step 2: Apply 224x224 Resize (MVP-2.1) -- DONE

Completed as part of MVP-2.x series.

## Step 3: Frame Stacking + Wider Head (MVP-2.2) -- DONE

Completed as part of MVP-2.x series.

## Step 4: Domain Adaptation (MVP-2.3) -- DONE

Completed 2026-03-26. Trainable CNN replaces frozen ConvNeXt. val_acc=76.8% (best ever), place_table 76% (3.5x over MVP-2.2), collect_stone 6% (first nonzero). Pipeline ran clean: 2 test revision cycles, implementation on first try, training/eval/acceptance all passed. 143 tests passing (98 skipped). Artifacts in `artifacts/models/mvp2.3/` and `artifacts/eval/mvp2.3/`.

## Step 5: Portfolio Polish (MVP-3) -- DONE

**Spec:** `specs/mvp-3-spec.md` (9 acceptance criteria)

**Deliverables:**
1. README.md -- architecture diagram, results table, pipeline section, quick start, honest limitations
2. `scripts/demo_policy.py` updated for VLA models (--policy-type vla-cnn, --num-frames, --instructions)
3. 9 MVP-2.3 demo videos (3 per instruction)
4. `scripts/plot_results.py` -- training curves and task success rate comparison plots
5. `docs/agentic-pipeline.md` -- standalone pipeline documentation for external readers

**Two portfolio stories:** (1) The VLA agent (ML), (2) The agentic TDD pipeline (engineering/open-source contribution)

## Step 6: Pipeline Generalization (Post MVP-3) -- DONE

Completed 2026-03-26.

**What was done:**
1. **`PipelineConfig` dataclass** -- All hardcoded paths/commands in `core.py` replaced with config fields (`specs_dir`, `tests_dir`, `source_dirs`, `state_dir`, `test_command`, `context_file`, `hash_targets`, `prompts_dir`). Defaults match vla-game-agent layout.
2. **Package-relative prompts** -- `PromptBuilder` defaults to `Path(__file__).parent / "prompts"` (works from any repo without config).
3. **TOML config loading** -- `scripts/run_pipeline.py --config pipeline.toml` loads `PipelineConfig` from file.
4. **Repo-root override** -- `scripts/run_pipeline.py --repo-root <path>` targets any repository.
5. **Gemini provider** -- `providers/gemini.py` added, uses local Gemini CLI with stdin-based prompting and JSON output.
6. **Provider `state_dir` contract** -- `Provider.run_role()` now receives `state_dir` from the pipeline, eliminating hardcoded `.pipeline-state` in adapters.
7. **Prompt templates genericized** -- `.md` files reference "configured tests directory" / "configured test command" instead of hardcoded paths.

**E2E validation (see `specs/pipeline-e2e-testing-plan.md`):**

| Repo | Codex | Claude | Gemini |
|------|-------|--------|--------|
| vla-game-agent (default config) | DONE | DONE | DONE |
| Fresh repo (custom config) | DONE | -- | DONE |

**Remaining:**
- **Extract as standalone package** -- The pipeline still lives in `src/vla_agent/pipeline/`. Could be extracted to its own repo/package.
- **Claude provider cleanup** -- Claude returns full JSON envelope in reviewer output (works via fallback normalization, but adapter could extract `structured_output` directly).

**ML side is complete after MVP-3.** Further VLA improvements (more tasks, RL, attention-based memory) are diminishing portfolio returns. The experiment progression tells a complete story.
