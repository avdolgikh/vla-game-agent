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

## Step 6: Pipeline Generalization (Post MVP-3) -- NEXT

The agentic TDD pipeline is a standalone portfolio asset and potential open-source tool. After MVP-3, the priority shifts from the ML project to the pipeline itself.

**What to do:**
1. **Generalize pipeline code** -- Remove project-specific assumptions from `src/vla_agent/pipeline/core.py` and providers. The pipeline should work for any repo, not just vla-game-agent. Extract into a reusable package or standalone repo.
2. **Add Gemini provider** -- Spec exists (`specs/gemini-provider-spec.md`). Straightforward adapter following the Codex provider pattern.
3. **Actualize Claude provider** -- `providers/claude.py` may have drifted during Codex provider refinements (stages 6-8 retry loop, prompt-via-stdin, encoding fixes). Verify it still works end-to-end and align with Codex provider capabilities.
4. **Verify all providers work** -- Run a simple spec through each provider (Claude, Codex, Gemini) to confirm the provider-agnostic promise is real.

**Why this matters:**
- Most "AI coding" demos are one-shot generation. A multi-stage, self-correcting pipeline with quality gates is genuinely novel.
- Provider-agnostic design means it's not locked to one vendor -- but only if all providers actually work.
- As a standalone tool, this has community value beyond the VLA project.

**ML side is complete after MVP-3.** Further VLA improvements (more tasks, RL, attention-based memory) are diminishing portfolio returns. The experiment progression tells a complete story.
