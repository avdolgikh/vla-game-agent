# Spec: MVP-3 — Portfolio Polish

## Status

Draft

## Goal

Transform the repository from a working research codebase into a portfolio-ready open-source project. A visitor landing on the GitHub page should understand within 60 seconds: what this project is, what it achieved, and why it's interesting. The two portfolio assets are: (1) the VLA game agent itself (the ML story), and (2) the agentic TDD pipeline that built it (the engineering story).

---

## Problem Statement

The repo currently has strong technical artifacts (trained models, eval metrics, 143 tests, 16 specs) but minimal presentation:

- README is 28 lines with a random rollout command
- No architecture diagram
- No training curves or comparison plots
- Demo videos exist but only for MVP-1 (CNN-only, pre-VLA)
- The agentic TDD pipeline is undocumented for external readers
- No experiment comparison table visible without reading AGENTS.md

A portfolio visitor sees a bare repo with no story.

---

## Design

### Two portfolio stories

**Story 1 — The ML project:** A tiny instruction-conditioned vision-to-action agent for Crafter. Takes a game frame + text instruction, predicts a player action. Demonstrates the VLA paradigm in a game environment: language grounding enables task-specific behavior that vision-only models cannot achieve.

**Story 2 — The engineering process:** An agentic TDD pipeline where one command generates tests, reviews them, implements code, validates, and reviews again — fully automated with human approval only at spec stage. Provider-agnostic (Claude, Codex), auto-resumable, with frozen-test guardrails. This is a reusable open-source tool, not just a project detail.

### Deliverables

#### D-1: README rewrite

Replace the current README with a comprehensive project page. Structure:

1. **Header** — Project title, one-line description, badges (Python, PyTorch, License)
2. **Hero section** — 2-3 sentence pitch. What it is, what it demonstrates.
3. **Architecture diagram** — ASCII art showing: Frame + Instruction → [CNN] + [Text Encoder] → [Action Head] → Action. Show the VLA architecture visually.
4. **Key results table** — Comparison across all milestones (MVP-1 through MVP-2.3) showing val_acc and per-task success rates. The central result: MVP-1 (no text) vs MVP-2.3 (text + CNN + frames) = language grounding matters.
5. **Demo section** — Embedded GIFs or links to demo videos showing the agent playing Crafter with different instructions.
6. **Quick start** — Setup, training, evaluation commands (copy-paste ready).
7. **Project structure** — Brief layout of `src/`, `scripts/`, `specs/`, `artifacts/`.
8. **Agentic TDD Pipeline section** — Dedicated section explaining the pipeline as a standalone engineering contribution. State machine diagram (ASCII), provider architecture, how it works, link to `specs/agentic-pipeline.md`.
9. **Experiment log** — Condensed version of the experiment progression: what was tried, what worked, what didn't, and why.
10. **Honest limitations** — Single-frame reactive model, 3-task vocabulary, scripted data, no RL fine-tuning.
11. **License**

**Length target:** 200-400 lines. Detailed enough to be informative, concise enough to read in 5 minutes.

#### D-2: Demo videos for MVP-2.3

The current `demo_policy.py` only supports `CrafterCNN` (MVP-1). Update it to support VLA models (`--policy-type vla`, `--policy-type vla-cnn`) with instruction display and frame stacking. Then record demo videos for MVP-2.3:

- 3 episodes per instruction (9 total): collect_wood, place_table, collect_stone
- Include the instruction text in the video filename
- Use the best MVP-2.3 model (`artifacts/models/mvp2.3/best_model.pt`)
- Output to `artifacts/demo/mvp2.3/`

**Script changes to `scripts/demo_policy.py`:**
1. Add `--policy-type` argument (choices: `cnn`, `vla`, `vla-cnn`; default: `cnn`)
2. Add `--num-frames` argument (default: 1)
3. Add `--instructions` argument (optional, comma-separated; defaults to all 3 for VLA types)
4. Load model via the same `_load_policy` pattern as `evaluate_policy.py`
5. Use frame buffer for stacked-frame models (same as evaluation)
6. For VLA types, record separate episodes per instruction
7. Include instruction in video filename (e.g., `collect_wood_seed2000.mp4`)
8. Backward compatible: `--policy-type cnn` behaves identically to current script

#### D-3: Training curves plot

Generate a comparison plot from existing `train_log.json` files:

- X-axis: epoch, Y-axis: val_acc
- One line per milestone: MVP-1, MVP-2, MVP-2.1, MVP-2.2, MVP-2.3
- Clear legend with milestone descriptions
- Save as `artifacts/figures/training_curves.png`
- Also generate `artifacts/figures/task_success_rates.png`: grouped bar chart of per-task success rates across milestones

**Implementation:** A standalone script `scripts/plot_results.py` that reads from `artifacts/models/*/train_log.json` and `artifacts/eval/*/eval_results.json`. Uses matplotlib. No new dependencies (matplotlib is already a transitive dep via crafter).

#### D-4: Agentic TDD pipeline documentation

Create `docs/agentic-pipeline.md` — a standalone document explaining the pipeline for external readers:

1. **Motivation** — Why automate the TDD loop with AI agents?
2. **How it works** — The 8-stage state machine (spec → tests → test review → implement → validate → code review → train → evaluate → verify)
3. **State machine diagram** — ASCII art with stage names and transitions
4. **Provider architecture** — How the same pipeline runs on different AI backends (Claude, Codex, future Gemini)
5. **Guardrails** — Frozen-test protection, reviewer immutability, file snapshot hashing
6. **Auto-resume** — How interrupted runs pick up where they left off
7. **Artifact pipeline** — Stages 6-8: automated training, evaluation, acceptance checking
8. **Real example** — Walk through an actual pipeline run (MVP-2.3) showing the log output at each stage
9. **How to use** — Commands, spec format, how to add a new provider
10. **Design decisions** — Why external orchestrator + dumb agents, why role-based tiers, why JSON review contracts

**Length target:** 150-300 lines. Should stand alone as a blog-post-quality document.

**Important:** This is documentation, not a spec for code changes. It describes what already exists.

---

## Files to Create

| File | Description |
|------|-------------|
| `README.md` | Complete rewrite (overwrite existing) |
| `docs/agentic-pipeline.md` | Pipeline documentation for external readers |
| `artifacts/figures/training_curves.png` | val_acc comparison plot |
| `artifacts/figures/task_success_rates.png` | Per-task success rate bar chart |
| `artifacts/demo/mvp2.3/*.mp4` | Demo videos (9 total, 3 per instruction) |

## Files to Modify

| File | Change |
|------|--------|
| `scripts/demo_policy.py` | Add VLA/VLA-CNN support, instruction handling, frame stacking |
| `scripts/plot_results.py` | New script for generating comparison plots |

## What NOT to Change

- No changes to `src/vla_agent/` (models, data, envs, pipeline)
- No changes to training or evaluation scripts (except demo_policy.py)
- No changes to test files
- No changes to specs (they are historical records)

---

## Acceptance Criteria

### AC-1: README completeness

`README.md` contains all sections listed in D-1. Architecture diagram is present as ASCII art. Key results table shows all milestones with val_acc and per-task success rates.

### AC-2: README clarity

A reader unfamiliar with the project can understand from the README alone: (a) what VLA means in this context, (b) what the agent does, (c) how results progressed from MVP-1 to MVP-2.3, (d) that the pipeline is a separate engineering contribution.

### AC-3: Demo videos generated

`artifacts/demo/mvp2.3/` contains at least 9 MP4 files (3 per instruction). Each filename includes the instruction name and seed.

### AC-4: demo_policy.py supports VLA

`scripts/demo_policy.py --policy-type vla-cnn --model artifacts/models/mvp2.3/best_model.pt --num-frames 4` generates videos without error. Backward compatible: `--policy-type cnn` still works.

### AC-5: Training curves plot

`artifacts/figures/training_curves.png` exists and shows val_acc curves for MVP-1 through MVP-2.3.

### AC-6: Task success rates plot

`artifacts/figures/task_success_rates.png` exists and shows grouped bars for collect_wood, place_table, collect_stone across milestones.

### AC-7: Pipeline documentation

`docs/agentic-pipeline.md` exists and covers all sections listed in D-4. The document is self-contained (doesn't require reading AGENTS.md to understand the pipeline).

### AC-8: Plot script is runnable

`uv run python scripts/plot_results.py` produces both figures without error from existing artifact data.

### AC-9: No production code changes

No files under `src/vla_agent/` are modified. No test files are modified.

---

## Notes

### On the pipeline as portfolio asset

The agentic TDD pipeline is a genuine engineering contribution worth highlighting. Key selling points:
- **One-command automation:** `uv run python scripts/run_pipeline.py <task-id> --provider codex` takes a spec and delivers tested, reviewed, trained code
- **Provider-agnostic:** Same pipeline works across AI backends (Claude, Codex; Gemini planned)
- **Self-healing:** Auto-resume from interruption, retry loops on fixable failures
- **Guardrailed:** Frozen tests, reviewer immutability, file-hash verification
- **Battle-tested:** Successfully delivered MVP-2 through MVP-2.3 — real models trained, real metrics verified

The README should present this as a first-class project contribution, not a footnote.

### On honest positioning

Per the initial spec's guidance: this is an "instruction-conditioned visual game agent" / "tiny VLA-style game bot." Not a foundation model, not a general game-playing agent, not a robotics system. The README should be honest about scope while being confident about what was achieved.

### On the Gemini provider

The Gemini provider spec (`specs/gemini-provider-spec.md`) exists but is not part of MVP-3 scope. It can be mentioned in the README as a planned extension. The Claude provider (`providers/claude.py`) may need actualization after recent Codex provider refinements — this is also out of MVP-3 scope but worth noting for future work.

### On matplotlib availability

Verify matplotlib is available before implementing plots. If not present, add it as a dev dependency via `uv add --dev matplotlib`.
