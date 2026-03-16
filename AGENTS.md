# AGENTS.md - Repo-Wide Rules & Context

All agents (test-writer, implementer, reviewer) inherit these rules automatically.

---

## Project

**vla-game-agent** — An instruction-conditioned vision-to-action agent for Crafter.
VLA-style game bot: takes a game frame + a text instruction → predicts a player action.

### Why VLA, Not RL

Reinforcement learning answers "maximize this reward signal." It needs per-task reward functions, millions of environment interactions, and produces policies that can't be redirected with natural language. VLA (Vision-Language-Action) answers "given what you see and what I told you, what should you do?" One model handles many tasks, steered by language instructions. This is the paradigm shift from "one policy per task" to "one model that follows instructions."

The project demonstrates this shift in a clean, minimal setting. The core result is the gap between MVP-1 (vision-only, no instructions → collapses to one behavior) and MVP-2 (vision + language → task-specific behavior). That gap quantifies the value of language grounding for goal-directed behavior.

### Why Pretrained Components

Training vision and language representations from scratch on ~500 episodes (~33K samples) is infeasible. Pretrained encoders solve this: a frozen vision encoder provides visual understanding, a frozen text encoder provides language understanding, and only a small trainable action head learns the (visual_features + text_features) → action mapping. The hard problems (seeing, reading) are already solved by pretraining on large datasets. This is the standard VLA paradigm (RT-2, Octo, OpenVLA in robotics), applied here to a game environment.

### VLA Architecture (MVP-2)

```
Frame (64×64 RGB) → [Frozen Vision Encoder] → vision_features (e.g., 512-d)
Instruction text  → [Frozen Text Encoder]   → text_features   (e.g., 512-d)

concat(vision_features, text_features) → [Trainable MLP Action Head] → 8 action logits
```

Encoders are frozen — gradients only flow through the action head. This fits in 12 GB VRAM (RTX 4070) and is trainable on 500 episodes.

### Practical Purpose

Once trained, the VLA is controlled entirely through language. Same game frame, different instruction → different behavior. This applies beyond games: robotics ("pick up the red cup"), autonomous systems (LLM reasons in text, VLA executes physically), and any domain where you need one agent that follows diverse instructions without retraining per task.

## Architecture Decisions Log

Document every significant decision here as it happens.

- **2026-03-12**: Crafter is the primary environment. Reason: cleaner instruction-following story, lower overhead, faster iteration.
- **2026-03-12**: Agentic TDD is the only development workflow. Flow: spec -> tests -> test review -> implement -> validate -> code review -> done.
- **2026-03-12**: UV is the sole package/environment manager.
- **2026-03-12**: Pipeline state auto-resumes from `.pipeline-state/<task-id>.json`; stage output is logged to `.pipeline-state/<task-id>.log`.
- **2026-03-12**: Pipeline architecture stays provider-agnostic in the shared core with thin provider adapters. Runtime selection is via `--provider`.
- **2026-03-12**: Pipeline guardrails must be provider-independent. Frozen-test protection and reviewer immutability are enforced by file snapshots/hashes, not only provider-native controls.
- **2026-03-12**: Role behavior is defined by capability tier, not provider model names. Test-writing, implementation, and validation use an economy tier; review uses a premium tier. Concrete model names live only in provider-specific config/adapter code.
- **2026-03-12**: Review prompts must embed the approved spec and an explicit raw-JSON response contract. Reviewer stages may perform one automatic repair retry on malformed output.
- **2026-03-13**: Canonical reviewer JSON schema requires `additionalProperties: false` and requires every declared property, including `blocking`.
- **2026-03-13**: Initial reviewer prompts must include a compact artifact snapshot, and reviewer outputs that claim missing inputs despite embedded artifacts are invalid.
- **2026-03-13**: On this Windows setup, do not rely on Python `tempfile` or freshly created runtime temp directories for provider scratch space or pytest basetemp unless proven stable.
- **2026-03-13**: Codex scratch artifacts must live under `.pipeline-state/`, not in Python temp directories.
- **2026-03-13**: Pipeline stage advancement must verify required workspace effects, not only subprocess exit status.
- **2026-03-13**: Shared prompts must state that runs are non-interactive and that embedded context is authoritative.
- **2026-03-13**: On this Windows setup, Codex provider stages use `danger-full-access`; the orchestrator's frozen-test and reviewer-immutability guards remain the primary safety controls.
- **2026-03-13**: Reviewer prompts use the approved spec as scope and ignore unrelated dirty-worktree changes outside that scope.
- **2026-03-13**: Reviewer artifact snapshots prioritize spec-referenced files before unrelated repository files.
- **2026-03-13**: Next provider expansion target is the local Gemini CLI runtime.
- **2026-03-13**: Gemini provider scope is centered on reliable non-interactive local CLI execution and reviewer-output normalization, not on speculative provider-specific flags. Windows executable resolution and subprocess diagnostics are first-class requirements; approval-mode details stay adapter-level unless proven necessary by the CLI.
- **2026-03-13**: On Windows, Codex provider prompts must be sent via stdin (codex exec -) rather than as a positional CLI argument to avoid command-line length failures on large embedded specs/prompts.
- **2026-03-13**: Pipeline logging on this Windows console must tolerate non-ASCII provider output by writing to stdout with encoding replacement instead of raw print(), while preserving UTF-8 transcripts on disk.
- **2026-03-13**: Repo rule added: preserve existing document encoding and visible typography in specs/docs unless the user explicitly requests formatting or character-set changes.
- **2026-03-15**: Trajectory collection is split into two phases. Phase 1 (MVP-0b): 10 episodes per policy to validate the pipeline end-to-end. Phase 2 (MVP-1): 100–500 episodes per policy to produce the actual training dataset. MVP-0b's deliverable is working infrastructure, not training-scale data.
- **2026-03-15**: Milestone progression follows deliberate capability layering. MVP-1 is a vision-only baseline (frame → CNN → action, no text input) trained on all 3 policies mixed — it intentionally cannot distinguish tasks. MVP-2 adds instruction conditioning (frame + text → action), making it a true VLA. The gap in success rates between MVP-1 and MVP-2 is the project's core result: language grounding matters for goal-directed behavior.
- **2026-03-15**: MLflow experiment tracking is in-scope for MVP-1. Local file-based backend (`mlruns/`), no server. Integrated into the training script with a `--no-mlflow` escape hatch for tests. Rationale: MVP-1 is the first training loop — tracking from the start avoids retrofitting and enables clean MVP-1 vs MVP-2 comparison.
- **2026-03-15**: MVP-1 spec approved (`specs/mvp-1-spec.md`). 11 acceptance criteria. Next pipeline step: write tests.
- **2026-03-16**: MVP-1 agentic TDD pipeline launched via Codex provider (`uv run python scripts/run_pipeline.py mvp-1 --provider codex`). Pipeline auto-executes: tests → test review → implement → validate → code review → done. State in `.pipeline-state/mvp-1.json`, transcript in `.pipeline-state/mvp-1.log`.
- **2026-03-16**: MVP-1 pipeline completed. Post-pipeline review found 5 issues; all fixed. Added Rules #10–#12 (no monkey-patching, test tier separation, config consistency). Added `tests/conftest.py` to auto-skip integration/slow tests. Removed 60-line MLflow monkey-patch from `__init__.py`.
- **2026-03-16**: MVP-1 training complete. Best val_acc=71.6% at epoch 8/20 (overfitting after epoch 8 — train loss kept dropping but val loss rose). AC-6 (>60% val_acc) passed.
- **2026-03-16**: MVP-1 evaluation complete (50 episodes). Per-task success rates: collect_wood=8%, place_table=84%, collect_stone=10%. Model collapsed to place_table as dominant behavior — cannot distinguish tasks without instruction input. AC-8 (asymmetric success rates) confirmed. This is the motivating baseline for MVP-2: the gap between MVP-1 and MVP-2 measures the value of language grounding.
- **2026-03-16**: MVP-1 qualitative assessment: demo videos show the model exhibits near-random movement with occasional accidental task completion. The scripted experts use full game state (world map, player position, material coordinates) for precise navigation; the CNN sees only a 64×64 frame with no memory and no game state access. The 84% place_table rate is misleading — in 300 steps of semi-random movement, the agent stumbles into the place_table action sequence because it overlaps with the dominant training signal. The model does not navigate or plan.
- **2026-03-16**: VLA vs RL design rationale. RL answers "maximize this reward signal" — it needs per-task reward functions, millions of interactions, and produces policies that can't be redirected with language. VLA answers "given what you see and what I told you, what should you do?" — one model handles many tasks, steered by natural language instructions. Language becomes the universal interface for controlling agents. This is the paradigm shift from "one policy per task" to "one model that follows instructions." The project demonstrates this shift in a clean minimal setting.
- **2026-03-16**: Training VLA from scratch is infeasible with our data scale (~33K samples, 500 episodes). Pretrained encoders are required: a frozen vision encoder (pretrained on large image datasets) provides visual understanding, a frozen text encoder (pretrained on large text corpora) provides language understanding, and only a small trainable action head learns the mapping (visual_features + text_features) → action logits. This is the standard VLA paradigm (RT-2, Octo, OpenVLA in robotics), applied to a game environment.
- **2026-03-16**: MVP-2 VLA architecture: Frame (64×64) → [Frozen Vision Encoder] → vision_features; Instruction text → [Frozen Text Encoder] → text_features; concat(vision_features, text_features) → [Trainable MLP Action Head] → 8 action logits. Encoders are frozen — gradients only flow through the action head. This fits in 12 GB VRAM and is trainable on 500 episodes because the hard problems (seeing, reading) are solved by pretraining.
- **2026-03-16**: Research spec created (`specs/vla-components-research-spec.md`) to identify pretrained open-source vision and text encoder components for MVP-2. Self-contained document that can be delegated to any AI. Key constraints: RTX 4070 (12 GB VRAM), PyTorch, ≤ ~1B total params, permissive license, HuggingFace/timm availability. CLIP-family models are natural candidates (shared vision+text embedding space).

---

## Milestone Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| MVP-0a | Env wrapper + random rollout | **Done** (84 tests, all passing) |
| MVP-0b | Scripted policies + trajectory data | **Done** (104 tests, all passing) |
| MVP-1 | Vision-only imitation baseline ("V" only — no text) | **Done** — val_acc=71.6%, asymmetric success (8%/84%/10%) |
| MVP-2 | Instruction-conditioned VLA policy ("V+L→A") | Research phase — selecting pretrained components |
| MVP-3 | Portfolio polish | Planned |

### MVP-0a Deliverables

- `src/vla_agent/envs/crafter_env.py` - Gymnasium-style wrapper with 7-action reduced space
- `scripts/random_rollout.py` - Random policy rollout, saves frames + `episode.json` + optional mp4
- 84 tests across `test_crafter_env.py` and `test_random_rollout.py`

### MVP-0b Deliverables (spec: `specs/mvp-0b-spec.md`)

- `src/vla_agent/envs/crafter_env.py` - Expanded to 8-action space (added `make_wood_pickaxe`), info dict has `player_pos` + `player_facing`
- `src/vla_agent/policies.py` - Three scripted policies: `CollectWoodPolicy`, `PlaceTablePolicy`, `CollectStonePolicy` with shared `GreedyNavigator`
- `scripts/collect_trajectories.py` - Data collection script, saves `.npz` + `manifest.json`
- `tests/test_policies.py` + `tests/test_data_collection.py` - 20 new tests (104 total)
- Trajectory format: `observations (T+1, 64, 64, 3)`, `actions (T,)`, `rewards (T,)` per episode
- MVP-0a tests updated for 8-action space

### How to Run MVP-0b

```bash
# Run tests
uv run python -m pytest

# Collect trajectories (one command per policy)
uv run python scripts/collect_trajectories.py --policy collect_wood --num-episodes 10
uv run python scripts/collect_trajectories.py --policy place_table --num-episodes 10
uv run python scripts/collect_trajectories.py --policy collect_stone --num-episodes 10

# CLI defaults: --base-seed 42, --max-steps 300, --output-dir artifacts/trajectories/<policy>
# Outputs: artifacts/trajectories/<policy>/manifest.json + episode_NNN.npz files
```

**Status:** Done. Code implemented (104 tests passing), validation trajectories collected (10 episodes per policy, 100% success). Ready for MVP-1.

### MVP-1 Deliverables (spec: `specs/mvp-1-spec.md`)

- `src/vla_agent/data.py` - `TrajectoryDataset` (loads `.npz`, episode-level train/val split, action counts)
- `src/vla_agent/models.py` - `CrafterCNN` (Nature DQN encoder for 64×64, ~350K params)
- `scripts/train_imitation.py` - Behavioral cloning training (cross-entropy, Adam, MLflow tracking, `--no-mlflow` flag)
- `scripts/evaluate_policy.py` - Rollout evaluation (greedy argmax, per-task success rates)
- `tests/test_data_model.py` - 55 unit tests (dataset, model, save/load, reproducibility)
- `tests/test_training_eval.py` - 13 tests (2 unit + 11 integration, auto-skipped by default)
- `tests/conftest.py` - Auto-skips `integration` and `slow` tests unless `-m` is passed
- New dependency: `mlflow>=3.10.1`

### How to Run MVP-1

```bash
# Unit tests only (2s)
uv run python -m pytest

# Integration tests (30s, spawns training/eval subprocesses)
uv run python -m pytest -m integration

# Full training (20 epochs on real trajectories)
uv run python scripts/train_imitation.py \
    --data-dirs artifacts/trajectories/collect_wood artifacts/trajectories/place_table artifacts/trajectories/collect_stone \
    --output-dir artifacts/models/mvp1 \
    --epochs 20 --batch-size 64 --lr 1e-3 --seed 42

# Evaluation (50 episodes)
uv run python scripts/evaluate_policy.py \
    --model artifacts/models/mvp1/best_model.pt \
    --num-episodes 50 --base-seed 1000 \
    --output-dir artifacts/eval/mvp1
```

### MVP-1 Results

- **Training:** 20 epochs, best val_acc=71.6% at epoch 8. Overfitting after epoch 8 (train loss ↓, val loss ↑). AC-6 (>60% val_acc) passed.
- **Evaluation (50 episodes):**
  - `collect_wood`: 4/50 (8.0%)
  - `place_table`: 42/50 (84.0%)
  - `collect_stone`: 5/50 (10.0%)
- **Interpretation:** Model collapsed to `place_table` as dominant behavior. Without instruction input, it cannot distinguish tasks — it learned one mixed policy. AC-8 (asymmetric success rates) confirmed. This baseline motivates MVP-2: instruction conditioning should enable task-specific behavior.

**Status:** Done. All 11 acceptance criteria met. Artifacts in `artifacts/models/mvp1/` and `artifacts/eval/mvp1/`.

### MVP-1 Qualitative Assessment

Demo videos (`artifacts/demo/`) show the model exhibits near-random movement with occasional accidental task completion. The scripted experts use full game state (world map, player position, material coordinates) for precise pathfinding and multi-step plans. The CNN model sees only a single 64×64 frame — no world map, no coordinates, no inventory, no memory. Each frame is an independent decision.

The 84% place_table success rate is misleading: in 300 steps of semi-random movement, the agent stumbles into the place_table action sequence because those actions (move, chop, place) overlap with the dominant training signal. The model does not navigate or plan — it just replays the statistically dominant action pattern.

This is the expected and intended result. MVP-1 exists to establish the baseline that MVP-2 improves upon.

### MVP-2 Preparation

- **Research spec**: `specs/vla-components-research-spec.md` — self-contained document for identifying pretrained open-source vision and text encoder components
- **Key constraints**: RTX 4070 (12 GB VRAM), PyTorch, ≤ ~1B total params, permissive license, HuggingFace/timm availability
- **Architecture**: Frozen vision encoder + frozen text encoder + trainable MLP action head (see VLA Architecture section above)
- **Expected outcome**: Per-task success rates should rise across all 3 tasks when instructions are provided, proving that language grounding enables task-specific behavior

---

## Rule #1: Document Everything On The Fly

Every significant decision, convention, or discovery must be added to this file immediately.
Do not defer documentation.

## Rule #2: UV Only - No Pip

- Use `uv` for all package management, venv creation, and script execution.
- Commands: `uv sync`, `uv run pytest`, `uv run python`, `uv add <pkg>`.
- Never use `pip install`, `pip freeze`, `python -m pip`, or bare `python` outside of `uv run`.

## Rule #3: Agentic TDD Pipeline Is The Only Workflow

All production code is developed through `specs/agentic-pipeline.md`:

```text
spec (human-approved) -> tests -> test review -> implement -> validate -> code review -> done
```

- Every feature starts as an approved spec in `specs/`.
- Tests are written first, reviewed, then frozen.
- Implementation is written against frozen tests.
- After tests pass, a validation stage runs scripts/code from the spec end-to-end.
- No ad-hoc coding outside this pipeline.

## Rule #4: Spec-Driven Development

- Specs live in `specs/` and must be approved before entering the pipeline.
- Specs define acceptance criteria that map directly to tests.
- If a spec is ambiguous, ask. Do not guess.

## Rule #5: No Unnecessary Files

- Do not create placeholder files, empty modules, or stubs for later.
- Only create files required by the current spec.
- No README, docs, or notebooks unless a spec calls for them.

## Rule #6: Reproducibility

- All randomness must flow from explicit seeds.
- No global mutable state.
- Tests must be deterministic.

## Rule #7: Git Commits

- Commit messages are one line only.
- Never mention AI, co-authors, or tools.

## Rule #8: Keep It Simple

- Hardcode before configuring.
- Minimal dependencies.
- Prefer the smallest correct diff.

## Rule #9: Preserve Existing Document Encoding & Typography

- Do not rewrite existing Unicode symbols, typography, or file encoding in docs/specs unless the user explicitly requests it.
- Do not replace arrows, box-drawing characters, bullets, quotes, or similar formatting with different characters as a side effect of unrelated edits.
- If a document requires content edits, preserve its existing visible text formatting exactly unless the requested task is specifically about documentation formatting.

## Rule #10: No Monkey-Patching Library Internals

- Never patch, replace, or override functions/methods from third-party libraries at import time or runtime.
- If a library API doesn't work as expected (e.g., Windows path handling), use the library's own configuration mechanisms or pass correctly formatted inputs.
- If no clean solution exists, isolate the workaround in one wrapper function at the call site — not in `__init__.py` via global patching.
- Monkey-patches break silently on library upgrades and create invisible coupling.

## Rule #11: Test Tier Separation

- **Unit tests** run in seconds, use no subprocesses, no network, no GPU, no Crafter. They are the default `pytest` target.
- **Integration tests** (marked `@pytest.mark.integration`) may spawn subprocesses, train models, or run environment rollouts. They are **skipped by default** and run only via `pytest -m integration`.
- **Slow tests** (marked `@pytest.mark.slow`) are long-running integration tests (full training, 50+ episode evals). Skipped by default, run via `pytest -m slow`.
- `tests/conftest.py` enforces this: unmarked `pytest` runs skip `integration` and `slow` automatically.
- A test that launches training or evaluation is **never** a unit test — always mark it `@pytest.mark.integration` or `@pytest.mark.slow`.

## Rule #12: Config Consistency Across Outputs

- When the same config value is written to multiple outputs (JSON logs, MLflow params, console), use identical types and representations everywhere.
- Do not use `None`/null in one output and `"none"` (string) in another for the same field.
- Define the canonical representation once and reuse it.

---

## Tech Stack

| Concern | Tool |
|---------|------|
| Language | Python 3.11+ |
| Package manager | UV (only) |
| ML framework | PyTorch |
| Environment | Crafter |
| Env interface | Gymnasium-style custom wrapper |
| Testing | pytest |
| Linting/formatting | ruff |
| Video export | imageio[ffmpeg] |
| Experiment tracking | TBD |

## Package Layout

```text
src/vla_agent/         # library code (no prints, no scripts)
scripts/               # runnable scripts (CLI entry points)
tests/                 # pytest tests
specs/                 # approved specs
.claude/agents/        # subagent definitions
```

## Conventions

- Test files: `tests/test_<module>.py`
- Integration tests: `@pytest.mark.integration`
- Spec files: `specs/<task-id>-spec.md`
- No print in library code. Only scripts print.
- Imports: absolute from `vla_agent`

## TODO / Consider Later

- Pin Python version via UV with `uv python pin 3.12`
- Add `[project.scripts]` entries in `pyproject.toml`

