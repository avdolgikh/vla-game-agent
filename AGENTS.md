# AGENTS.md - Repo-Wide Rules & Context

All agents (test-writer, implementer, reviewer) inherit these rules automatically.

---

## Project

**vla-game-agent** - A tiny instruction-conditioned vision-to-action agent for Crafter.
VLA-style game bot: takes a game frame + a text instruction -> predicts a player action.

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

---

## Milestone Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| MVP-0a | Env wrapper + random rollout | **Done** (84 tests, all passing) |
| MVP-0b | Scripted policies + trajectory data | **Done** (104 tests, all passing) |
| MVP-1 | Vision-only imitation baseline ("V" only — no text) | **Code done** — next: full training run |
| MVP-2 | Instruction-conditioned VLA policy ("V+L→A") | Planned |
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

**Status:** Code implemented and reviewed (79 unit + 11 integration tests passing). Next: full 20-epoch training run + 50-episode evaluation to validate AC-6 (>60% val_acc) and AC-8 (asymmetric success rates).

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

