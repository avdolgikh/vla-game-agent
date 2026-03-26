# AGENTS.md - Repo-Wide Rules & Context

All agents (test-writer, implementer, reviewer) inherit these rules automatically.

---

## Project

**vla-game-agent** -- An instruction-conditioned vision-to-action agent for Crafter.
VLA-style game bot: takes a game frame + a text instruction -> predicts a player action.

### Why VLA, Not RL

Reinforcement learning answers "maximize this reward signal." It needs per-task reward functions, millions of environment interactions, and produces policies that can't be redirected with natural language. VLA (Vision-Language-Action) answers "given what you see and what I told you, what should you do?" One model handles many tasks, steered by language instructions. This is the paradigm shift from "one policy per task" to "one model that follows instructions."

The project demonstrates this shift in a clean, minimal setting. The core result is the gap between MVP-1 (vision-only, no instructions -> collapses to one behavior) and MVP-2 (vision + language -> task-specific behavior). That gap quantifies the value of language grounding for goal-directed behavior.

### VLA Architecture (MVP-2.3, final)

```
4 Frames (64x64 RGB)            Instruction ("collect wood")
        |                                  |
  [Trainable CNN]               [Frozen all-MiniLM-L6-v2]
  3-conv, 256-d per frame            384-d text embedding
        |                                  |
  [Mean Pool x4 frames]                   |
        |                                  |
        +-----------concat-----------------+
                      |
               [MLP Action Head]
                 640 -> 256 -> 8
                      |
                8 action logits
```

~504K trainable parameters (CNN + action head). Text encoder is frozen.

---

## Milestone Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| MVP-0a | Env wrapper + random rollout | **Done** (84 tests, all passing) |
| MVP-0b | Scripted policies + trajectory data | **Done** (104 tests, all passing) |
| MVP-1 | Vision-only imitation baseline ("V" only -- no text) | **Done** -- val_acc=71.6%, asymmetric success (8%/84%/10%) |
| MVP-2 | Instruction-conditioned VLA policy ("V+L->A") | **Done** -- val_acc=51.6%, collect_wood 72%/place_table 22%/collect_stone 0% |
| MVP-2.1 | 224x224 resize (proven quick win) | **Done** -- val_acc=60.9%, collect_wood 48%/place_table 12%/collect_stone 0% |
| MVP-2.2 | Frame stacking + wider head (temporal context) | **Done** -- val_acc=55.0%, collect_wood 58%/place_table 22%/collect_stone 0% |
| MVP-2.3 | Domain adaptation (trainable CNN replacing frozen ConvNeXt) | **Done** -- val_acc=76.8%, collect_wood 38%/place_table 76%/collect_stone 6% |
| Pipeline-ext | Artifact produce/validate/accept pipeline stages | **Done** -- stages 6-8 with implementer retry loop, 34 tests passing |
| MVP-3 | Portfolio polish | **Done** -- README, report.md, pipeline docs, plots, 9 demo videos |

---

## Detail Documents

- [Architecture Decisions Log](agents/architecture-decisions.md) -- chronological record of every significant decision
- [Experiment Log](agents/experiment-log.md) -- per-milestone deliverables, results, insights, and commands
- [Roadmap](agents/roadmap.md) -- next steps plan for future sessions

---

## Rules

### Rule #1: Document Everything On The Fly

Every significant decision, convention, or discovery must be added to this file (or the appropriate detail doc) immediately. Do not defer documentation.

### Rule #2: UV Only - No Pip

- Use `uv` for all package management, venv creation, and script execution.
- Commands: `uv sync`, `uv run pytest`, `uv run python`, `uv add <pkg>`.
- Never use `pip install`, `pip freeze`, `python -m pip`, or bare `python` outside of `uv run`.

### Rule #3: Agentic TDD Pipeline Is The Only Workflow

All production code is developed through `specs/agentic-pipeline.md`:

```text
spec (human-approved) -> tests -> test review -> implement -> validate -> code review -> done
```

- Every feature starts as an approved spec in `specs/`.
- Tests are written first, reviewed, then frozen.
- Implementation is written against frozen tests.
- After tests pass, a validation stage runs scripts/code from the spec end-to-end.
- No ad-hoc coding outside this pipeline.

### Rule #4: Spec-Driven Development

- Specs live in `specs/` and must be approved before entering the pipeline.
- Specs define acceptance criteria that map directly to tests.
- If a spec is ambiguous, ask. Do not guess.

### Rule #5: No Unnecessary Files

- Do not create placeholder files, empty modules, or stubs for later.
- Only create files required by the current spec.
- No README, docs, or notebooks unless a spec calls for them.

### Rule #6: Reproducibility

- All randomness must flow from explicit seeds.
- No global mutable state.
- Tests must be deterministic.

### Rule #7: Git Commits

- Commit messages are one line only.
- Never mention AI, co-authors, or tools.

### Rule #8: Keep It Simple

- Hardcode before configuring.
- Minimal dependencies.
- Prefer the smallest correct diff.

### Rule #9: Preserve Existing Document Encoding & Typography

- Do not rewrite existing Unicode symbols, typography, or file encoding in docs/specs unless the user explicitly requests it.
- Do not replace arrows, box-drawing characters, bullets, quotes, or similar formatting with different characters as a side effect of unrelated edits.

### Rule #10: No Monkey-Patching Library Internals

- Never patch, replace, or override functions/methods from third-party libraries at import time or runtime.
- If a library API doesn't work as expected, use the library's own configuration mechanisms or pass correctly formatted inputs.
- If no clean solution exists, isolate the workaround in one wrapper function at the call site -- not in `__init__.py` via global patching.

### Rule #11: Test Tier Separation

- **Unit tests** run in seconds, use no subprocesses, no network, no GPU, no Crafter. They are the default `pytest` target.
- **Integration tests** (marked `@pytest.mark.integration`) may spawn subprocesses, train models, or run environment rollouts. They are **skipped by default** and run only via `pytest -m integration`.
- **Slow tests** (marked `@pytest.mark.slow`) are long-running integration tests. Skipped by default, run via `pytest -m slow`.
- `tests/conftest.py` enforces this: unmarked `pytest` runs skip `integration` and `slow` automatically.
- A test that launches training or evaluation is **never** a unit test.

### Rule #12: Config Consistency Across Outputs

- When the same config value is written to multiple outputs (JSON logs, MLflow params, console), use identical types and representations everywhere.
- Do not use `None`/null in one output and `"none"` (string) in another for the same field.

### Rule #13: Never Manipulate sys.path or Python Environment

- **Never** add `sys.path` manipulation code to production modules.
- **Never** create files like `.python-version`, `_path.py`, `_bootstrap.py`, or `runtime.py` to work around import issues.
- **Never** rename, move, or delete `.venv/` or any virtualenv directory.
- If imports fail in a subprocess, the fix belongs in the **subprocess launcher**, not in the imported module's source code.

### Rule #14: Never Silently Degrade Hardware Acceleration

- If a spec command requires `--device cuda` and CUDA is unavailable, the command must **fail loudly**, not silently fall back to CPU.
- **Never** add `try/except` around CUDA detection that falls back to CPU without raising.
- **Never** remove `--device cuda` from a spec command to "fix" a CUDA error.

### Rule #15: Implementer Scope -- Fix the Bug, Not the World

- When the implementer agent is invoked for an artifact fix (Stage 6b), it must **only** fix the specific error reported.
- **Never** refactor unrelated files, restructure the project, or add new utility modules during an artifact fix.
- **Never** modify scripts (`scripts/*.py`) during an artifact fix.

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

## Package Layout

```text
src/vla_agent/         # library code (no prints, no scripts)
scripts/               # runnable scripts (CLI entry points)
tests/                 # pytest tests
specs/                 # approved specs
agents/                # detail docs (decisions, experiments, roadmap)
docs/                  # public documentation and assets
```

## Conventions

- Test files: `tests/test_<module>.py`
- Integration tests: `@pytest.mark.integration`
- Spec files: `specs/<task-id>-spec.md`
- No print in library code. Only scripts print.
- Imports: absolute from `vla_agent`
- CUDA torch configured in venv via `pyproject.toml` index. All commands use `uv run python` uniformly.
