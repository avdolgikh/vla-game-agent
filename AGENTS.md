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

---

## Milestone Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| MVP-0a | Env wrapper + random rollout | **Done** (84 tests, all passing) |
| MVP-0b | Scripted policies + trajectory data | Next up (no spec yet) |
| MVP-1 | Vision-only imitation baseline | Planned |
| MVP-2 | Instruction-conditioned policy | Planned |
| MVP-3 | Portfolio polish | Planned |

### MVP-0a Deliverables

- `src/vla_agent/envs/crafter_env.py` - Gymnasium-style wrapper with 7-action reduced space
- `scripts/random_rollout.py` - Random policy rollout, saves frames + `episode.json` + optional mp4
- 84 tests across `test_crafter_env.py` and `test_random_rollout.py`

### MVP-0b Ideas

- Scripted policies for `collect wood`, `collect stone`, `place table`
- Trajectory / dataset schema for `(obs, action, instruction, reward, metadata)`
- Data collection script that runs scripted policies and saves trajectories to disk

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
