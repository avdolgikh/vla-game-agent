# Agentic TDD Pipeline

A fully automated test-driven development pipeline powered by AI agents. One command takes a human-approved spec and delivers tested, reviewed, trained code — with no manual intervention after launch.

```
uv run python scripts/run_pipeline.py <task-id> --provider codex
```

---

## Motivation

Traditional TDD requires a developer to manually cycle through writing tests, implementing code, running tests, and reviewing. This pipeline automates the entire loop using AI agents in specialized roles: a test-writer generates tests from a spec, a reviewer checks them, an implementer writes code to pass the tests, and a reviewer verifies the result. Each agent sees only what it needs and cannot modify artifacts outside its role.

The result: write a spec, run one command, walk away. Come back to green tests, reviewed code, and (if the spec defines it) trained models with verified metrics.

---

## How It Works

The pipeline is an 8-stage state machine. A human approves a spec. Everything after that is automated.

### State Machine

```
SPEC_APPROVED          (human approves spec)
    |
    v
TESTS_GENERATED        Stage 1: AI writes tests from spec
    |
    v
TESTS_FROZEN      <-+  Stage 2: AI reviews tests (approve/revise loop)
    |               |       max 4 revision cycles
    +-- revise? ----+
    |
    v
CODE_IMPLEMENTED       Stage 3: AI implements code against frozen tests
    |
    v
CODE_VALIDATED         Stage 4: AI runs pytest to confirm all tests pass
    |
    v
CODE_REVIEWED     <-+  Stage 5: AI reviews code (approve/revise loop)
    |               |       max 4 revision cycles
    +-- revise? ----+
    |
    v
ARTIFACTS_PRODUCED     Stage 6: Run training command from spec
    |
    v
ARTIFACTS_VALIDATED    Stage 7: Run evaluation command from spec
    |
    v
VERIFIED               Stage 8: Check acceptance criteria (metrics thresholds)
```

Stages 6-8 are optional — they only run if the spec contains an `## Artifact Pipeline` section. Specs without it stop at `CODE_REVIEWED`.

### Role Separation

Each stage is executed by an AI agent in a specific role:

| Role | Purpose | Capabilities |
|------|---------|-------------|
| **test-writer** | Write tests from spec | Read, Write, Edit, Grep, Glob, Bash |
| **implementer** | Write code to pass tests | Read, Write, Edit, Grep, Glob, Bash |
| **reviewer** | Review tests or code | Read, Grep, Glob, Bash (no Write/Edit) |

The reviewer cannot modify files — it can only read and judge. This separation prevents a single agent from both writing and approving its own work.

### Capability Tiers

Roles map to capability tiers, not specific model names:

- **Economy tier** (test-writer, implementer): Optimized for code generation speed
- **Premium tier** (reviewer): Optimized for reasoning and judgment

Concrete model names live only in provider-specific adapter code, keeping the pipeline core vendor-agnostic.

---

## Provider Architecture

The pipeline is provider-agnostic. The core orchestrator (`src/vla_agent/pipeline/core.py`) knows nothing about specific AI services. It delegates execution to provider adapters that implement a simple protocol:

```python
class Provider(Protocol):
    name: str

    def run_role(
        self,
        *,
        role: str,          # "test-writer", "implementer", "reviewer"
        prompt: str,         # Full prompt with spec, context, instructions
        repo_root: Path,
        schema: dict | None, # JSON schema for reviewer output
    ) -> ProviderExecution:
        ...
```

### Available Providers

| Provider | Backend | Status |
|----------|---------|--------|
| `codex` | OpenAI Codex CLI | Active, battle-tested |
| `claude` | Claude Code CLI | Implemented, needs actualization |
| `gemini` | Gemini CLI (local) | Spec ready, not yet implemented |

Adding a new provider means writing one adapter file (~50-100 lines) that translates the `run_role` call into the provider's CLI invocation.

---

## Guardrails

### Frozen Tests

After the reviewer approves tests (Stage 2), a SHA-256 hash of the `tests/` directory is recorded. Before every subsequent stage, the pipeline verifies this hash. If any test file has been modified, the pipeline fails immediately. This prevents the implementer from "cheating" by modifying tests to make them pass.

### Reviewer Immutability

Before and after every reviewer stage, the pipeline hashes the entire repository working tree. If the reviewer modified any files during its execution, the pipeline fails. Reviewers are read-only by design.

### Retry with Reset

Stages 6-8 (artifact production) use a fix-and-retry loop. If training crashes or metrics fail acceptance:

1. The implementer agent diagnoses and fixes the code
2. The frozen-test hash is re-verified (the fix must not break tests)
3. Pytest is re-run as a gate
4. The pipeline restarts from training (Stage 6)

Any code fix invalidates trained artifacts, so the loop always resets to the beginning of the artifact stages.

---

## Auto-Resume

Pipeline state is persisted to `.pipeline-state/<task-id>.json` after each completed stage:

```json
{
  "task": "mvp-2.3",
  "stage": "CODE_REVIEWED",
  "iteration": 0,
  "provider": "codex",
  "frozen_tests_hash": "da1261ba..."
}
```

On startup, the pipeline reads this file and skips all completed stages. If the process is interrupted (machine restart, network failure, timeout), re-running the same command picks up exactly where it left off.

---

## Artifact Pipeline (Stages 6-8)

Specs can define an `## Artifact Pipeline` section in YAML format:

```yaml
## Artifact Pipeline

### Training
command: uv run python scripts/train_imitation.py --model-type vla-cnn ...
required_files:
  - artifacts/models/mvp2.3/best_model.pt
metrics_file: artifacts/models/mvp2.3/train_log.json
metrics_checks:
  - path: best_val_acc
    op: ">"
    value: 0.60
    label: "val_acc above 60%"

### Evaluation
command: uv run python scripts/evaluate_policy.py ...
required_files:
  - artifacts/eval/mvp2.3/eval_results.json
metrics_file: artifacts/eval/mvp2.3/eval_results.json
metrics_checks:
  - path: success_rates.collect_wood
    op: ">"
    value: 0.08
    label: "collect_wood exceeds baseline"

### Acceptance
summary_file: artifacts/eval/mvp2.3/eval_results.json
all_checks_must_pass: false
min_checks_pass: 1
```

Stage 6 runs the training command, checks for required output files, and validates metrics. Stage 7 does the same for evaluation. Stage 8 aggregates all checks and declares the pipeline VERIFIED or FAILED.

---

## Real Example: MVP-2.3

The MVP-2.3 milestone (trainable CNN replacing frozen ConvNeXt) was delivered entirely through this pipeline:

```bash
uv run python scripts/run_pipeline.py mvp-2.3 --provider codex
```

**What happened automatically:**

1. **Test generation** — Agent wrote CNN-specific tests for the new `vision_type="cnn"` parameter
2. **Test review** — Reviewer flagged missing coverage (AC-2, AC-4, AC-8). Two revision cycles.
3. **Tests frozen** — Hash `da1261ba...` recorded
4. **Implementation** — Agent added `vision_type` to `CrafterVLA`, updated training/evaluation scripts
5. **Validation** — 143 tests passed, 98 skipped
6. **Code review** — Approved on first pass
7. **Training** — 20 epochs on CUDA, best val_acc=76.8%
8. **Evaluation** — 150 episodes (50 per instruction), collect_wood 38%, place_table 76%, collect_stone 6%
9. **Acceptance** — 2/2 checks passed. Pipeline VERIFIED.

Total: one command, zero human intervention after spec approval. The pipeline wrote tests, implemented code, trained a model, and verified it met acceptance criteria.

---

## How to Use

### 1. Write a spec

Create `specs/<task-id>-spec.md` with:
- Acceptance criteria (mapped directly to tests)
- Files to modify
- Optionally, an `## Artifact Pipeline` section for training/evaluation

### 2. Run the pipeline

```bash
uv run python scripts/run_pipeline.py <task-id> --provider codex
```

### 3. Monitor progress

- Live output is printed to the terminal
- Full transcript is saved to `.pipeline-state/<task-id>.log`
- State file at `.pipeline-state/<task-id>.json`

### 4. Resume after interruption

Re-run the same command. The pipeline reads the state file and continues from the last completed stage.

---

## Design Decisions

**External orchestrator + focused agents.** The Python orchestrator owns the state machine, stage transitions, and guardrails. Agents are stateless — they receive a prompt, do their job, and return output. This separation means agents can be swapped (different providers, different models) without changing the pipeline logic.

**Role-based capability tiers, not model names.** The pipeline core says "use an economy-tier agent for implementation." The provider adapter decides that means "gpt-5.1-codex-mini" or "claude-sonnet" or "gemini-pro." Model names change; capability requirements don't.

**JSON reviewer contracts.** Reviewers must return structured JSON with `decision` (approve/revise), `summary`, and `blocking` (list of issues). This makes reviewer output machine-parseable. If the output is malformed, the pipeline attempts one automatic repair pass before failing.

**Hash-based guardrails over trust.** The pipeline doesn't trust that the reviewer won't edit files — it verifies with SHA-256 hashes. The pipeline doesn't trust that the implementer won't touch tests — it verifies the frozen hash. Trust is not a security model.

**Specs as the single source of truth.** Every pipeline run starts from a human-approved spec. The spec defines what to test, what to build, and what success looks like. Agents cannot expand scope beyond the spec.
