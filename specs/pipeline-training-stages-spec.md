# Spec: Pipeline Training & Verification Stages

## Status

Draft

## Goal

Extend the agentic TDD pipeline with three new stages that close the gap between "code approved" and "artifacts verified." Today the pipeline stops at DONE (code review approved). The human must then manually run training, evaluation, and check acceptance criteria. These new stages automate that entire tail.

After this change, a single `run_pipeline.py` invocation goes from approved spec all the way to verified artifacts — no human in the loop.

---

## Problem Statement

The current pipeline covers:

```
SPEC_APPROVED → TESTS_GENERATED → TESTS_FROZEN → CODE_IMPLEMENTED → CODE_VALIDATED → DONE
```

"DONE" means the code compiles, tests pass, and a reviewer approved it. It does **not** mean:
- The model trains successfully on real data
- The model meets accuracy/performance targets
- Evaluation produces results that satisfy acceptance criteria

In MVP-2, this gap required hours of manual work: running training (debugging StochasticDepth, trying hyperparameters), running evaluation (150 episodes), and comparing results to spec thresholds. All of this should be pipeline-automated.

---

## Design

### New Stage Order

```
SPEC_APPROVED     → 0
TESTS_GENERATED   → 1
TESTS_FROZEN      → 2
CODE_IMPLEMENTED  → 3
CODE_VALIDATED    → 4
CODE_REVIEWED     → 5  (renamed from DONE)
TRAINED           → 6  (NEW)
EVALUATED         → 7  (NEW)
VERIFIED          → 8  (NEW — terminal state, replaces DONE)
```

### Spec Extensions

Specs that include training/evaluation must define a new `## Artifact Pipeline` section. This section is machine-parseable and tells the pipeline what to run and what to check.

Example (as it would appear in `mvp-2-spec.md`):

```markdown
## Artifact Pipeline

### Training
command: >
  uv run python scripts/train_imitation.py
  --model-type vla
  --data-dirs artifacts/trajectories/collect_wood artifacts/trajectories/place_table artifacts/trajectories/collect_stone
  --output-dir artifacts/models/mvp2
  --experiment-name mvp2
  --epochs 20 --batch-size 64 --lr 1e-3 --seed 42 --no-mlflow
required_files:
  - artifacts/models/mvp2/best_model.pt
  - artifacts/models/mvp2/train_log.json
metrics_file: artifacts/models/mvp2/train_log.json
metrics_checks:
  - path: best_val_acc
    op: ">"
    value: 0.45
    label: "AC-6: val_acc soft target"

### Evaluation
command: >
  uv run python scripts/evaluate_policy.py
  --model artifacts/models/mvp2/best_model.pt
  --policy-type vla
  --num-episodes 50 --base-seed 1000
  --output-dir artifacts/eval/mvp2
required_files:
  - artifacts/eval/mvp2/eval_results.json
metrics_file: artifacts/eval/mvp2/eval_results.json
metrics_checks:
  - path: success_rates.collect_wood
    op: ">"
    value: 0.08
    label: "AC-8: collect_wood exceeds MVP-1 baseline"

### Acceptance
summary_file: artifacts/eval/mvp2/eval_results.json
all_checks_must_pass: false
min_checks_pass: 1
```

Specs that do **not** have an `## Artifact Pipeline` section skip stages 6-8 and terminate at CODE_REVIEWED (same behavior as today).

---

## Stage 6: Training

**Trigger:** CODE_REVIEWED reached.

**What the pipeline does:**

1. Parse the spec's `## Artifact Pipeline → Training` block.
2. Run the training `command` as a subprocess with `cwd=repo_root`.
3. Stream stdout/stderr to the pipeline log.
4. If the process exits non-zero → `EXIT_TRAINING_FAILED`.
5. Verify all `required_files` exist and are non-empty.
6. Load `metrics_file` (JSON). For each entry in `metrics_checks`:
   - Navigate the JSON by `path` (dot-separated, e.g., `best_val_acc` or `success_rates.collect_wood`).
   - Apply the comparison `op` against `value`.
   - Log pass/fail with the `label`.
7. If any metrics check fails → log a warning but do NOT fail the pipeline (metrics may be soft targets). Record which checks passed/failed in the pipeline state.
8. Save state: `TRAINED`.

**Timeout:** Training may take hours on CPU. The pipeline should not impose a hard timeout. If the user wants one, they can wrap the pipeline invocation.

**No LLM invocation.** This stage runs a deterministic script — no provider is involved.

### State

```json
{
  "stage": "TRAINED",
  "training_metrics": {
    "best_val_acc": 0.516,
    "checks": [
      {"label": "AC-6: val_acc soft target", "passed": true, "actual": 0.516, "threshold": 0.45}
    ]
  }
}
```

---

## Stage 7: Evaluation

**Trigger:** TRAINED reached.

**What the pipeline does:**

1. Parse the spec's `## Artifact Pipeline → Evaluation` block.
2. Run the evaluation `command` as a subprocess.
3. Stream stdout/stderr to the pipeline log.
4. If the process exits non-zero → `EXIT_EVALUATION_FAILED`.
5. Verify all `required_files` exist and are non-empty.
6. Load `metrics_file` and run `metrics_checks` (same logic as Stage 6).
7. Record results in pipeline state.
8. Save state: `EVALUATED`.

**No LLM invocation.** Deterministic script execution.

### State

```json
{
  "stage": "EVALUATED",
  "evaluation_metrics": {
    "success_rates": {"collect_wood": 0.72, "place_table": 0.22, "collect_stone": 0.0},
    "checks": [
      {"label": "AC-8: collect_wood exceeds MVP-1 baseline", "passed": true, "actual": 0.72, "threshold": 0.08}
    ]
  }
}
```

---

## Stage 8: Acceptance Verification

**Trigger:** EVALUATED reached.

**What the pipeline does:**

1. Parse the spec's `## Artifact Pipeline → Acceptance` block.
2. Load `summary_file` (the evaluation results JSON).
3. Collect all `metrics_checks` results from stages 6 and 7.
4. Apply the acceptance rule:
   - If `all_checks_must_pass: true` → all checks must pass → VERIFIED or EXIT_ACCEPTANCE_FAILED.
   - If `all_checks_must_pass: false` → at least `min_checks_pass` checks must pass.
5. Log a summary table:
   ```
   === Acceptance Verification ===
   [PASS] AC-6: val_acc soft target (0.516 > 0.45)
   [PASS] AC-8: collect_wood exceeds MVP-1 baseline (0.72 > 0.08)
   Result: 2/2 checks passed. Pipeline VERIFIED.
   ```
6. Save state: `VERIFIED` (new terminal state).

**No LLM invocation.** Pure arithmetic.

---

## Exit Codes (new)

| Code | Constant | Meaning |
|------|----------|---------|
| 11 | `EXIT_TRAINING_FAILED` | Training command exited non-zero |
| 12 | `EXIT_EVALUATION_FAILED` | Evaluation command exited non-zero |
| 13 | `EXIT_ACCEPTANCE_FAILED` | Acceptance checks did not meet minimum |
| 14 | `EXIT_ARTIFACT_MISSING` | Required artifact file missing after run |

---

## Acceptance Criteria

### AC-1: Specs without Artifact Pipeline section are unaffected

Pipeline behavior is identical to today for specs that lack the `## Artifact Pipeline` section. Stage order change (DONE → CODE_REVIEWED → VERIFIED) is backward compatible: specs without artifact stages jump from CODE_REVIEWED directly to VERIFIED.

### AC-2: Training stage runs the spec-defined command

Given a spec with a Training block, the pipeline executes the command, verifies required files, and checks metrics. Non-zero exit code → `EXIT_TRAINING_FAILED`.

### AC-3: Evaluation stage runs the spec-defined command

Given a spec with an Evaluation block, the pipeline executes the command after training, verifies required files, and checks metrics. Non-zero exit code → `EXIT_EVALUATION_FAILED`.

### AC-4: Acceptance stage aggregates and decides

All metrics checks from stages 6 and 7 are collected. The acceptance rule (`all_checks_must_pass` / `min_checks_pass`) determines the final verdict. Pass → VERIFIED. Fail → `EXIT_ACCEPTANCE_FAILED`.

### AC-5: Pipeline state includes metrics

After each new stage, the pipeline state JSON includes the metrics values, check results (passed/failed, actual vs threshold), and labels.

### AC-6: Resume works across new stages

If the pipeline crashes mid-training or mid-evaluation, re-running it resumes from the last completed stage (same resume logic as existing stages).

### AC-7: Metrics path navigation works with nested JSON

`path: success_rates.collect_wood` correctly navigates `{"success_rates": {"collect_wood": 0.72}}` to extract `0.72`.

### AC-8: Logging

All three new stages log their activity to the pipeline transcript (`.pipeline-state/<task>.log`) in the same format as existing stages.

---

## Implementation Notes

- **No provider involvement.** Stages 6-8 are pure subprocess + JSON parsing. No LLM calls.
- **Spec parsing.** The `## Artifact Pipeline` section needs a simple YAML-like parser or could use actual YAML fenced blocks. The exact format is left to the implementer as long as it's machine-parseable and human-readable.
- **Timeout.** Training on CPU can take hours. The pipeline should not timeout — just log progress. The user kills it if needed.
- **Soft vs hard checks.** The `all_checks_must_pass` flag lets specs distinguish hard requirements (must pass to verify) from soft targets (informational, logged but not blocking).

---

## How to Test

### Unit tests

- Parse a mock spec with `## Artifact Pipeline` section → extract training/evaluation/acceptance configs.
- Metrics path navigation: `"best_val_acc"` → scalar, `"success_rates.collect_wood"` → nested.
- Acceptance logic: all_checks_must_pass true/false, min_checks_pass thresholds.
- State serialization with metrics.

### Integration tests

- Run pipeline on a trivial spec that includes a training command (`echo '{"best_val_acc": 0.9}' > /tmp/metrics.json`) → verify TRAINED state.
- Run pipeline with a failing training command → verify EXIT_TRAINING_FAILED.
- Run pipeline with acceptance checks that fail → verify EXIT_ACCEPTANCE_FAILED.

---

## Open Questions

1. **Should training retry on failure?** Currently: no. One attempt, fail fast. Retries with different hyperparameters would require a more complex spec format.
2. **Should the pipeline support multiple training runs (hyperparameter sweep)?** Currently: no. One command, one run. Hyperparameter search is out of scope.
3. **Should metrics checks support expressions beyond simple comparisons?** Currently: only `>`, `>=`, `<`, `<=`, `==`. Complex expressions (e.g., "at least 2 of 3 tasks improve") would need a mini-DSL.
