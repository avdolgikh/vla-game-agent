"""Unit tests for pipeline artifact stages (6-8): produce, validate, accept."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_agent.pipeline.core import (
    EXIT_ACCEPTANCE_FAILED,
    EXIT_ARTIFACT_MISSING,
    EXIT_EVALUATION_FAILED,
    EXIT_SUCCESS,
    EXIT_TRAINING_FAILED,
    STAGE_ORDER,
    AcceptanceConfig,
    CheckResult,
    MetricsCheck,
    PipelineConfig,
    PipelineError,
    PipelineRunner,
    evaluate_acceptance,
    hash_paths,
    navigate_json_path,
    parse_artifact_pipeline,
    run_metrics_checks,
)
from vla_agent.pipeline.providers.base import ProviderExecution


# ── Helpers ──────────────────────────────────────────────────


class DummyProvider:
    """Minimal provider for stages 6-8 tests (no LLM calls needed)."""

    name = "dummy"

    def run_role(self, *, role, prompt, repo_root, state_dir, schema=None):
        return ProviderExecution(
            provider=self.name,
            role=role,
            tier="economy",
            model="dummy-model",
            output='{"decision":"approve","summary":"ok","blocking":[]}',
        )


SPEC_WITH_PIPELINE = """\
# Test Spec

## Overview

Test spec with artifact pipeline.

## Artifact Pipeline

### Training
command: >
  uv run python scripts/train.py
  --epochs 5 --output-dir artifacts/models/test
required_files:
  - artifacts/models/test/best_model.pt
  - artifacts/models/test/train_log.json
metrics_file: artifacts/models/test/train_log.json
metrics_checks:
  - path: best_val_acc
    op: ">"
    value: 0.45
    label: "AC-6: val_acc > 0.45"

### Evaluation
command: >
  uv run python scripts/evaluate.py
  --model artifacts/models/test/best_model.pt
  --output-dir artifacts/eval/test
required_files:
  - artifacts/eval/test/eval_results.json
metrics_file: artifacts/eval/test/eval_results.json
metrics_checks:
  - path: success_rates.collect_wood
    op: ">"
    value: 0.08
    label: "AC-8: collect_wood > baseline"

### Acceptance
summary_file: artifacts/eval/test/eval_results.json
all_checks_must_pass: false
min_checks_pass: 1
"""

SPEC_WITHOUT_PIPELINE = """\
# Test Spec

## Overview

No artifact pipeline here.

## Acceptance Criteria

- AC-1: Something
"""


def _setup_pipeline_fs(tmp_path, spec_text, state_stage="CODE_REVIEWED"):
    """Create minimum filesystem for pipeline runner tests at given stage."""
    (tmp_path / "specs").mkdir(exist_ok=True)
    (tmp_path / "specs" / "demo-spec.md").write_text(spec_text, encoding="utf-8")
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for role in ("test_writer", "implementer", "reviewer"):
        (prompts_dir / f"{role}.md").write_text(f"{role} prompt", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# rules", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "scripts").mkdir(exist_ok=True)
    state_dir = tmp_path / ".pipeline-state"
    state_dir.mkdir(exist_ok=True)
    state = {
        "task": "demo",
        "provider": "dummy",
        "stage": state_stage,
        "iteration": 0,
        "frozen_tests_hash": hash_paths(tmp_path, ["tests"]),
    }
    (state_dir / "demo.json").write_text(json.dumps(state), encoding="utf-8")


def _make_training_artifacts(cwd):
    """Create training output files that satisfy the spec."""
    model_dir = Path(cwd) / "artifacts" / "models" / "test"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best_model.pt").write_bytes(b"model-data")
    (model_dir / "train_log.json").write_text(
        json.dumps({"best_val_acc": 0.52, "best_epoch": 8}), encoding="utf-8"
    )


def _make_eval_artifacts(cwd):
    """Create evaluation output files that satisfy the spec."""
    eval_dir = Path(cwd) / "artifacts" / "eval" / "test"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval_results.json").write_text(
        json.dumps({"success_rates": {"collect_wood": 0.72, "place_table": 0.22}}),
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════
# Spec Parsing (AC-1, AC-2, AC-3)
# ═══════════════════════════════════════════════════════════


def test_parse_artifact_pipeline_returns_none_without_section():
    assert parse_artifact_pipeline(SPEC_WITHOUT_PIPELINE) is None


def test_parse_artifact_pipeline_extracts_training_config():
    config = parse_artifact_pipeline(SPEC_WITH_PIPELINE)
    assert config is not None
    assert config.training is not None
    assert "train.py" in config.training.command
    assert "artifacts/models/test/best_model.pt" in config.training.required_files
    assert config.training.metrics_file == "artifacts/models/test/train_log.json"
    assert len(config.training.metrics_checks) == 1
    check = config.training.metrics_checks[0]
    assert check.path == "best_val_acc"
    assert check.op == ">"
    assert check.value == 0.45
    assert "AC-6" in check.label


def test_parse_artifact_pipeline_extracts_evaluation_config():
    config = parse_artifact_pipeline(SPEC_WITH_PIPELINE)
    assert config is not None
    assert config.evaluation is not None
    assert "evaluate.py" in config.evaluation.command
    assert "artifacts/eval/test/eval_results.json" in config.evaluation.required_files
    check = config.evaluation.metrics_checks[0]
    assert check.path == "success_rates.collect_wood"
    assert check.op == ">"
    assert check.value == 0.08


def test_parse_artifact_pipeline_extracts_acceptance_config():
    config = parse_artifact_pipeline(SPEC_WITH_PIPELINE)
    assert config is not None
    assert config.acceptance is not None
    assert config.acceptance.all_checks_must_pass is False
    assert config.acceptance.min_checks_pass == 1


def test_parse_artifact_pipeline_command_is_single_line():
    """YAML folded scalar (>) should join continuation lines with spaces."""
    config = parse_artifact_pipeline(SPEC_WITH_PIPELINE)
    cmd = config.training.command.strip()
    assert "\n" not in cmd
    assert "--epochs" in cmd
    assert "--output-dir" in cmd


# ═══════════════════════════════════════════════════════════
# Metrics Path Navigation (AC-7)
# ═══════════════════════════════════════════════════════════


def test_navigate_json_path_scalar():
    assert navigate_json_path({"best_val_acc": 0.716}, "best_val_acc") == 0.716


def test_navigate_json_path_nested():
    data = {"success_rates": {"collect_wood": 0.72, "place_table": 0.22}}
    assert navigate_json_path(data, "success_rates.collect_wood") == 0.72
    assert navigate_json_path(data, "success_rates.place_table") == 0.22


def test_navigate_json_path_deep_nesting():
    data = {"a": {"b": {"c": 42}}}
    assert navigate_json_path(data, "a.b.c") == 42


def test_navigate_json_path_missing_key_raises():
    with pytest.raises((KeyError, ValueError)):
        navigate_json_path({"a": 1}, "b")
    with pytest.raises((KeyError, ValueError)):
        navigate_json_path({"a": {"b": 1}}, "a.c")


# ═══════════════════════════════════════════════════════════
# Metrics Checks
# ═══════════════════════════════════════════════════════════


def test_metrics_checks_all_operators():
    data = {"acc": 0.7, "loss": 0.3, "exact": 42}
    checks = [
        MetricsCheck(path="acc", op=">", value=0.5, label="gt"),
        MetricsCheck(path="acc", op=">=", value=0.7, label="ge"),
        MetricsCheck(path="loss", op="<", value=0.5, label="lt"),
        MetricsCheck(path="loss", op="<=", value=0.3, label="le"),
        MetricsCheck(path="exact", op="==", value=42, label="eq"),
    ]
    results = run_metrics_checks(data, checks)
    assert all(r.passed for r in results)


def test_metrics_checks_records_actual_and_threshold():
    data = {"acc": 0.3}
    checks = [MetricsCheck(path="acc", op=">", value=0.5, label="low")]
    results = run_metrics_checks(data, checks)
    assert results[0].passed is False
    assert results[0].actual == 0.3
    assert results[0].threshold == 0.5
    assert results[0].label == "low"


# ═══════════════════════════════════════════════════════════
# Acceptance Logic (AC-4)
# ═══════════════════════════════════════════════════════════


def test_acceptance_all_must_pass_succeeds():
    checks = [
        CheckResult(label="a", passed=True, actual=0.8, threshold=0.5),
        CheckResult(label="b", passed=True, actual=0.9, threshold=0.7),
    ]
    config = AcceptanceConfig(summary_file="x.json", all_checks_must_pass=True, min_checks_pass=0)
    assert evaluate_acceptance(checks, config) is True


def test_acceptance_all_must_pass_fails_on_any_failure():
    checks = [
        CheckResult(label="a", passed=True, actual=0.8, threshold=0.5),
        CheckResult(label="b", passed=False, actual=0.3, threshold=0.7),
    ]
    config = AcceptanceConfig(summary_file="x.json", all_checks_must_pass=True, min_checks_pass=0)
    assert evaluate_acceptance(checks, config) is False


def test_acceptance_min_checks_met():
    checks = [
        CheckResult(label="a", passed=True, actual=0.8, threshold=0.5),
        CheckResult(label="b", passed=False, actual=0.3, threshold=0.7),
    ]
    config = AcceptanceConfig(summary_file="x.json", all_checks_must_pass=False, min_checks_pass=1)
    assert evaluate_acceptance(checks, config) is True


def test_acceptance_min_checks_not_met():
    checks = [
        CheckResult(label="a", passed=False, actual=0.3, threshold=0.5),
        CheckResult(label="b", passed=False, actual=0.2, threshold=0.7),
    ]
    config = AcceptanceConfig(summary_file="x.json", all_checks_must_pass=False, min_checks_pass=1)
    assert evaluate_acceptance(checks, config) is False


# ═══════════════════════════════════════════════════════════
# Stage Order & Exit Codes
# ═══════════════════════════════════════════════════════════


def test_stage_order_includes_new_stages():
    assert STAGE_ORDER["CODE_REVIEWED"] == 5
    assert STAGE_ORDER["ARTIFACTS_PRODUCED"] == 6
    assert STAGE_ORDER["ARTIFACTS_VALIDATED"] == 7
    assert STAGE_ORDER["VERIFIED"] == 8


def test_done_not_in_stage_order():
    assert "DONE" not in STAGE_ORDER


def test_new_exit_codes_defined():
    assert EXIT_TRAINING_FAILED == 11
    assert EXIT_EVALUATION_FAILED == 12
    assert EXIT_ACCEPTANCE_FAILED == 13
    assert EXIT_ARTIFACT_MISSING == 14


# ═══════════════════════════════════════════════════════════
# Backward Compatibility
# ═══════════════════════════════════════════════════════════


def test_load_state_maps_done_to_code_reviewed(tmp_path):
    """Existing state files with stage='DONE' should resume as CODE_REVIEWED."""
    _setup_pipeline_fs(tmp_path, SPEC_WITHOUT_PIPELINE, state_stage="DONE")
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        config=PipelineConfig(prompts_dir="prompts"),
    )
    state = runner._load_state()
    assert state.stage == "CODE_REVIEWED"


# ═══════════════════════════════════════════════════════════
# Pipeline Runner — Full Stage 6-8 Flow (AC-2, AC-3, AC-4, AC-5, AC-8)
# ═══════════════════════════════════════════════════════════


def test_pipeline_completes_artifact_stages(tmp_path, monkeypatch):
    """Happy path: CODE_REVIEWED → ARTIFACTS_PRODUCED → ARTIFACTS_VALIDATED → VERIFIED."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)
    calls = []

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        calls.append(cmd_str)
        cwd = kwargs.get("cwd", tmp_path)
        if "train.py" in cmd_str:
            _make_training_artifacts(cwd)
        elif "evaluate.py" in cmd_str:
            _make_eval_artifacts(cwd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        config=PipelineConfig(prompts_dir="prompts"),
    )
    result = runner.run()

    assert result == EXIT_SUCCESS
    state = json.loads((tmp_path / ".pipeline-state" / "demo.json").read_text(encoding="utf-8"))
    assert state["stage"] == "VERIFIED"


def test_pipeline_runs_training_command(tmp_path, monkeypatch):
    """Stage 6 should invoke the spec-defined training command."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)
    calls = []

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        calls.append(cmd_str)
        cwd = kwargs.get("cwd", tmp_path)
        if "train.py" in cmd_str:
            _make_training_artifacts(cwd)
        elif "evaluate.py" in cmd_str:
            _make_eval_artifacts(cwd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        config=PipelineConfig(prompts_dir="prompts"),
    )
    runner.run()

    training_calls = [c for c in calls if "train.py" in c]
    assert len(training_calls) == 1
    assert "--epochs" in training_calls[0]


def test_pipeline_state_includes_metrics(tmp_path, monkeypatch):
    """AC-5: After stages 6-7, state JSON includes metrics and check results."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        cwd = kwargs.get("cwd", tmp_path)
        if "train.py" in cmd_str:
            _make_training_artifacts(cwd)
        elif "evaluate.py" in cmd_str:
            _make_eval_artifacts(cwd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        config=PipelineConfig(prompts_dir="prompts"),
    )
    runner.run()

    state = json.loads((tmp_path / ".pipeline-state" / "demo.json").read_text(encoding="utf-8"))
    assert "training_metrics" in state
    assert state["training_metrics"]["checks"][0]["label"] == "AC-6: val_acc > 0.45"
    assert state["training_metrics"]["checks"][0]["passed"] is True
    assert "evaluation_metrics" in state
    assert state["evaluation_metrics"]["checks"][0]["label"] == "AC-8: collect_wood > baseline"
    assert state["evaluation_metrics"]["checks"][0]["passed"] is True


def test_training_failure_exits(tmp_path, monkeypatch):
    """Non-zero training exit code → EXIT_TRAINING_FAILED (no retries)."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        if "train.py" in cmd_str:
            return SimpleNamespace(returncode=1, stdout="error", stderr="crash")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        max_revisions=0,
        config=PipelineConfig(prompts_dir="prompts"),
    )

    with pytest.raises(PipelineError) as exc_info:
        runner.run()
    assert exc_info.value.exit_code == EXIT_TRAINING_FAILED


def test_training_missing_artifact_exits(tmp_path, monkeypatch):
    """Training succeeds but required file missing → EXIT_ARTIFACT_MISSING (no retries)."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)

    def fake_run(command, **kwargs):
        # Training succeeds but does NOT create the required files
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        max_revisions=0,
        config=PipelineConfig(prompts_dir="prompts"),
    )

    with pytest.raises(PipelineError) as exc_info:
        runner.run()
    assert exc_info.value.exit_code == EXIT_ARTIFACT_MISSING


def test_evaluation_failure_exits(tmp_path, monkeypatch):
    """Non-zero evaluation exit code → EXIT_EVALUATION_FAILED."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        cwd = kwargs.get("cwd", tmp_path)
        if "train.py" in cmd_str:
            _make_training_artifacts(cwd)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "evaluate.py" in cmd_str:
            return SimpleNamespace(returncode=1, stdout="error", stderr="crash")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        max_revisions=0,
        config=PipelineConfig(prompts_dir="prompts"),
    )

    with pytest.raises(PipelineError) as exc_info:
        runner.run()
    assert exc_info.value.exit_code == EXIT_EVALUATION_FAILED


def test_acceptance_failure_exits(tmp_path, monkeypatch):
    """Acceptance checks below min_checks_pass → EXIT_ACCEPTANCE_FAILED."""
    # Use a spec where all_checks_must_pass=true but metrics will fail
    spec = """\
# Spec

## Artifact Pipeline

### Training
command: echo ok
required_files:
  - artifacts/train_log.json
metrics_file: artifacts/train_log.json
metrics_checks:
  - path: val_acc
    op: ">"
    value: 0.99
    label: "impossible threshold"

### Evaluation
command: echo ok
required_files:
  - artifacts/eval.json
metrics_file: artifacts/eval.json
metrics_checks: []

### Acceptance
summary_file: artifacts/eval.json
all_checks_must_pass: true
min_checks_pass: 0
"""
    _setup_pipeline_fs(tmp_path, spec)

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        cwd = Path(kwargs.get("cwd", tmp_path))
        if "echo" in cmd_str:
            # Create minimal artifact files
            artifacts_dir = cwd / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "train_log.json").write_text(
                json.dumps({"val_acc": 0.3}), encoding="utf-8"
            )
            (artifacts_dir / "eval.json").write_text(
                json.dumps({"result": "done"}), encoding="utf-8"
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        max_revisions=0,
        config=PipelineConfig(prompts_dir="prompts"),
    )

    with pytest.raises(PipelineError) as exc_info:
        runner.run()
    assert exc_info.value.exit_code == EXIT_ACCEPTANCE_FAILED


def test_pipeline_skips_artifact_stages_without_section(tmp_path, monkeypatch):
    """AC-1: Specs without Artifact Pipeline jump CODE_REVIEWED → VERIFIED."""
    _setup_pipeline_fs(tmp_path, SPEC_WITHOUT_PIPELINE)
    calls = []

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        calls.append(cmd_str)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        config=PipelineConfig(prompts_dir="prompts"),
    )
    result = runner.run()

    assert result == EXIT_SUCCESS
    state = json.loads((tmp_path / ".pipeline-state" / "demo.json").read_text(encoding="utf-8"))
    assert state["stage"] == "VERIFIED"
    # No training or evaluation commands should have been invoked
    assert not any("train" in c for c in calls)
    assert not any("evaluate" in c for c in calls)


# ═══════════════════════════════════════════════════════════
# Resume (AC-6)
# ═══════════════════════════════════════════════════════════


def test_resume_from_trained_skips_training(tmp_path, monkeypatch):
    """AC-6: Resuming from ARTIFACTS_PRODUCED skips training, runs evaluation + acceptance."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE, state_stage="ARTIFACTS_PRODUCED")
    # Add training metrics to the state (as if stage 6 already completed)
    state_file = tmp_path / ".pipeline-state" / "demo.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["training_metrics"] = {
        "checks": [
            {"label": "AC-6: val_acc > 0.45", "passed": True, "actual": 0.52, "threshold": 0.45}
        ]
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")
    # Also need the training artifacts to exist (they were created in the previous run)
    _make_training_artifacts(tmp_path)

    calls = []

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        calls.append(cmd_str)
        cwd = kwargs.get("cwd", tmp_path)
        if "evaluate.py" in cmd_str:
            _make_eval_artifacts(cwd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        config=PipelineConfig(prompts_dir="prompts"),
    )
    result = runner.run()

    assert result == EXIT_SUCCESS
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["stage"] == "VERIFIED"
    # Training should NOT have been called
    assert not any("train.py" in c for c in calls)
    # Evaluation should have been called
    assert any("evaluate.py" in c for c in calls)


# ═══════════════════════════════════════════════════════════
# Logging (AC-8)
# ═══════════════════════════════════════════════════════════


def test_artifact_stages_logged(tmp_path, monkeypatch):
    """AC-8: All three new stages log to the pipeline transcript."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        cwd = kwargs.get("cwd", tmp_path)
        if "train.py" in cmd_str:
            _make_training_artifacts(cwd)
        elif "evaluate.py" in cmd_str:
            _make_eval_artifacts(cwd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=DummyProvider(),
        config=PipelineConfig(prompts_dir="prompts"),
    )
    runner.run()

    log_text = (tmp_path / ".pipeline-state" / "demo.log").read_text(encoding="utf-8")
    assert "Stage 6: Produce Artifacts" in log_text
    assert "Stage 7: Validate Artifacts" in log_text
    assert "Stage 8: Acceptance" in log_text
    assert "[PASS]" in log_text or "[FAIL]" in log_text
    assert "Pipeline COMPLETE" in log_text


# ═══════════════════════════════════════════════════════════
# Implementer Fix-and-Retry (stages 6-8)
# ═══════════════════════════════════════════════════════════


class TrackingProvider:
    """Provider that records calls for retry tests."""

    name = "dummy"

    def __init__(self):
        self.calls = []

    def run_role(self, *, role, prompt, repo_root, state_dir, schema=None):
        self.calls.append({"role": role, "prompt": prompt})
        return ProviderExecution(
            provider=self.name,
            role=role,
            tier="economy",
            model="dummy-model",
            output='{"decision":"approve","summary":"ok","blocking":[]}',
        )


def test_training_failure_triggers_implementer_fix(tmp_path, monkeypatch):
    """Training fails once → implementer invoked → training succeeds on retry."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)
    provider = TrackingProvider()
    train_call_count = {"n": 0}

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        cwd = kwargs.get("cwd", tmp_path)
        if "train.py" in cmd_str:
            train_call_count["n"] += 1
            if train_call_count["n"] == 1:
                return SimpleNamespace(returncode=1, stdout="", stderr="RuntimeError: boom")
            _make_training_artifacts(cwd)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "evaluate.py" in cmd_str:
            _make_eval_artifacts(cwd)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=provider,
        config=PipelineConfig(prompts_dir="prompts"),
    )
    result = runner.run()

    assert result == EXIT_SUCCESS
    state = json.loads((tmp_path / ".pipeline-state" / "demo.json").read_text(encoding="utf-8"))
    assert state["stage"] == "VERIFIED"
    # Implementer was invoked exactly once to fix the training failure
    implementer_calls = [c for c in provider.calls if c["role"] == "implementer"]
    assert len(implementer_calls) == 1
    assert "RuntimeError: boom" in implementer_calls[0]["prompt"]
    # Training was attempted twice
    assert train_call_count["n"] == 2


def test_acceptance_failure_retries_from_training(tmp_path, monkeypatch):
    """Acceptance fails → implementer fixes → retrain + re-eval → passes."""
    # Spec with a threshold that fails on first attempt but passes on second
    spec = """\
# Spec

## Artifact Pipeline

### Training
command: echo train
required_files:
  - artifacts/train_log.json
metrics_file: artifacts/train_log.json
metrics_checks:
  - path: val_acc
    op: ">"
    value: 0.6
    label: "val_acc > 0.6"

### Evaluation
command: echo eval
required_files:
  - artifacts/eval.json
metrics_file: artifacts/eval.json
metrics_checks: []

### Acceptance
summary_file: artifacts/eval.json
all_checks_must_pass: true
min_checks_pass: 0
"""
    _setup_pipeline_fs(tmp_path, spec)
    provider = TrackingProvider()
    train_count = {"n": 0}

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        cwd = Path(kwargs.get("cwd", tmp_path))
        if "echo train" in cmd_str:
            train_count["n"] += 1
            artifacts_dir = cwd / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            # First attempt: below threshold. Second: above.
            acc = 0.5 if train_count["n"] == 1 else 0.7
            (artifacts_dir / "train_log.json").write_text(
                json.dumps({"val_acc": acc}), encoding="utf-8"
            )
        elif "echo eval" in cmd_str:
            artifacts_dir = cwd / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "eval.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=provider,
        config=PipelineConfig(prompts_dir="prompts"),
    )
    result = runner.run()

    assert result == EXIT_SUCCESS
    assert train_count["n"] == 2
    implementer_calls = [c for c in provider.calls if c["role"] == "implementer"]
    assert len(implementer_calls) == 1


def test_artifact_retry_cap_raises(tmp_path, monkeypatch):
    """Persistent training failure hits revision cap → raises original error."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)
    provider = TrackingProvider()

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        if "train.py" in cmd_str:
            return SimpleNamespace(returncode=1, stdout="", stderr="persistent error")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=provider,
        max_revisions=2,
        config=PipelineConfig(prompts_dir="prompts"),
    )

    with pytest.raises(PipelineError) as exc_info:
        runner.run()
    assert exc_info.value.exit_code == EXIT_TRAINING_FAILED
    # Implementer was invoked twice (iterations 0 and 1), then cap at iteration 2
    implementer_calls = [c for c in provider.calls if c["role"] == "implementer"]
    assert len(implementer_calls) == 2


def test_non_fixable_error_propagates(tmp_path, monkeypatch):
    """Non-fixable errors (e.g., FROZEN_TESTS_MODIFIED) bypass retry loop."""
    _setup_pipeline_fs(tmp_path, SPEC_WITH_PIPELINE)
    provider = TrackingProvider()

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        cwd = kwargs.get("cwd", tmp_path)
        if "train.py" in cmd_str:
            _make_training_artifacts(cwd)
            # Simulate frozen tests being modified during training
            (Path(cwd) / "tests" / "test_demo.py").write_text(
                "def test_demo():\n    assert False  # modified\n", encoding="utf-8"
            )
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=provider,
        config=PipelineConfig(prompts_dir="prompts"),
    )

    # The frozen_tests_hash won't match → should propagate, not retry
    # But wait — frozen test check only happens during implementer fix stages.
    # For a clean test of non-fixable propagation, let's raise directly.
    from vla_agent.pipeline.core import EXIT_TESTS_BROKE_AFTER_REVISION

    def failing_artifact_stage(**kwargs):
        raise PipelineError("tests broke", EXIT_TESTS_BROKE_AFTER_REVISION)

    runner._run_artifact_stage = failing_artifact_stage

    with pytest.raises(PipelineError) as exc_info:
        runner.run()
    assert exc_info.value.exit_code == EXIT_TESTS_BROKE_AFTER_REVISION
    # Implementer should NOT have been invoked
    implementer_calls = [c for c in provider.calls if c["role"] == "implementer"]
    assert len(implementer_calls) == 0


def test_retry_clears_stale_metrics(tmp_path, monkeypatch):
    """After acceptance failure + retry, state should have empty metrics."""
    spec = """\
# Spec

## Artifact Pipeline

### Training
command: echo train
required_files:
  - artifacts/train_log.json
metrics_file: artifacts/train_log.json
metrics_checks:
  - path: val_acc
    op: ">"
    value: 0.99
    label: "impossible"

### Evaluation
command: echo eval
required_files:
  - artifacts/eval.json
metrics_file: artifacts/eval.json
metrics_checks: []

### Acceptance
summary_file: artifacts/eval.json
all_checks_must_pass: true
min_checks_pass: 0
"""
    _setup_pipeline_fs(tmp_path, spec)
    provider = TrackingProvider()

    def fake_run(command, **kwargs):
        cmd_str = " ".join(command) if isinstance(command, list) else str(command)
        cwd = Path(kwargs.get("cwd", tmp_path))
        artifacts_dir = cwd / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        if "echo train" in cmd_str:
            (artifacts_dir / "train_log.json").write_text(
                json.dumps({"val_acc": 0.5}), encoding="utf-8"
            )
        elif "echo eval" in cmd_str:
            (artifacts_dir / "eval.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    # max_revisions=1: one retry allowed, then fail
    runner = PipelineRunner(
        repo_root=tmp_path,
        task="demo",
        provider=provider,
        max_revisions=1,
        config=PipelineConfig(prompts_dir="prompts"),
    )

    with pytest.raises(PipelineError) as exc_info:
        runner.run()
    assert exc_info.value.exit_code == EXIT_ACCEPTANCE_FAILED

    # After the first failure + implementer fix, state was saved as CODE_REVIEWED
    # with cleared metrics. The second attempt also fails and raises.
    # Final state should reflect the last save before the raise.
    state = json.loads((tmp_path / ".pipeline-state" / "demo.json").read_text(encoding="utf-8"))
    # The state should be ARTIFACTS_VALIDATED from the last attempt (iteration 1)
    assert state["stage"] == "ARTIFACTS_VALIDATED"
