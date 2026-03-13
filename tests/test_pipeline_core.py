"""Unit tests for provider-agnostic pipeline core helpers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_agent.pipeline.core import (
    EXIT_FROZEN_TESTS_MODIFIED,
    EXIT_INVALID_REVIEW_OUTPUT,
    EXIT_REVIEWER_MODIFIED_FILES,
    EXIT_STAGE_NO_EFFECT,
    EXIT_STATE_PROVIDER_MISMATCH,
    EXIT_TESTS_BROKE_AFTER_REVISION,
    PipelineError,
    PipelineRunner,
    PromptBuilder,
    hash_paths,
    normalize_review_output,
)
from vla_agent.pipeline.providers.base import ProviderExecution


class DummyProvider:
    """Minimal provider used for unit tests."""

    name = "dummy"

    def run_role(self, *, role, prompt, repo_root, schema=None):  # noqa: ANN001
        return ProviderExecution(
            provider=self.name,
            role=role,
            tier="economy",
            model="dummy-model",
            output='{"decision":"approve","summary":"ok","blocking":[]}',
        )


def test_review_schema_matches_codex_strict_requirements():
    from vla_agent.pipeline.core import REVIEW_SCHEMA

    assert REVIEW_SCHEMA["additionalProperties"] is False
    assert REVIEW_SCHEMA["required"] == ["decision", "summary", "blocking"]


def test_normalize_review_output_accepts_structured_output():
    raw = json.dumps(
        {
            "structured_output": {
                "decision": "approve",
                "summary": "Looks good",
                "blocking": [],
            }
        }
    )
    decision = normalize_review_output(raw)
    assert decision.decision == "approve"
    assert decision.summary == "Looks good"
    assert decision.blocking == []


def test_normalize_review_output_uses_fallback_for_fenced_json():
    raw = """
    Reviewer notes:
    ```json
    {"decision":"revise","summary":"Needs work","blocking":["missing test"]}
    ```
    """
    decision = normalize_review_output(raw)
    assert decision.decision == "revise"
    assert decision.blocking == ["missing test"]
    assert decision.fallback_used is True


def test_normalize_review_output_rejects_invalid_payload():
    with pytest.raises(PipelineError) as exc_info:
        normalize_review_output("not json at all")
    assert exc_info.value.exit_code == EXIT_INVALID_REVIEW_OUTPUT


def test_hash_paths_is_deterministic(tmp_path: Path):
    root = tmp_path
    (root / "tests").mkdir()
    (root / "tests" / "b.txt").write_text("beta", encoding="utf-8")
    (root / "tests" / "a.txt").write_text("alpha", encoding="utf-8")
    first = hash_paths(root, ["tests"])
    second = hash_paths(root, ["tests"])
    assert first == second


def test_prompt_builder_includes_context(tmp_path: Path):
    prompts_dir = tmp_path / "src" / "vla_agent" / "pipeline" / "prompts"
    prompts_dir.mkdir(parents=True)
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    spec_path = specs_dir / "demo-task-spec.md"
    spec_path.write_text("# Demo spec\n\nAcceptance criteria here.", encoding="utf-8")
    (prompts_dir / "implementer.md").write_text("Base implementer prompt.", encoding="utf-8")
    builder = PromptBuilder(tmp_path)
    prompt = builder.render(
        role="implementer",
        task="demo-task",
        spec_path=spec_path,
        stage_name="Stage 3: Implementation",
        stage_instruction="Implement the code.",
        iteration=1,
        reviewer_feedback=["fix x", "fix y"],
    )
    assert "Base implementer prompt." in prompt
    assert "Task: demo-task" in prompt
    assert "Stage: Stage 3: Implementation" in prompt
    assert "fix x" in prompt
    assert "This pipeline is non-interactive." in prompt
    assert "Implement the code." in prompt
    assert "# Demo spec" in prompt


def test_prompt_builder_requires_raw_json_for_reviewer(tmp_path: Path):
    prompts_dir = tmp_path / "src" / "vla_agent" / "pipeline" / "prompts"
    prompts_dir.mkdir(parents=True)
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    spec_path = specs_dir / "demo-task-spec.md"
    spec_path.write_text("# Demo spec", encoding="utf-8")
    (prompts_dir / "reviewer.md").write_text("Base reviewer prompt.", encoding="utf-8")
    builder = PromptBuilder(tmp_path)
    prompt = builder.render(
        role="reviewer",
        task="demo-task",
        spec_path=spec_path,
        stage_name="Stage 2: Test Review",
        stage_instruction="Review the tests.",
    )
    assert "Base reviewer prompt." in prompt
    assert "## Required Final Response" in prompt
    assert '"decision":"approve|revise"' in prompt
    assert "Do not ask questions." in prompt


def test_runner_requires_matching_provider_in_state(tmp_path: Path):
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "demo-spec.md").write_text("# demo", encoding="utf-8")
    state_dir = tmp_path / ".pipeline-state"
    state_dir.mkdir()
    (state_dir / "demo.json").write_text(
        json.dumps(
            {
                "task": "demo",
                "provider": "claude",
                "stage": "SPEC_APPROVED",
                "iteration": 0,
            }
        ),
        encoding="utf-8",
    )
    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=DummyProvider())
    with pytest.raises(PipelineError) as exc_info:
        runner._load_state()
    assert exc_info.value.exit_code == EXIT_STATE_PROVIDER_MISMATCH


class RepairingProvider:
    """Provider stub that needs one reviewer output repair pass."""

    name = "dummy"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._outputs = iter(
            [
                ProviderExecution(
                    provider=self.name,
                    role="test-writer",
                    tier="economy",
                    model="dummy-model",
                    output="tests written",
                ),
                ProviderExecution(
                    provider=self.name,
                    role="reviewer",
                    tier="premium",
                    model="dummy-model",
                    output="Please send the review packet first.",
                ),
                ProviderExecution(
                    provider=self.name,
                    role="reviewer",
                    tier="premium",
                    model="dummy-model",
                    output='{"decision":"approve","summary":"tests look good","blocking":[]}',
                ),
                ProviderExecution(
                    provider=self.name,
                    role="implementer",
                    tier="economy",
                    model="dummy-model",
                    output="implemented",
                ),
                ProviderExecution(
                    provider=self.name,
                    role="implementer",
                    tier="economy",
                    model="dummy-model",
                    output="validated",
                ),
                ProviderExecution(
                    provider=self.name,
                    role="reviewer",
                    tier="premium",
                    model="dummy-model",
                    output='{"decision":"approve","summary":"code looks good","blocking":[]}',
                ),
            ]
        )

    def run_role(self, *, role, prompt, repo_root, schema=None):  # noqa: ANN001
        self.calls.append((role, prompt))
        return next(self._outputs)


def test_runner_repairs_invalid_reviewer_output_once(tmp_path: Path):
    (tmp_path / "specs").mkdir()
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "specs" / "demo-spec.md").write_text("# demo spec", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# repo rules", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts" / "test_writer.md").write_text(
        "test writer", encoding="utf-8"
    )
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts" / "implementer.md").write_text(
        "implementer", encoding="utf-8"
    )
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts" / "reviewer.md").write_text(
        "reviewer", encoding="utf-8"
    )
    provider = RepairingProvider()
    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=provider)

    assert runner.run() == 0
    log_text = (tmp_path / ".pipeline-state" / "demo.log").read_text(encoding="utf-8")
    assert "[review] invalid output; attempting one repair pass" in log_text
    assert provider.calls[2][0] == "reviewer"
    assert "## Repair Attempt" in provider.calls[2][1]
    assert "## Artifact Snapshot" in provider.calls[2][1]
    assert "tests/test_demo.py" in provider.calls[2][1]


def test_prompt_builder_strips_utf8_bom(tmp_path: Path):
    prompts_dir = tmp_path / "src" / "vla_agent" / "pipeline" / "prompts"
    prompts_dir.mkdir(parents=True)
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    spec_path = specs_dir / "demo-task-spec.md"
    spec_path.write_bytes("\ufeff# Demo spec".encode("utf-8"))
    (prompts_dir / "reviewer.md").write_bytes("\ufeffBase reviewer prompt.".encode("utf-8"))
    builder = PromptBuilder(tmp_path)
    prompt = builder.render(
        role="reviewer",
        task="demo-task",
        spec_path=spec_path,
        stage_name="Stage 2: Test Review",
        stage_instruction="Review the tests.",
    )
    assert "\ufeff" not in prompt
    assert prompt.startswith("Base reviewer prompt.")
    assert "# Demo spec" in prompt


def test_artifact_snapshot_is_capped(tmp_path: Path):
    (tmp_path / "src").mkdir()
    large_text = "x" * 20000
    (tmp_path / "src" / "large.py").write_text(large_text, encoding="utf-8")
    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=DummyProvider())
    snapshot = runner._artifact_snapshot(["src"])
    assert len(snapshot) < 3500
    assert "### Workspace Files" in snapshot
    assert "... [truncated]" in snapshot


def test_runner_fails_when_test_generation_requests_more_input_without_writing_tests(
    tmp_path: Path,
):
    (tmp_path / "specs").mkdir()
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "specs" / "demo-spec.md").write_text("# demo spec", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# repo rules", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts" / "test_writer.md").write_text(
        "test writer", encoding="utf-8"
    )
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts" / "implementer.md").write_text(
        "implementer", encoding="utf-8"
    )
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts" / "reviewer.md").write_text(
        "reviewer", encoding="utf-8"
    )

    class NoopTestWriterProvider:
        name = "dummy"

        def run_role(self, *, role, prompt, repo_root, schema=None):  # noqa: ANN001
            return ProviderExecution(
                provider="dummy",
                role=role,
                tier="economy",
                model="dummy-model",
                output=(
                    "Role Acknowledged\n\n"
                    "Please point me to the approved spec or task you want tests written for."
                ),
            )

    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=NoopTestWriterProvider())

    with pytest.raises(PipelineError) as exc_info:
        runner.run()
    assert exc_info.value.exit_code == EXIT_STAGE_NO_EFFECT
    assert "did not modify tests/" in str(exc_info.value)


def test_review_requests_missing_inputs_detects_placeholder_feedback():
    from vla_agent.pipeline.core import ReviewDecision, _review_requests_missing_inputs

    decision = ReviewDecision(
        decision="revise",
        summary="I do not have the review packet yet.",
        blocking=["Missing review inputs: spec and files under review."],
    )
    assert _review_requests_missing_inputs(decision) is True


def test_runner_includes_artifact_snapshot_in_initial_review_prompt(tmp_path: Path):
    (tmp_path / "specs").mkdir()
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "specs" / "demo-spec.md").write_text("# demo spec", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# repo rules", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "src" / "vla_agent" / "pipeline" / "prompts" / "reviewer.md").write_text(
        "reviewer", encoding="utf-8"
    )

    class SingleReviewProvider:
        name = "dummy"

        def __init__(self) -> None:
            self.prompt = ""

        def run_role(self, *, role, prompt, repo_root, schema=None):  # noqa: ANN001
            self.prompt = prompt
            return ProviderExecution(
                provider="dummy",
                role="reviewer",
                tier="premium",
                model="dummy-model",
                output='{"decision":"approve","summary":"ok","blocking":[]}',
            )

    provider = SingleReviewProvider()
    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=provider)
    decision = runner._run_review_role(
        prompt="Base review prompt",
        stage_label="Stage 2: Test Review",
        before_hash=runner._repo_hash(),
    )
    assert decision.decision == "approve"
    assert "## Artifact Snapshot" in provider.prompt
    assert "tests/test_demo.py" in provider.prompt


def test_enforce_test_freeze_detects_modified_tests(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_demo.py"
    test_file.write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "demo-spec.md").write_text("# demo spec\n", encoding="utf-8")

    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=DummyProvider())
    frozen_hash = runner._tests_hash()
    test_file.write_text("def test_demo():\n    assert False\n", encoding="utf-8")

    with pytest.raises(PipelineError) as exc_info:
        runner._enforce_test_freeze(frozen_hash)
    assert exc_info.value.exit_code == EXIT_FROZEN_TESTS_MODIFIED


def test_enforce_reviewer_immutability_detects_repo_changes(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("# repo rules", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "placeholder.py").write_text("print('hi')\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    spec_path = specs_dir / "demo-spec.md"
    spec_path.write_text("# demo spec\n", encoding="utf-8")

    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=DummyProvider())
    before_hash = runner._repo_hash()
    spec_path.write_text("# demo spec updated\n", encoding="utf-8")

    with pytest.raises(PipelineError) as exc_info:
        runner._enforce_reviewer_immutability(before_hash, "Stage 5: Code Review")
    assert exc_info.value.exit_code == EXIT_REVIEWER_MODIFIED_FILES


def test_run_pytest_gate_invokes_uv_run_python_module(tmp_path: Path, monkeypatch):
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "demo-spec.md").write_text("# demo spec\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# repo rules", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text(
        "def test_demo():\n    assert True\n", encoding="utf-8"
    )

    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=DummyProvider())
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)
    runner._run_pytest_gate("Gate: pytest after code revision")

    assert captured["command"] == ["uv", "run", "python", "-m", "pytest"]
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["check"] is False


def test_run_pytest_gate_fails_on_nonzero_exit(tmp_path: Path, monkeypatch):
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "demo-spec.md").write_text("# demo spec\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# repo rules", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    runner = PipelineRunner(repo_root=tmp_path, task="demo", provider=DummyProvider())

    def fake_run(command, **kwargs):  # noqa: ANN001
        return SimpleNamespace(returncode=1, stdout="fail", stderr="broken")

    monkeypatch.setattr("vla_agent.pipeline.core.subprocess.run", fake_run)

    with pytest.raises(PipelineError) as exc_info:
        runner._run_pytest_gate("Gate: pytest after code revision")
    assert exc_info.value.exit_code == EXIT_TESTS_BROKE_AFTER_REVISION
