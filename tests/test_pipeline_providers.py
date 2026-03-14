"""Unit tests for provider adapter command construction."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_agent.pipeline.core import PipelineError
from vla_agent.pipeline.providers.claude import ClaudeProvider
from vla_agent.pipeline.providers.codex import CodexProvider


def test_claude_provider_builds_schema_command():
    provider = ClaudeProvider()
    command = provider._command(
        role="reviewer",
        prompt="Return JSON",
        schema={"type": "object"},
    )
    assert command[:3] == ["claude", "-p", "Return JSON"]
    assert "--model" in command
    assert "opus" in command
    assert "--json-schema" in command


def test_codex_provider_uses_windows_cmd_shim(tmp_path: Path):
    provider = CodexProvider()
    command = provider._command(
        role="implementer",
        repo_root=tmp_path,
        output_path=tmp_path / "out.txt",
        schema_path=tmp_path / "schema.json",
    )
    assert command[0].endswith("codex.cmd")
    assert command[1] == "exec"
    assert "--model" in command
    assert "gpt-5.1-codex" in command
    assert "--output-schema" in command
    assert "--ephemeral" in command
    assert "--skip-git-repo-check" in command
    assert command[command.index("--model") + 1] == "gpt-5.1-codex"
    assert command[command.index("--output-schema") + 1].endswith("schema.json")
    assert command[command.index("--sandbox") + 1] == "danger-full-access"
    assert command[-1] == "-"
    assert "--ask-for-approval" not in command


def test_codex_reviewer_uses_danger_full_access_with_immutability_guard(tmp_path: Path):
    provider = CodexProvider()
    command = provider._command(
        role="reviewer",
        repo_root=tmp_path,
        output_path=tmp_path / "out.txt",
        schema_path=tmp_path / "schema.json",
    )
    assert command[command.index("--sandbox") + 1] == "danger-full-access"


def test_codex_provider_default_role_models_are_explicit():
    provider = CodexProvider()
    assert provider.role_configs["test-writer"].model == "gpt-5.1-codex-mini"
    assert provider.role_configs["implementer"].model == "gpt-5.1-codex"
    assert provider.role_configs["reviewer"].model == "gpt-5.2-codex"


def test_codex_provider_reads_last_message_from_workspace_scratch(tmp_path: Path, monkeypatch):
    provider = CodexProvider()
    provider.executable = tmp_path / "codex.cmd"
    provider.executable.write_text("@echo off\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("final codex answer\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("vla_agent.pipeline.providers.codex.subprocess.run", fake_run)

    result = provider.run_role(
        role="implementer",
        prompt="Implement code",
        repo_root=tmp_path,
    )

    assert result.output == "final codex answer"
    output_path = Path(captured["command"][captured["command"].index("--output-last-message") + 1])
    assert (
        output_path == tmp_path / ".pipeline-state" / "codex" / "implementer" / "last_message.txt"
    )
    assert captured["kwargs"] == {
        "cwd": tmp_path,
        "check": False,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "input": "Implement code",
    }


def test_codex_provider_surfaces_process_diagnostics_on_failure(tmp_path: Path, monkeypatch):
    provider = CodexProvider()
    provider.executable = tmp_path / "codex.cmd"
    provider.executable.write_text("@echo off\n", encoding="utf-8")

    def fake_run(command, **kwargs):  # noqa: ANN001
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text("partial response\n", encoding="utf-8")
        return SimpleNamespace(returncode=1, stdout="provider stdout", stderr="provider stderr")

    monkeypatch.setattr("vla_agent.pipeline.providers.codex.subprocess.run", fake_run)

    with pytest.raises(PipelineError) as exc_info:
        provider.run_role(
            role="reviewer",
            prompt="Review code",
            repo_root=tmp_path,
        )

    message = str(exc_info.value)
    assert "provider stdout" in message
    assert "provider stderr" in message
    assert "partial response" in message


def test_provider_role_tiers_are_explicit():
    claude = ClaudeProvider()
    codex = CodexProvider()
    assert claude.role_configs["implementer"].tier == "economy"
    assert claude.role_configs["reviewer"].tier == "premium"
    assert codex.role_configs["implementer"].tier == "economy"
    assert codex.role_configs["reviewer"].tier == "premium"
