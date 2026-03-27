"""Gemini CLI provider adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from vla_agent.pipeline.core import EXIT_PROVIDER_EXEC_FAILED, PipelineError
from vla_agent.pipeline.providers.base import ProviderExecution


@dataclass(frozen=True)
class RoleConfig:
    tier: str
    model: str


class GeminiProvider:
    """Runs pipeline roles through the Gemini CLI."""

    name = "gemini"

    def __init__(self) -> None:
        economy = os.getenv("GEMINI_MODEL_ECONOMY", "gemini-2.5-flash").strip()
        premium = os.getenv("GEMINI_MODEL_PREMIUM", "gemini-2.5-pro").strip()
        self.role_configs = {
            "test-writer": RoleConfig(tier="economy", model=economy),
            "implementer": RoleConfig(tier="economy", model=economy),
            "reviewer": RoleConfig(tier="premium", model=premium),
        }
        self.executable = Path(os.getenv("APPDATA", "")) / "npm" / "gemini.cmd"

    def _tail(self, text: str, limit: int = 2000) -> str:
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return "[truncated]\n" + stripped[-limit:]

    def _command(self, *, role: str) -> list[str]:
        config = self.role_configs[role]
        return [
            str(self.executable),
            "--yolo",
            "-m",
            config.model,
            "-o",
            "json",
        ]

    def _extract_response(self, stdout: str) -> str:
        """Extract the response text from Gemini CLI JSON output."""
        try:
            data = json.loads(stdout)
            return data.get("response", "")
        except (json.JSONDecodeError, TypeError):
            return stdout

    def run_role(
        self,
        *,
        role: str,
        prompt: str,
        repo_root: Path,
        state_dir: Path,
        schema: dict[str, Any] | None = None,
    ) -> ProviderExecution:
        if not self.executable.exists():
            raise PipelineError(
                "FAIL: Gemini CLI was not found at the expected Windows npm shim path.",
                EXIT_PROVIDER_EXEC_FAILED,
            )
        config = self.role_configs[role]
        command = self._command(role=role)
        # Feed prompt via stdin to avoid Windows command-line length limits.
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=prompt,
        )
        if result.returncode != 0:
            details: list[str] = []
            if result.stdout.strip():
                details.append("stdout:\n" + self._tail(result.stdout))
            if result.stderr.strip():
                details.append("stderr:\n" + self._tail(result.stderr))
            suffix = ("\n" + "\n\n".join(details)) if details else ""
            raise PipelineError(
                f"FAIL: Gemini provider execution failed for role {role} (exit {result.returncode}).{suffix}",
                EXIT_PROVIDER_EXEC_FAILED,
            )
        output = self._extract_response(result.stdout).strip()
        return ProviderExecution(
            provider=self.name,
            role=role,
            tier=config.tier,
            model=config.model,
            output=output,
        )
