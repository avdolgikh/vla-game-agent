"""OpenCode CLI provider adapter (Ollama local models).

Local models via Ollama cannot use opencode's tool system (no function calling).
This provider compensates by:
1. Injecting instructions to output files in a structured ``FILE: <path>`` format.
2. Parsing code blocks from stdout and writing them to disk.
3. For the reviewer role, replacing the pipeline's truncated artifact snapshot
   with full file contents so the model can review accurately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


_FILE_OUTPUT_INSTRUCTIONS = """
IMPORTANT: You are running without tool access. You CANNOT create or edit files directly.
Instead, output every file you want to create or modify using this EXACT format:

FILE: <relative/path/to/file.py>
```
<file content here>
```

Rules for file output:
- The FILE: line must appear on its own line immediately before the opening ```.
- Use relative paths from the repository root.
- Output the COMPLETE file content, not diffs or patches.
- You may output multiple FILE blocks.
"""

# Matches FILE: <path> followed by a fenced code block.
_FILE_BLOCK_RE = re.compile(
    r"^FILE:\s*(.+?)\s*$\n```[^\n]*\n(.*?)^```",
    re.MULTILINE | re.DOTALL,
)

# Matches the artifact snapshot section the pipeline injects.
_ARTIFACT_SNAPSHOT_RE = re.compile(
    r"## Artifact Snapshot\n.*",
    re.DOTALL,
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from terminal output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def extract_file_blocks(output: str) -> list[tuple[str, str]]:
    """Extract (path, content) pairs from structured model output."""
    clean = _strip_ansi(output)
    return [(m.group(1).strip(), m.group(2)) for m in _FILE_BLOCK_RE.finditer(clean)]


def _find_task_files(repo_root: Path, dirs: list[str], task_terms: list[str]) -> list[Path]:
    """Find .py files in dirs whose stem matches any task term exactly."""
    results: list[Path] = []
    for d in dirs:
        target = repo_root / d
        if not target.is_dir():
            continue
        for path in sorted(target.rglob("*.py")):
            if not path.is_file():
                continue
            stem = path.stem.lower()
            if stem in task_terms:
                results.append(path)
    return results


def _build_full_snapshot(repo_root: Path, files: list[Path]) -> str:
    """Build an untruncated artifact snapshot from specific files."""
    if not files:
        return "(No matching artifacts found.)"
    sections: list[str] = []
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        sections.append(f"### {rel}\n```python\n{content}\n```")
    return "\n\n".join(sections)


def _extract_task_terms(prompt: str) -> list[str]:
    """Extract task-relevant search terms from the prompt.

    Collects module/file names mentioned in the spec section of the prompt
    and falls back to the task name.
    """
    terms: set[str] = set()
    # Extract file/module names from backtick-quoted paths in the prompt.
    for raw in re.findall(r"`([^`\n]+\.py)`", prompt):
        name = Path(raw).stem.lower()
        terms.add(name)
        if not name.startswith("test_"):
            terms.add(f"test_{name}")
    # Fall back to task name if no file-based terms found.
    if not terms:
        m = re.search(r"- Task:\s*(\S+)", prompt)
        if m:
            terms.add(m.group(1).strip().lower().replace("-", "_"))
    return sorted(terms)


class OpenCodeProvider:
    """Runs pipeline roles through the OpenCode CLI with local Ollama models."""

    name = "opencode"

    def __init__(self) -> None:
        model = os.getenv("OPENCODE_MODEL", "ollama/qwen3.5:latest").strip()
        self.role_configs = {
            "test-writer": RoleConfig(tier="economy", model=model),
            "implementer": RoleConfig(tier="economy", model=model),
            "reviewer": RoleConfig(tier="premium", model=model),
        }
        self.executable = Path(os.getenv("APPDATA", "")) / "npm" / "opencode.cmd"

    def _tail(self, text: str, limit: int = 2000) -> str:
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return "[truncated]\n" + stripped[-limit:]

    def _augment_prompt(self, prompt: str, role: str, repo_root: Path) -> str:
        """Add role-specific instructions to compensate for lack of tool access."""
        if role == "reviewer":
            # Replace the pipeline's truncated artifact snapshot with full content
            # of task-relevant files only.
            task_terms = _extract_task_terms(prompt)
            if task_terms:
                files = _find_task_files(repo_root, ["tests", "src"], task_terms)
                if files:
                    full_snapshot = _build_full_snapshot(repo_root, files)
                    # Replace the truncated snapshot.
                    prompt = _ARTIFACT_SNAPSHOT_RE.sub(
                        "## Artifact Snapshot (COMPLETE - NOT TRUNCATED)\n\n"
                        + full_snapshot
                        + "\n\n"
                        + "IMPORTANT: The file contents above are COMPLETE. "
                        "Do NOT claim they are truncated or missing. "
                        "Review them against the spec and return your decision.\n",
                        prompt,
                    )
            return prompt
        # For test-writer and implementer: add file output instructions.
        return prompt + "\n" + _FILE_OUTPUT_INSTRUCTIONS

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
                "FAIL: OpenCode CLI was not found at the expected Windows npm shim path.",
                EXIT_PROVIDER_EXEC_FAILED,
            )
        config = self.role_configs[role]
        augmented_prompt = self._augment_prompt(prompt, role, repo_root)

        command = [
            str(self.executable),
            "run",
            "-m",
            config.model,
            "--dir",
            str(repo_root),
        ]

        # Feed prompt via stdin to avoid Windows command-line length limits.
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=augmented_prompt,
        )

        if result.returncode != 0:
            details: list[str] = []
            if result.stdout.strip():
                details.append("stdout:\n" + self._tail(result.stdout))
            if result.stderr.strip():
                details.append("stderr:\n" + self._tail(result.stderr))
            suffix = ("\n" + "\n\n".join(details)) if details else ""
            raise PipelineError(
                f"FAIL: OpenCode provider execution failed for role {role} "
                f"(exit {result.returncode}).{suffix}",
                EXIT_PROVIDER_EXEC_FAILED,
            )

        output = result.stdout.strip()

        # Write extracted file blocks to disk (test-writer and implementer).
        if role != "reviewer":
            for rel_path, content in extract_file_blocks(output):
                target = repo_root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

        return ProviderExecution(
            provider=self.name,
            role=role,
            tier=config.tier,
            model=config.model,
            output=_strip_ansi(output),
        )
