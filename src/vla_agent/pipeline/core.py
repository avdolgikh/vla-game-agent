"""Provider-agnostic orchestration for the agentic TDD pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

from vla_agent.pipeline.providers.base import Provider, ProviderExecution


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "revise"]},
        "summary": {"type": "string"},
        "blocking": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "summary", "blocking"],
    "additionalProperties": False,
}

STAGE_ORDER = {
    "SPEC_APPROVED": 0,
    "TESTS_GENERATED": 1,
    "TESTS_FROZEN": 2,
    "CODE_IMPLEMENTED": 3,
    "CODE_VALIDATED": 4,
    "CODE_REVIEWED": 5,
    "ARTIFACTS_PRODUCED": 6,
    "ARTIFACTS_VALIDATED": 7,
    "VERIFIED": 8,
}

REPO_HASH_TARGETS = [
    ".claude",
    "AGENTS.md",
    "pyproject.toml",
    "scripts",
    "specs",
    "src",
    "tests",
]

TRANSIENT_PATH_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
TRANSIENT_SUFFIXES = {".pyc", ".pyo"}

EXIT_SUCCESS = 0
EXIT_SPEC_NOT_FOUND = 1
EXIT_TEST_REVISION_CAP = 2
EXIT_CODE_REVISION_CAP = 3
EXIT_FROZEN_TESTS_MODIFIED = 4
EXIT_TESTS_BROKE_AFTER_REVISION = 5
EXIT_STATE_PROVIDER_MISMATCH = 6
EXIT_REVIEWER_MODIFIED_FILES = 7
EXIT_INVALID_REVIEW_OUTPUT = 8
EXIT_PROVIDER_EXEC_FAILED = 9
EXIT_STAGE_NO_EFFECT = 10
EXIT_TRAINING_FAILED = 11
EXIT_EVALUATION_FAILED = 12
EXIT_ACCEPTANCE_FAILED = 13
EXIT_ARTIFACT_MISSING = 14


class PipelineError(RuntimeError):
    """Pipeline failure with an explicit exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class ReviewDecision:
    """Normalized reviewer decision."""

    decision: str
    summary: str
    blocking: list[str] = field(default_factory=list)
    fallback_used: bool = False


def _review_requests_missing_inputs(decision: ReviewDecision) -> bool:
    haystack = " ".join([decision.summary, *decision.blocking]).lower()
    phrases = [
        "missing review input",
        "missing review inputs",
        "review packet",
        "provide the spec",
        "provide the review packet",
        "what should i review",
        "where are the artifacts",
        "i don't have",
        "i do not have",
        "task id",
    ]
    return any(phrase in haystack for phrase in phrases)


def _stage_requested_more_input(raw: str) -> bool:
    haystack = raw.lower()
    phrases = [
        "role acknowledged",
        "please point me to",
        "please provide",
        "clarify any particular",
        "what task",
        "what should i review",
        "review packet",
        "share that too",
        "i'm set in the",
        "i?m set in the",
        "i'm ready to act",
        "i?m ready to act",
        "please include",
    ]
    return any(phrase in haystack for phrase in phrases)


@dataclass
class PipelineState:
    """Persisted pipeline state."""

    task: str
    provider: str
    stage: str
    iteration: int = 0
    frozen_tests_hash: str | None = None
    training_metrics: dict | None = None
    evaluation_metrics: dict | None = None


class PipelineLogger:
    """Writes pipeline output to stdout and the transcript log."""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write_stdout(self, message: str) -> None:
        stream = sys.stdout
        line = f"{message}\n"
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            encoding = stream.encoding or "utf-8"
            buffer.write(line.encode(encoding, errors="replace"))
            buffer.flush()
            return
        stream.write(line)

    def log(self, message: str = "") -> None:
        self._write_stdout(message)
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(message)
            handle.write("\n")


class PromptBuilder:
    """Renders shared role prompts with stage-specific context."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.prompts_dir = repo_root / "src" / "vla_agent" / "pipeline" / "prompts"

    def role_prompt(self, role: str) -> str:
        filename = role.replace("-", "_") + ".md"
        prompt_path = self.prompts_dir / filename
        return prompt_path.read_text(encoding="utf-8-sig").strip()

    def render(
        self,
        *,
        role: str,
        task: str,
        spec_path: Path,
        stage_name: str,
        stage_instruction: str,
        iteration: int = 0,
        reviewer_feedback: list[str] | None = None,
    ) -> str:
        feedback = reviewer_feedback or []
        spec_text = spec_path.read_text(encoding="utf-8-sig").strip()
        sections = [
            self.role_prompt(role),
            "",
            "## Pipeline Context",
            f"- Task: {task}",
            f"- Stage: {stage_name}",
            f"- Iteration: {iteration}",
            f"- Spec path: {spec_path.as_posix()}",
            "- Repo rules path: AGENTS.md",
            "- This pipeline is non-interactive. Do not ask the human for more input; use the embedded spec/context and inspect the repository directly.",
            "- Tests must be run with `uv run pytest` or `uv run python -m pytest`.",
        ]
        if feedback:
            sections.extend(["- Reviewer blocking feedback to address:"])
            sections.extend([f"  - {item}" for item in feedback])
        sections.extend(
            [
                "",
                "## Approved Spec",
                spec_text,
                "",
                "## Stage Instructions",
                stage_instruction.strip(),
            ]
        )
        if role == "reviewer":
            sections.extend(
                [
                    "",
                    "## Required Final Response",
                    "Return exactly one raw JSON object matching this shape:",
                    '{"decision":"approve|revise","summary":"string","blocking":["string"]}',
                    "Do not ask questions. Do not add markdown fences. Do not add any prose before or after the JSON object.",
                ]
            )
        return "\n".join(sections).strip() + "\n"


def _json_candidates(raw: str) -> list[str]:
    candidates: list[str] = []
    stripped = raw.strip()
    if stripped:
        candidates.append(stripped)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidates.extend(fenced)
    for start in (match.start() for match in re.finditer(r"\{", raw)):
        depth = 0
        for idx in range(start, len(raw)):
            char = raw[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start : idx + 1])
                    break
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def normalize_review_output(raw: str) -> ReviewDecision:
    """Validate and normalize provider review output."""

    fallback_used = False
    last_error: Exception | None = None
    for candidate in _json_candidates(raw):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            fallback_used = True
            continue

        source = payload.get("structured_output", payload) if isinstance(payload, dict) else payload
        if not isinstance(source, dict):
            fallback_used = True
            continue
        decision = source.get("decision")
        summary = source.get("summary")
        blocking = source.get("blocking", [])
        if decision not in {"approve", "revise"}:
            fallback_used = True
            continue
        if not isinstance(summary, str) or not summary.strip():
            fallback_used = True
            continue
        if blocking is None:
            blocking = []
        if not isinstance(blocking, list) or not all(isinstance(item, str) for item in blocking):
            fallback_used = True
            continue
        return ReviewDecision(
            decision=decision,
            summary=summary,
            blocking=blocking,
            fallback_used=fallback_used,
        )

    detail = f" Unable to normalize review output: {last_error}" if last_error else ""
    raise PipelineError(
        f"FAIL: reviewer output did not match the canonical schema.{detail}",
        EXIT_INVALID_REVIEW_OUTPUT,
    )


def _is_hashable_file(repo_root: Path, path: Path) -> bool:
    relative = path.relative_to(repo_root)
    if any(part in TRANSIENT_PATH_PARTS for part in relative.parts):
        return False
    if path.suffix in TRANSIENT_SUFFIXES:
        return False
    return True


def _iter_hashable_files(repo_root: Path, relative_targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for relative in relative_targets:
        target = repo_root / relative
        if not target.exists():
            continue
        if target.is_file():
            if _is_hashable_file(repo_root, target):
                files.append(target)
            continue
        for child in sorted(path for path in target.rglob("*") if path.is_file()):
            if _is_hashable_file(repo_root, child):
                files.append(child)
    return sorted(files)


def hash_paths(repo_root: Path, relative_targets: list[str]) -> str:
    """Return a deterministic SHA-256 over file paths and content."""

    digest = hashlib.sha256()
    for path in _iter_hashable_files(repo_root, relative_targets):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class MetricsCheck:
    """A single metrics threshold check."""

    path: str
    op: str
    value: float
    label: str


@dataclass
class CheckResult:
    """Result of a single metrics check."""

    label: str
    passed: bool
    actual: float
    threshold: float


@dataclass
class StageConfig:
    """Configuration for a training or evaluation stage."""

    command: str
    required_files: list[str]
    metrics_file: str
    metrics_checks: list[MetricsCheck]


@dataclass
class AcceptanceConfig:
    """Configuration for the acceptance verification stage."""

    summary_file: str
    all_checks_must_pass: bool
    min_checks_pass: int


@dataclass
class ArtifactPipelineConfig:
    """Parsed Artifact Pipeline section from a spec."""

    training: StageConfig | None
    evaluation: StageConfig | None
    acceptance: AcceptanceConfig | None


def navigate_json_path(data: dict, path: str) -> Any:
    """Navigate nested JSON using dot-separated path."""
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Key '{key}' not found at path '{path}'")
        current = current[key]
    return current


def run_metrics_checks(data: dict, checks: list[MetricsCheck]) -> list[CheckResult]:
    """Run comparison checks against metrics data."""
    ops = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }
    results: list[CheckResult] = []
    for check in checks:
        actual = navigate_json_path(data, check.path)
        passed = ops[check.op](actual, check.value)
        results.append(
            CheckResult(label=check.label, passed=passed, actual=actual, threshold=check.value)
        )
    return results


def evaluate_acceptance(checks: list[CheckResult], config: AcceptanceConfig) -> bool:
    """Determine if acceptance criteria are met."""
    if config.all_checks_must_pass:
        return all(c.passed for c in checks)
    return sum(1 for c in checks if c.passed) >= config.min_checks_pass


def _extract_section(text: str, heading: str, level: int = 2) -> str | None:
    """Extract content between a markdown heading and the next same-or-higher-level heading."""
    hashes = "#" * level
    pattern = re.compile(rf"^{hashes} {re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None
    start = match.end()
    next_heading = re.compile(rf"^#{{1,{level}}} ", re.MULTILINE)
    next_match = next_heading.search(text, start)
    if next_match:
        return text[start : next_match.start()]
    return text[start:]


def _parse_stage_config(raw: dict) -> StageConfig:
    """Build a StageConfig from parsed YAML data."""
    return StageConfig(
        command=raw["command"].strip(),
        required_files=raw.get("required_files", []),
        metrics_file=raw.get("metrics_file", ""),
        metrics_checks=[
            MetricsCheck(path=c["path"], op=c["op"], value=c["value"], label=c["label"])
            for c in raw.get("metrics_checks", [])
        ],
    )


def parse_artifact_pipeline(spec_text: str) -> ArtifactPipelineConfig | None:
    """Parse the ## Artifact Pipeline section from spec markdown."""
    section = _extract_section(spec_text, "Artifact Pipeline", level=2)
    if section is None:
        return None

    training = None
    training_text = _extract_section(section, "Training", level=3)
    if training_text:
        training = _parse_stage_config(yaml.safe_load(training_text))

    evaluation = None
    eval_text = _extract_section(section, "Evaluation", level=3)
    if eval_text:
        evaluation = _parse_stage_config(yaml.safe_load(eval_text))

    acceptance = None
    accept_text = _extract_section(section, "Acceptance", level=3)
    if accept_text:
        data = yaml.safe_load(accept_text)
        acceptance = AcceptanceConfig(
            summary_file=data.get("summary_file", ""),
            all_checks_must_pass=data.get("all_checks_must_pass", False),
            min_checks_pass=data.get("min_checks_pass", 0),
        )

    return ArtifactPipelineConfig(training=training, evaluation=evaluation, acceptance=acceptance)


class PipelineRunner:
    """Runs the provider-generalized autonomous pipeline."""

    def __init__(
        self,
        *,
        repo_root: Path,
        task: str,
        provider: Provider,
        max_revisions: int = 4,
    ) -> None:
        self.repo_root = repo_root
        self.task = task
        self.provider = provider
        self.max_revisions = max_revisions
        self.spec_path = repo_root / "specs" / f"{task}-spec.md"
        self.state_dir = repo_root / ".pipeline-state"
        self.state_file = self.state_dir / f"{task}.json"
        self.log_file = self.state_dir / f"{task}.log"
        self.logger = PipelineLogger(self.log_file)
        self.prompts = PromptBuilder(repo_root)

    def _log_stage_header(self, label: str) -> None:
        self.logger.log("")
        self.logger.log("=" * len(label))
        self.logger.log(label)
        self.logger.log("=" * len(label))

    def _load_state(self) -> PipelineState:
        if not self.state_file.exists():
            return PipelineState(task=self.task, provider=self.provider.name, stage="SPEC_APPROVED")

        payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        stage = payload["stage"]
        if stage == "DONE":
            stage = "CODE_REVIEWED"
        state = PipelineState(
            task=payload["task"],
            provider=payload["provider"],
            stage=stage,
            iteration=int(payload.get("iteration", 0)),
            frozen_tests_hash=payload.get("frozen_tests_hash"),
            training_metrics=payload.get("training_metrics"),
            evaluation_metrics=payload.get("evaluation_metrics"),
        )
        if state.provider != self.provider.name:
            raise PipelineError(
                (
                    "FAIL: resume provider mismatch. "
                    f"State recorded provider={state.provider}, requested provider={self.provider.name}."
                ),
                EXIT_STATE_PROVIDER_MISMATCH,
            )
        self.logger.log(
            f"[resume] Continuing from stage={state.stage} iteration={state.iteration} provider={state.provider}"
        )
        return state

    def _save_state(
        self,
        stage: str,
        *,
        iteration: int = 0,
        frozen_tests_hash: str | None = None,
        training_metrics: dict | None = None,
        evaluation_metrics: dict | None = None,
    ) -> None:
        current_hash = frozen_tests_hash
        current_training = training_metrics
        current_evaluation = evaluation_metrics
        if self.state_file.exists():
            previous = json.loads(self.state_file.read_text(encoding="utf-8"))
            if current_hash is None:
                current_hash = previous.get("frozen_tests_hash")
            if current_training is None:
                current_training = previous.get("training_metrics")
            if current_evaluation is None:
                current_evaluation = previous.get("evaluation_metrics")
        state = PipelineState(
            task=self.task,
            provider=self.provider.name,
            stage=stage,
            iteration=iteration,
            frozen_tests_hash=current_hash,
            training_metrics=current_training,
            evaluation_metrics=current_evaluation,
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        details = f" provider={self.provider.name}"
        if state.frozen_tests_hash:
            details += f" frozen_tests_hash={state.frozen_tests_hash}"
        self.logger.log(f"[state] {stage} (iteration={iteration}{details})")

    def _past(self, resume_stage: str, target_stage: str) -> bool:
        return STAGE_ORDER[resume_stage] >= STAGE_ORDER[target_stage]

    def _start_iteration(self, state: PipelineState, loop_entry_stage: str) -> int:
        return state.iteration if state.stage == loop_entry_stage else 0

    def _repo_hash(self) -> str:
        return hash_paths(self.repo_root, REPO_HASH_TARGETS)

    def _tests_hash(self) -> str:
        return hash_paths(self.repo_root, ["tests"])

    def _enforce_reviewer_immutability(self, before_hash: str, stage_label: str) -> None:
        after_hash = self._repo_hash()
        if before_hash != after_hash:
            raise PipelineError(
                f"FAIL: reviewer stage modified repository files during {stage_label}.",
                EXIT_REVIEWER_MODIFIED_FILES,
            )

    def _enforce_test_freeze(self, frozen_tests_hash: str | None) -> None:
        if not frozen_tests_hash:
            raise PipelineError(
                "FAIL: frozen tests hash missing from pipeline state.",
                EXIT_FROZEN_TESTS_MODIFIED,
            )
        current_hash = self._tests_hash()
        if current_hash != frozen_tests_hash:
            raise PipelineError(
                "FAIL: frozen test files were modified after the test-freeze boundary.",
                EXIT_FROZEN_TESTS_MODIFIED,
            )

    def _spec_priority_terms(self) -> list[str]:
        terms = {self.task.lower(), self.task.lower().replace("-", "_")}
        try:
            spec_text = self.spec_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return sorted(term for term in terms if term)
        for raw in re.findall(r"`([^`\n]+)`", spec_text):
            normalized = raw.replace("\\", "/").strip().lower()
            if "/" not in normalized and not normalized.endswith(".py"):
                continue
            terms.add(normalized)
            terms.add(Path(normalized).name.lower())
            terms.add(Path(normalized).stem.lower())
            if normalized.endswith(".py"):
                terms.add(f"test_{Path(normalized).stem.lower()}.py")
        return sorted(term for term in terms if term)

    def _artifact_snapshot(self, relative_targets: list[str]) -> str:
        max_total_chars = 3000
        max_file_chars = 900
        candidate_files = [
            path
            for path in _iter_hashable_files(self.repo_root, relative_targets)
            if path.suffix
            in {
                ".py",
                ".md",
                ".toml",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".sh",
                ".ps1",
            }
        ]
        if not candidate_files:
            return ""

        priority_terms = self._spec_priority_terms()

        def sort_key(path: Path) -> tuple[int, int, int, str]:
            relative = path.relative_to(self.repo_root).as_posix().lower()
            hits = sum(1 for term in priority_terms if term in relative)
            exact = 0 if relative in priority_terms else 1
            return (exact, -hits, len(relative), relative)

        candidate_files = sorted(candidate_files, key=sort_key)

        sections: list[str] = []
        listed_files = [path.relative_to(self.repo_root).as_posix() for path in candidate_files]
        manifest_lines = listed_files[:20]
        if len(listed_files) > 20:
            manifest_lines.append(f"... [{len(listed_files) - 20} more files]")
        manifest = "### Workspace Files\n```text\n" + "\n".join(manifest_lines) + "\n```"
        sections.append(manifest)
        total_chars = len(manifest)
        truncated = False
        for path in candidate_files:
            relative = path.relative_to(self.repo_root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) > max_file_chars:
                text = text[: max_file_chars - 15].rstrip() + "\n... [truncated]"
            section = f"### {relative}\n```text\n{text}\n```"
            if total_chars + len(section) > max_total_chars:
                truncated = True
                break
            sections.append(section)
            total_chars += len(section)
        if truncated:
            sections.append(
                "... [artifact snapshot truncated to stay within Windows command limits]"
            )
        return "\n\n".join(sections)

    def _ensure_tests_stage_effect(
        self,
        *,
        before_hash: str,
        execution: ProviderExecution,
        stage_label: str,
        allow_existing: bool = False,
    ) -> None:
        after_hash = self._tests_hash()
        if after_hash != before_hash:
            return
        if allow_existing and any(_iter_hashable_files(self.repo_root, ["tests"])):
            return
        base_message = f"FAIL: {stage_label} did not modify tests/."
        if stage_label == "Stage 1: Test Generation":
            raise PipelineError(base_message, EXIT_STAGE_NO_EFFECT)
        if "Revision" in stage_label:
            raise PipelineError(
                base_message,
                EXIT_STAGE_NO_EFFECT,
            )
        if _stage_requested_more_input(execution.output):
            raise PipelineError(
                f"FAIL: {stage_label} did not modify tests/ and instead requested more input.",
                EXIT_STAGE_NO_EFFECT,
            )

    def _run_pytest_gate(self, label: str) -> None:
        self.logger.log("")
        self.logger.log(label)
        result = subprocess.run(
            ["uv", "run", "python", "-m", "pytest"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            for line in result.stdout.rstrip().splitlines():
                self.logger.log(line)
        if result.stderr:
            for line in result.stderr.rstrip().splitlines():
                self.logger.log(line)
        if result.returncode != 0:
            raise PipelineError(
                "FAIL: tests broke after code revision.", EXIT_TESTS_BROKE_AFTER_REVISION
            )

    def _run_role(
        self,
        *,
        role: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        stage_label: str,
    ) -> ProviderExecution:
        self.logger.log(
            f"[provider] launching provider={self.provider.name} role={role} stage={stage_label}"
        )
        execution = self.provider.run_role(
            role=role,
            prompt=prompt,
            repo_root=self.repo_root,
            schema=schema,
        )
        self.logger.log(
            f"[provider] provider={execution.provider} role={execution.role} "
            f"tier={execution.tier} model={execution.model}"
        )
        if execution.output.strip():
            for line in execution.output.rstrip().splitlines():
                self.logger.log(line)
        else:
            self.logger.log(f"[provider] {stage_label} produced no terminal output.")
        return execution

    def _run_review_role(
        self,
        *,
        prompt: str,
        stage_label: str,
        before_hash: str,
    ) -> ReviewDecision:
        review_targets = ["tests"] if "Test Review" in stage_label else ["src", "tests", "scripts"]
        artifact_snapshot = self._artifact_snapshot(review_targets)
        review_prompt = (
            prompt
            + "\n\n## Artifact Snapshot\n"
            + (artifact_snapshot if artifact_snapshot else "(No matching text artifacts found.)")
            + "\n"
        )
        execution = self._run_role(
            role="reviewer",
            prompt=review_prompt,
            schema=REVIEW_SCHEMA,
            stage_label=stage_label,
        )
        self._enforce_reviewer_immutability(before_hash, stage_label)
        try:
            decision = normalize_review_output(execution.output)
            if _review_requests_missing_inputs(decision):
                raise PipelineError(
                    "FAIL: reviewer requested missing inputs even though the pipeline supplied context.",
                    EXIT_INVALID_REVIEW_OUTPUT,
                )
            return decision
        except PipelineError as exc:
            if exc.exit_code != EXIT_INVALID_REVIEW_OUTPUT:
                raise
            self.logger.log("[review] invalid output; attempting one repair pass")
            repair_prompt = (
                review_prompt
                + "\n\n## Repair Attempt\n"
                + "Your previous response was invalid because it was not exactly one usable review decision. "
                + "This pipeline is non-interactive: do not ask questions or request more input. "
                + "The approved spec and the current artifacts to review are already embedded above. "
                + "Return exactly one raw JSON object with keys `decision`, `summary`, and `blocking`.\n\n"
                + "Previous invalid response:\n"
                + execution.output.strip()
                + "\n"
            )
            repair_execution = self._run_role(
                role="reviewer",
                prompt=repair_prompt,
                schema=REVIEW_SCHEMA,
                stage_label=f"{stage_label} Output Repair",
            )
            self._enforce_reviewer_immutability(before_hash, f"{stage_label} Output Repair")
            decision = normalize_review_output(repair_execution.output)
            if _review_requests_missing_inputs(decision):
                raise PipelineError(
                    "FAIL: reviewer requested missing inputs even after a self-contained repair prompt.",
                    EXIT_INVALID_REVIEW_OUTPUT,
                )
            return decision

    def _run_implementer_fix(
        self,
        *,
        error: PipelineError,
        frozen_tests_hash: str | None,
        iteration: int,
    ) -> None:
        """Invoke the implementer to fix an artifact pipeline failure."""
        self._log_stage_header(f"Stage 6b: Artifact Fix (iter {iteration})")
        error_context = str(error)
        fix_prompt = self.prompts.render(
            role="implementer",
            task=self.task,
            spec_path=self.spec_path,
            stage_name="Stage 6b: Artifact Fix",
            stage_instruction=(
                "The artifact pipeline failed. Diagnose the issue and fix the production code "
                "or training/evaluation scripts to resolve the error below. "
                "Do not modify frozen tests. "
                "Run `uv run python -m pytest` after your fix to ensure tests still pass.\n\n"
                "## Error Details\n\n"
                f"{error_context}"
            ),
            iteration=iteration,
            reviewer_feedback=[error_context],
        )
        self._run_role(
            role="implementer",
            prompt=fix_prompt,
            stage_label="Stage 6b: Artifact Fix",
        )
        self._enforce_test_freeze(frozen_tests_hash)
        self._run_pytest_gate("Gate: pytest after artifact fix")

    def _run_artifact_stage(
        self,
        *,
        config: StageConfig,
        stage_label: str,
        error_exit_code: int,
    ) -> list[CheckResult]:
        """Run an artifact stage: execute command, verify files, check metrics."""
        self._log_stage_header(stage_label)
        self.logger.log(f"[artifact] Running: {config.command}")
        env = os.environ.copy()
        src_dir = str(self.repo_root / "src")
        env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
        # Strip virtualenv so the spec command's Python uses its own packages,
        # not the pipeline orchestrator's venv (which may lack CUDA, etc.).
        env.pop("VIRTUAL_ENV", None)
        env.pop("PYTHONHOME", None)
        venv_prefix = os.path.normcase(str((self.repo_root / ".venv").resolve()))
        path_parts = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(
            p for p in path_parts if not os.path.normcase(p).startswith(venv_prefix)
        )
        result = subprocess.run(
            config.command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
            shell=True,
            env=env,
        )
        if result.stdout:
            for line in result.stdout.rstrip().splitlines():
                self.logger.log(line)
        if result.stderr:
            for line in result.stderr.rstrip().splitlines():
                self.logger.log(line)
        if result.returncode != 0:
            output_tail = ""
            if result.stderr:
                output_tail = "\n".join(result.stderr.rstrip().splitlines()[-20:])
            elif result.stdout:
                output_tail = "\n".join(result.stdout.rstrip().splitlines()[-20:])
            detail = (
                f"\nCommand: {config.command}\nOutput (last 20 lines):\n{output_tail}"
                if output_tail
                else ""
            )
            raise PipelineError(
                f"FAIL: {stage_label} command exited with code {result.returncode}.{detail}",
                error_exit_code,
            )
        for rel_path in config.required_files:
            full_path = self.repo_root / rel_path
            if not full_path.exists() or full_path.stat().st_size == 0:
                raise PipelineError(
                    f"FAIL: required artifact missing or empty: {rel_path} ({stage_label})\n"
                    f"Command: {config.command}",
                    EXIT_ARTIFACT_MISSING,
                )
        checks: list[CheckResult] = []
        if config.metrics_file and config.metrics_checks:
            metrics_path = self.repo_root / config.metrics_file
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            checks = run_metrics_checks(metrics_data, config.metrics_checks)
            for check in checks:
                status = "PASS" if check.passed else "WARN"
                self.logger.log(f"[{status}] {check.label} ({check.actual} vs {check.threshold})")
        return checks

    def run(self) -> int:
        if not self.spec_path.exists():
            raise PipelineError(
                f"ERROR: Spec not found: {self.spec_path.as_posix()}", EXIT_SPEC_NOT_FOUND
            )

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger.log("")
        self.logger.log(f"=== Pipeline: {self.task} ===")
        self.logger.log(f"=== Spec: {self.spec_path.as_posix()} ===")
        self.logger.log(f"=== Provider: {self.provider.name} ===")

        state = self._load_state()

        if not self._past(state.stage, "TESTS_GENERATED"):
            self._log_stage_header("Stage 1: Test Generation")
            before_tests_hash = self._tests_hash()
            prompt = self.prompts.render(
                role="test-writer",
                task=self.task,
                spec_path=self.spec_path,
                stage_name="Stage 1: Test Generation",
                stage_instruction=(
                    "Read the approved spec and AGENTS.md. Write tests for this task covering all "
                    "acceptance criteria in tests/. Confirm the tests are red with `uv run python -m pytest`. "
                    "Do not write production code."
                ),
            )
            execution = self._run_role(
                role="test-writer",
                prompt=prompt,
                stage_label="Stage 1: Test Generation",
            )
            self._ensure_tests_stage_effect(
                before_hash=before_tests_hash,
                execution=execution,
                stage_label="Stage 1: Test Generation",
                allow_existing=True,
            )
            self._save_state("TESTS_GENERATED")
            state = self._load_state()

        if not self._past(state.stage, "TESTS_FROZEN"):
            start = self._start_iteration(state, "TESTS_GENERATED")
            for iteration in range(start, self.max_revisions + 1):
                self._log_stage_header(f"Stage 2: Test Review (iter {iteration})")
                before_hash = self._repo_hash()
                prompt = self.prompts.render(
                    role="reviewer",
                    task=self.task,
                    spec_path=self.spec_path,
                    stage_name="Stage 2: Test Review",
                    stage_instruction=(
                        "Review only the test files relevant to the approved spec for this task. Ignore unrelated workspace changes outside the spec scope. "
                        "Return only the canonical review decision JSON."
                    ),
                    iteration=iteration,
                )
                decision = self._run_review_role(
                    prompt=prompt,
                    stage_label="Stage 2: Test Review",
                    before_hash=before_hash,
                )
                if decision.fallback_used:
                    self.logger.log("[review] fallback normalization path used")
                self.logger.log(f"Decision: {decision.decision}")
                if decision.decision == "approve":
                    frozen_tests_hash = self._tests_hash()
                    self._save_state("TESTS_FROZEN", frozen_tests_hash=frozen_tests_hash)
                    break
                if iteration == self.max_revisions:
                    raise PipelineError(
                        f"FAIL: test revision cap reached after {self.max_revisions} iterations.",
                        EXIT_TEST_REVISION_CAP,
                    )
                self._log_stage_header(f"Stage 2b: Test Revision (iter {iteration})")
                before_tests_hash = self._tests_hash()
                revise_prompt = self.prompts.render(
                    role="test-writer",
                    task=self.task,
                    spec_path=self.spec_path,
                    stage_name="Stage 2b: Test Revision",
                    stage_instruction=(
                        "Revise the test suite to address the reviewer blocking feedback. "
                        "Do not touch production code. Re-run `uv run python -m pytest` after revisions."
                    ),
                    iteration=iteration,
                    reviewer_feedback=decision.blocking,
                )
                execution = self._run_role(
                    role="test-writer",
                    prompt=revise_prompt,
                    stage_label="Stage 2b: Test Revision",
                )
                self._ensure_tests_stage_effect(
                    before_hash=before_tests_hash,
                    execution=execution,
                    stage_label="Stage 2b: Test Revision",
                )
                self._save_state("TESTS_GENERATED", iteration=iteration + 1)
            self.logger.log("")
            self.logger.log(">>> Tests frozen <<<")
            state = self._load_state()

        frozen_tests_hash = state.frozen_tests_hash

        if not self._past(state.stage, "CODE_IMPLEMENTED"):
            self._log_stage_header("Stage 3: Implementation")
            prompt = self.prompts.render(
                role="implementer",
                task=self.task,
                spec_path=self.spec_path,
                stage_name="Stage 3: Implementation",
                stage_instruction=(
                    "Read the approved spec and frozen tests. Implement the minimal production code "
                    "needed to make the tests pass. Do not modify frozen tests. Run "
                    "`uv run python -m pytest` to verify."
                ),
            )
            self._run_role(
                role="implementer",
                prompt=prompt,
                stage_label="Stage 3: Implementation",
            )
            self._enforce_test_freeze(frozen_tests_hash)
            self._save_state("CODE_IMPLEMENTED", frozen_tests_hash=frozen_tests_hash)
            state = self._load_state()

        if not self._past(state.stage, "CODE_VALIDATED"):
            self._log_stage_header("Stage 4: Validation")
            prompt = self.prompts.render(
                role="implementer",
                task=self.task,
                spec_path=self.spec_path,
                stage_name="Stage 4: Validation",
                stage_instruction=(
                    "Validate the implementation end-to-end. If the spec defines runnable scripts or "
                    "CLI commands, run them with minimal arguments. If it does not, exercise the main "
                    "code via `uv run python -c ...`. Fix issues you find. Do not modify frozen tests. "
                    "Run `uv run python -m pytest` at the end."
                ),
            )
            self._run_role(
                role="implementer",
                prompt=prompt,
                stage_label="Stage 4: Validation",
            )
            self._enforce_test_freeze(frozen_tests_hash)
            self._save_state("CODE_VALIDATED", frozen_tests_hash=frozen_tests_hash)
            state = self._load_state()

        if not self._past(state.stage, "CODE_REVIEWED"):
            start = self._start_iteration(state, "CODE_VALIDATED")
            for iteration in range(start, self.max_revisions + 1):
                self._log_stage_header(f"Stage 5: Code Review (iter {iteration})")
                before_hash = self._repo_hash()
                prompt = self.prompts.render(
                    role="reviewer",
                    task=self.task,
                    spec_path=self.spec_path,
                    stage_name="Stage 5: Code Review",
                    stage_instruction=(
                        "Review only the implementation and frozen tests relevant to the approved spec for this task. Ignore unrelated workspace changes outside the spec scope. "
                        "Observe test status with `uv run python -m pytest` and return only the "
                        "canonical review decision JSON."
                    ),
                    iteration=iteration,
                )
                decision = self._run_review_role(
                    prompt=prompt,
                    stage_label="Stage 5: Code Review",
                    before_hash=before_hash,
                )
                if decision.fallback_used:
                    self.logger.log("[review] fallback normalization path used")
                self.logger.log(f"Decision: {decision.decision}")
                if decision.decision == "approve":
                    self._save_state("CODE_REVIEWED", frozen_tests_hash=frozen_tests_hash)
                    break
                if iteration == self.max_revisions:
                    raise PipelineError(
                        f"FAIL: code revision cap reached after {self.max_revisions} iterations.",
                        EXIT_CODE_REVISION_CAP,
                    )
                self._log_stage_header(f"Stage 5b: Code Revision (iter {iteration})")
                revise_prompt = self.prompts.render(
                    role="implementer",
                    task=self.task,
                    spec_path=self.spec_path,
                    stage_name="Stage 5b: Code Revision",
                    stage_instruction=(
                        "Revise the implementation to address the reviewer blocking feedback. "
                        "Do not modify frozen tests. Re-run `uv run python -m pytest` after revisions."
                    ),
                    iteration=iteration,
                    reviewer_feedback=decision.blocking,
                )
                self._run_role(
                    role="implementer",
                    prompt=revise_prompt,
                    stage_label="Stage 5b: Code Revision",
                )
                self._enforce_test_freeze(frozen_tests_hash)
                self._run_pytest_gate("Gate: pytest after code revision")
                self._save_state(
                    "CODE_VALIDATED",
                    iteration=iteration + 1,
                    frozen_tests_hash=frozen_tests_hash,
                )
            state = self._load_state()

        # ── Stages 6-8: Artifact Pipeline ──────────────────────
        spec_text = self.spec_path.read_text(encoding="utf-8-sig")
        artifact_config = parse_artifact_pipeline(spec_text)

        if artifact_config is None:
            if not self._past(state.stage, "VERIFIED"):
                self._save_state("VERIFIED")
        elif not self._past(state.stage, "VERIFIED"):
            fixable_codes = {
                EXIT_TRAINING_FAILED,
                EXIT_EVALUATION_FAILED,
                EXIT_ACCEPTANCE_FAILED,
                EXIT_ARTIFACT_MISSING,
            }
            start = (
                state.iteration
                if state.stage in ("CODE_REVIEWED", "ARTIFACTS_PRODUCED", "ARTIFACTS_VALIDATED")
                else 0
            )
            for iteration in range(start, self.max_revisions + 1):
                try:
                    if not self._past(state.stage, "ARTIFACTS_PRODUCED"):
                        training_checks = self._run_artifact_stage(
                            config=artifact_config.training,
                            stage_label="Stage 6: Produce Artifacts",
                            error_exit_code=EXIT_TRAINING_FAILED,
                        )
                        self._save_state(
                            "ARTIFACTS_PRODUCED",
                            iteration=iteration,
                            training_metrics={"checks": [asdict(c) for c in training_checks]},
                        )
                        state = self._load_state()

                    if not self._past(state.stage, "ARTIFACTS_VALIDATED"):
                        eval_checks = self._run_artifact_stage(
                            config=artifact_config.evaluation,
                            stage_label="Stage 7: Validate Artifacts",
                            error_exit_code=EXIT_EVALUATION_FAILED,
                        )
                        self._save_state(
                            "ARTIFACTS_VALIDATED",
                            iteration=iteration,
                            evaluation_metrics={"checks": [asdict(c) for c in eval_checks]},
                        )
                        state = self._load_state()

                    if not self._past(state.stage, "VERIFIED"):
                        self._log_stage_header("Stage 8: Acceptance")
                        all_checks: list[CheckResult] = []
                        for raw in (state.training_metrics or {}).get("checks", []):
                            all_checks.append(CheckResult(**raw))
                        for raw in (state.evaluation_metrics or {}).get("checks", []):
                            all_checks.append(CheckResult(**raw))

                        for check in all_checks:
                            status = "PASS" if check.passed else "FAIL"
                            self.logger.log(
                                f"[{status}] {check.label} ({check.actual} vs {check.threshold})"
                            )

                        accepted = evaluate_acceptance(all_checks, artifact_config.acceptance)
                        passed = sum(1 for c in all_checks if c.passed)
                        total = len(all_checks)
                        verdict = "VERIFIED" if accepted else "FAILED"
                        self.logger.log(
                            f"Result: {passed}/{total} checks passed. Pipeline {verdict}."
                        )

                        if not accepted:
                            raise PipelineError(
                                f"FAIL: acceptance verification failed "
                                f"({passed}/{total} checks passed).",
                                EXIT_ACCEPTANCE_FAILED,
                            )
                        self._save_state("VERIFIED")

                    break  # All stages passed

                except PipelineError as exc:
                    if exc.exit_code not in fixable_codes:
                        raise
                    if iteration == self.max_revisions:
                        raise
                    self._run_implementer_fix(
                        error=exc,
                        frozen_tests_hash=frozen_tests_hash,
                        iteration=iteration,
                    )
                    self._save_state(
                        "CODE_REVIEWED",
                        iteration=iteration + 1,
                        frozen_tests_hash=frozen_tests_hash,
                        training_metrics={},
                        evaluation_metrics={},
                    )
                    state = self._load_state()

        self.logger.log("")
        self.logger.log(f"=== Pipeline COMPLETE: {self.task} ===")
        return EXIT_SUCCESS


def run_from_cli(task: str, provider: Provider, repo_root: Path, max_revisions: int = 4) -> int:
    """CLI entry point helper."""

    runner = PipelineRunner(
        repo_root=repo_root,
        task=task,
        provider=provider,
        max_revisions=max_revisions,
    )
    try:
        return runner.run()
    except PipelineError as exc:
        runner.logger.log(str(exc))
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive fallback
        runner.logger.log(f"FAIL: unexpected pipeline error: {exc}")
        return EXIT_PROVIDER_EXEC_FAILED


def main(argv: list[str] | None = None) -> int:
    """Support module execution for debugging."""

    raise SystemExit("Use scripts/run_pipeline.py instead of module execution.")


if __name__ == "__main__":
    sys.exit(main())
