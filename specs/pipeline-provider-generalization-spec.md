# Pipeline Provider Generalization Spec

## Summary

Generalize the autonomous agentic TDD pipeline so the same pipeline can run on multiple coding runtimes, starting with Claude Code and Codex, while keeping the orchestration logic, state machine, and guardrails provider-agnostic.

The current pipeline is autonomous after the spec is approved by a human, but its implementation is tightly coupled to Claude CLI conventions. This spec defines a refactor that preserves the existing workflow and behavior while isolating provider-specific concerns behind a small adapter layer.

## Motivation

The current implementation in `scripts/run_pipeline.sh` already externalizes most of the real pipeline logic:

- stage transitions
- bounded review loops
- resume from `.pipeline-state/<task-id>.json`
- transcript logging
- frozen-test enforcement
- pytest gate after code revisions

The main issue is not the workflow itself. The issue is that invocation, agent manifests, hooks, and structured review parsing are Claude-specific. That prevents the same pipeline from being run against Codex without forking the whole implementation.

## Goals

- Keep a single autonomous pipeline workflow for all supported runtimes.
- Preserve the current stage machine and guardrails.
- Minimize provider-specific code.
- Support at least two providers:
  - Claude Code
  - Codex
- Keep pipeline roles stable:
  - `test-writer`
  - `implementer`
  - `reviewer`
- Select runtime with a provider flag such as `--provider claude` or `--provider codex`.
- Move correctness-critical behavior out of provider-only hook systems and into the main orchestrator.
- Preserve role-level cost/quality intent across providers via abstract capability tiers rather than provider model names.
- Keep execution aligned with repo rules:
  - UV-only command execution
  - no provider-specific assumptions in core logic
  - no correctness dependency on local Claude-only config or git ownership quirks

## Non-Goals

- Redesign the actual TDD workflow.
- Add new pipeline stages.
- Introduce a generic plugin ecosystem for arbitrary providers beyond what is needed for Claude Code and Codex.
- Add configuration complexity beyond the smallest structure needed to support multiple providers.

## Current Claude-Specific Coupling

The following concerns are Claude-specific today and must be isolated:

1. CLI invocation patterns such as `claude -p --agent ...`.
2. Claude agent manifests under `.claude/agents/`.
3. Claude settings and hooks under `.claude/settings.json`.
4. Claude local permission allowances under `.claude/settings.local.json`.
5. Claude-specific permission flags such as `--permission-mode bypassPermissions`.
6. Claude-specific structured-output assumptions in review response parsing.
7. Claude session environment handling such as unsetting `CLAUDECODE`.
8. Claude-specific model identifiers embedded in agent manifests such as `model: sonnet` and `model: opus`.

These are implementation details of one runtime and must not remain embedded in the core pipeline state machine.

## Required Design

### 1. Provider-Agnostic Core

The following concerns must live in shared base code and must not depend on Claude or Codex:

- stage ordering
- loop caps
- resume logic
- state persistence
- transcript logging
- frozen-test checks
- pytest gates
- validation-stage behavior
- review decision schema
- prompt construction for pipeline roles

The core pipeline must reason in terms of:

- task id
- spec path
- pipeline role
- provider name
- stage
- review decision

The core pipeline must not reason in terms of:

- `claude`
- `codex`
- `sonnet`
- `opus`
- `.claude/...`
- provider-specific hook file paths

### 2. Stable Role Model

The pipeline role names remain:

- `test-writer`
- `implementer`
- `reviewer`

These are pipeline concepts, not provider names.

The runtime selector must therefore use `--provider`.


### 3. Thin Provider Adapters

Each provider adapter must be responsible only for:

- turning a role + prompt into a concrete CLI invocation
- applying provider-specific flags and environment setup
- requesting structured review output where supported
- normalizing raw provider output into the shared review-decision contract

Provider adapters must not implement pipeline logic, revision caps, state transitions, or guardrails.

Provider adapters may satisfy a role either through provider-native subagents or by injecting the role instructions directly into a single-agent prompt flow. The core pipeline must not require every provider to support Claude-style on-disk agent manifests.

### 4. Shared Prompt Sources

Role instructions must be shared across providers as much as possible.

The desired split is:

- shared role prompt content in one provider-neutral location
- provider adapters that inject only the minimal wrapper needed for the target runtime

The shared prompt content should cover:

- responsibilities of each role
- repo rules from `AGENTS.md`
- restrictions such as "do not modify frozen tests"
- required commands such as `uv run pytest`

Provider-specific prompt copies should be avoided unless a runtime has a hard capability difference that cannot be handled in a thin wrapper.

Role prompts should be treated as templates rendered by the core orchestrator. The core should inject the role context before handing the prompt to a provider adapter.

The injected context should include, as applicable:

- task id
- spec path and/or spec content
- relevant repo rules from `AGENTS.md`
- current pipeline stage
- current review iteration
- reviewer blocking feedback for revision stages
- frozen-test constraints
- validation expectations for runnable scripts or import checks

### 5. Orchestrator-Owned Guardrails

Any behavior required for correctness must be enforced by the orchestrator, not by Claude-only hooks.

This includes:

- formatting after edits, if formatting is required by the pipeline
- frozen-test protection
- pytest gates
- final validation gates
- reviewer-stage immutability checks

Provider-native hooks may still exist as convenience layers, but the pipeline must succeed correctly even when those hooks are absent.

If the orchestrator applies formatting automatically, it must do so in a way that does not mutate frozen tests or unrelated files after the test-freeze boundary. Automatic formatting must not become a backdoor that invalidates the frozen-test guarantee.

### 6. Canonical Review Contract

All review stages must produce the same logical decision structure:

```json
{
  "decision": "approve | revise",
  "summary": "string",
  "blocking": ["string"]
}
```

This schema is canonical at the pipeline level.

If a provider supports native JSON schema output, the adapter should use it.

If a provider does not support native schema enforcement in the same way, the adapter must still normalize the result into the same structure before the orchestrator consumes it.

The orchestrator must only depend on the normalized review contract, never on provider-native output shapes.

The orchestrator must validate the final normalized review decision against the canonical schema before acting on it.

If provider-native structured output is malformed or missing, the implementation should attempt a fallback extraction path such as:

- extracting the first valid JSON object from the response
- leniently parsing fenced JSON content
- re-validating the parsed object against the canonical schema

If fallback normalization still cannot produce a valid decision object, the stage must fail clearly rather than proceeding on ambiguous output.

### 7. Role Capability Tiers and Provider-Level Model Selection

Concrete model choice belongs inside provider configuration, not in the pipeline core.

The pipeline should preserve the intent of today's role setup through provider-agnostic capability tiers or profiles.

At minimum, the shared design should support this intent:

- `test-writer`: economy tier
- `implementer`: economy tier
- Stage 4 validation: `implementer` role, economy tier by default
- `reviewer`: premium tier

The exact tier names are flexible, but they must be provider-neutral. Example tier names could be `economy` and `premium`, or `standard` and `high_scrutiny`.

The core pipeline should ask for a role:

- reviewer
- implementer
- test-writer

The provider adapter may map roles and capability tiers to runtime-specific model defaults or reasoning settings.

Provider-specific layers may and should declare explicit concrete model names when that is useful for clarity and reproducibility.

For example, provider-specific config may explicitly encode mappings such as:

- Claude Code:
  - `test-writer` -> `sonnet`
  - `implementer` -> `sonnet`
  - `reviewer` -> `opus`
- Codex:
  - `test-writer` -> provider-specific economy/default model or setting
  - `implementer` -> provider-specific economy/default model or setting
  - `reviewer` -> provider-specific premium/strong-review model or setting

This explicit mapping must live only in provider-specific files or provider adapter code, not in the shared pipeline core or shared prompt sources.

The pipeline core must not encode assumptions like:

- reviewer equals opus
- implementer equals sonnet

Those are provider-specific runtime decisions.

The shared role specs and prompt sources must not hardcode provider model identifiers.

The pipeline logs should record the resolved provider, role, capability tier, and concrete provider-specific model or setting used for each stage.

### 8. Stage-to-Role Assignment

The generalized pipeline must preserve the current stage-to-role mapping:

- Stage 1 test generation: `test-writer`
- Stage 2 test review: `reviewer`
- Stage 3 implementation: `implementer`
- Stage 4 validation: `implementer`
- Stage 5 code review: `reviewer`

Stage 4 validation is intentionally performed by the `implementer` role, not a separate validation-only role. Its responsibility is to exercise the implementation beyond unit tests and fix any issues discovered while preserving frozen tests.

## Execution Requirements

The generalized pipeline must comply with the repo's UV-only rule.

This means:

- the canonical entry point is `uv run python ...`
- Python subprocesses must be invoked through `uv run python`, not bare `python`
- test commands must use `uv run pytest` or `uv run python -m pytest`
- dependency operations must use `uv`
- `pip` must never be used by the pipeline implementation

The current Bash runner uses bare `python` for JSON and state helpers. The generalized implementation must remove that dependency.

## Recommended File Layout

The implementation should move toward a structure similar to:

```text
scripts/
  run_pipeline.py
src/vla_agent/pipeline/
  core.py
  schemas/
    review_decision.json
  prompts/
    test_writer.md
    implementer.md
    reviewer.md
  providers/
    base.py
    claude.py
    codex.py
```

This is a target shape, not a requirement to create every file immediately if a smaller correct diff can achieve the same architecture.

## Recommended Implementation Language

The orchestrator should be migrated from Bash to Python.

Reasons:

- the current script already uses inline Python for JSON parsing and state management
- Python is a better fit for structured state, parsing, subprocess handling, and normalization
- a Python orchestrator avoids additional shell brittleness while staying aligned with the repo's Python-first codebase
- a Python implementation can still be run with `uv run python`

## Provider Interface

The shared adapter interface should be intentionally small.

An interface equivalent to the following is sufficient:

```python
class Provider(Protocol):
    def run_role(self, role: str, prompt: str, schema: dict | None = None) -> AgentResult: ...
    def normalize_review(self, raw: str) -> ReviewDecision: ...
```

The exact names may differ, but the design constraints are:

- one method to execute a role against a provider
- one path to normalize review output into the canonical contract
- no pipeline state logic in providers

## CLI Requirements

The new runner must support:

```text
uv run python scripts/run_pipeline.py <task-id> --provider claude
uv run python scripts/run_pipeline.py <task-id> --provider codex
```

The provider argument must be explicit.

Reasonable defaults are acceptable only if they are simple and unsurprising. If a default provider is used, it must be documented clearly and should not make the runtime ambiguous in logs.

## Logging Requirements

The pipeline must continue writing a full transcript to:

```text
.pipeline-state/<task-id>.log
```

The transcript must include enough information to tell:

- which provider was used
- which stage ran
- which iteration a review loop was on
- the normalized review decision
- any hard gate failure
- whether fallback normalization or fallback guardrail enforcement paths were used

## State and Resume Requirements

The existing resume behavior must be preserved.

State must continue to persist under:

```text
.pipeline-state/<task-id>.json
```

The state file must include the active provider as a required field. Resuming with a different `--provider` than the one recorded in state must fail clearly rather than silently switching runtimes.

Resume semantics must remain the same:

- skip previously completed stages
- resume at the correct loop iteration for review stages
- retain state after `DONE`

Provider selection must be visible in logs and persisted in state as part of the resume contract.

If the implementation stores frozen-test fingerprints or other guardrail metadata in state, that metadata must be treated as part of the resume contract.

## Guardrail Requirements

The generalized pipeline must preserve all existing hard guardrails:

- max 2 test review revisions
- max 2 code review revisions
- frozen tests may not change after tests are frozen
- pytest must pass after each code revision
- reviewer role must remain read-only in effect
- reviewer blocking items must be forwarded into revision prompts

The generalized pipeline must not weaken these guardrails in order to support multiple providers.

The implementation must not rely exclusively on `git diff -- tests/` for frozen-test enforcement. In this repo, git ownership and sandbox-user differences can make git-based checks unreliable in some runtimes. A provider-independent mechanism such as snapshotting or hashing the `tests/` tree at freeze time is acceptable and preferred.

Similarly, reviewer read-only behavior must be enforceable even if a provider lacks native tool restrictions. A practical mechanism is to snapshot or hash the writable tree before and after reviewer stages and fail the pipeline if the reviewer modified files.

The preferred file-integrity mechanism is a deterministic content hash, such as SHA-256, computed over sorted file paths and file contents. Hashing is preferred over heavyweight snapshotting unless the implementation needs snapshots for human-readable diffs.

## Migration Plan

Implementation should proceed in the smallest safe sequence:

1. Port the current `run_pipeline.sh` behavior into a Python runner without changing pipeline semantics.
2. Introduce a `ClaudeProvider` that preserves current behavior as closely as possible.
3. Extract shared role prompts from `.claude/agents/` into provider-neutral prompt files.
4. Add a `CodexProvider` implementing the same provider contract.
5. Update CLI entry to select provider explicitly with `--provider`.
6. Move any correctness-critical formatting or guardrail behavior into the orchestrator.
7. Make provider persistence and provider-match checks mandatory in pipeline state/resume logic.
8. Replace git-only frozen-test enforcement with a provider-independent file-integrity check, or add it alongside git checks as the authoritative gate.
9. Add reviewer-stage file-integrity enforcement so read-only behavior is guaranteed by the pipeline, not just requested in prompts.
10. Keep Claude-specific files only as compatibility or convenience layers if still useful, but do not make the pipeline depend on them.

## Acceptance Criteria

1. A single pipeline entry point can run the same task with either Claude Code or Codex, selected explicitly by provider.
2. The pipeline state machine, loop caps, resume logic, logging, and hard gates remain shared code.
3. Provider-specific logic is isolated to thin adapter modules.
4. Pipeline role prompts are shared across providers and are not duplicated wholesale per provider.
5. The orchestrator consumes one canonical review-decision structure regardless of provider.
6. Frozen-test enforcement and pytest gates remain active and provider-independent.
7. The generalized pipeline does not overload the meaning of `agent`; provider selection uses `--provider`.
8. The implementation does not require a forked pipeline for Codex.
9. The implementation uses UV-only command execution and does not depend on bare `python` or `pip`.
10. Reviewer read-only behavior and frozen-test protection are enforced by provider-independent file-integrity checks, not only by provider-native restrictions or git CLI assumptions.
11. Provider name is persisted in state and a resume attempt with a mismatched provider fails clearly.
12. Role prompts are rendered from shared templates with core-owned context injection.
13. Review decisions are schema-validated after normalization, with a documented fallback parsing path for malformed provider output.
14. Stage 4 validation is explicitly executed by the `implementer` role.
15. Shared role specifications preserve cost/quality intent through provider-neutral capability tiers instead of concrete provider model names.
16. Concrete provider model names remain explicit and visible in provider-specific config or adapter layers, and are logged at runtime per stage.

## Nice-to-Have

- support provider-specific defaults for reviewer and implementer models behind the adapter layer
- keep a compatibility wrapper so existing Claude-only usage can continue temporarily during migration

## Open Questions

- whether a compatibility shell wrapper should remain after the Python runner lands
- whether provider-neutral prompts should live under `src/vla_agent/pipeline/prompts/` or under a top-level non-package directory

## Explicit Rejections

The following approaches are rejected by this spec:

- forking the entire pipeline into separate Claude and Codex implementations
- duplicating all role prompts per provider by default
- keeping correctness-critical behavior inside `.claude/settings.json`
- encoding provider-specific model names directly into the core state machine
- relying exclusively on `git diff` for frozen-test enforcement
