# Spec: Gemini Provider (Local CLI)

## Status

Proposed

## Goal

Add `--provider gemini` so the autonomous pipeline can run through the local Gemini CLI in the same provider-agnostic flow used for Claude and Codex.

## Scope

- Add a Gemini provider adapter under `src/vla_agent/pipeline/providers/`.
- Wire `gemini` into `scripts/run_pipeline.py`.
- Keep the shared pipeline core provider-agnostic; only make shared-core changes if Gemini compatibility cannot be handled inside the adapter.

## Requirements

### CLI integration

- `scripts/run_pipeline.py` must accept `--provider gemini`.
- The Gemini adapter must invoke the local Gemini CLI executable in non-interactive mode from the repository root.
- The adapter must resolve the Gemini CLI executable robustly on Windows instead of assuming `gemini` is always available on `PATH`.
- If the Gemini CLI executable is unavailable, the provider must fail with a clear pipeline error.

### Role behavior

- The Gemini provider must support the existing pipeline roles: `test-writer`, `implementer`, and `reviewer`.
- Role/model selection must preserve the existing capability-tier intent:
  - `test-writer` and `implementer` use an economy-tier Gemini configuration.
  - `reviewer` uses a premium-tier Gemini configuration.
- Concrete Gemini model names may live only in the Gemini adapter/config layer, not in the shared pipeline core.
- Any Gemini-specific approval or tool-execution settings must remain adapter-level details unless the shared core requires a provider-independent contract change.

### Reviewer output contract

- Review stages must still produce output that the existing canonical reviewer JSON contract can consume:
  - `decision`
  - `summary`
  - `blocking`
- The adapter must prefer Gemini CLI features that make reviewer output reliably machine-readable, but must not assume a dedicated schema-enforcement flag exists.
- If Gemini returns malformed reviewer output, the existing single repair pass in the shared pipeline must remain usable.
- The provider must tolerate Gemini CLI output wrappers or formatting as long as the shared reviewer normalization path can recover a valid decision.

### Diagnostics

- Provider failures must include enough subprocess output to debug local Gemini CLI issues on Windows.
- The adapter must not depend on fragile runtime-created temp directories outside the repository workspace.
- The adapter must capture enough subprocess output to distinguish executable-resolution failures, CLI flag incompatibilities, and model/tool execution failures.

### Non-goals

- Do not redesign the shared pipeline around Gemini-specific approval semantics unless local Gemini CLI behavior proves that necessary.
- Do not add performance or latency guarantees for Gemini stage execution in this spec.

## Acceptance Criteria

### AC-1: CLI accepts Gemini

- `uv run python scripts/run_pipeline.py smoke-test --provider gemini` parses successfully and selects a Gemini provider.

### AC-2: Gemini adapter is local-CLI based

- The provider adapter launches the local Gemini CLI executable in non-interactive mode with the repository root as working directory.
- On Windows, executable lookup works when the CLI is installed as an npm shim such as `gemini.cmd`.

### AC-3: Reviewer output is machine-usable

- A reviewer-stage Gemini response can be normalized by the existing reviewer JSON handling path, including the repair path when needed.

### AC-4: Failures are debuggable

- If the Gemini CLI subprocess exits non-zero, the resulting pipeline error includes enough command output to diagnose the failure.

### AC-5: Adapter does not depend on shared-core Gemini hacks

- Gemini-specific subprocess flags, executable resolution, and output handling live in the Gemini adapter layer unless a shared-core change is explicitly required for all providers.

### AC-6: Smoke test works end-to-end

- From a clean `smoke-test` state, `uv run python scripts/run_pipeline.py smoke-test --provider gemini` can reach `DONE` on a machine with a working local Gemini CLI setup.
