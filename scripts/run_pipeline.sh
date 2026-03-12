#!/usr/bin/env bash
set -euo pipefail

# Allow running from within a Claude Code session
unset CLAUDECODE 2>/dev/null || true

# Ensure claude CLI is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Usage: "C:\Program Files\Git\bin\bash.exe" -l scripts/run_pipeline.sh <task-id>
TASK="${1:?Usage: ./scripts/run_pipeline.sh <task-id>}"
SPEC="specs/${TASK}-spec.md"
MAX_REVISIONS=2

# State & logging
STATE_DIR=".pipeline-state"
STATE_FILE="${STATE_DIR}/${TASK}.json"
LOG_FILE="${STATE_DIR}/${TASK}.log"

# Permission mode: bypassPermissions needed for headless agents to run Bash (uv run python -m pytest)
AGENT_PERMS="--permission-mode bypassPermissions"

REVIEW_SCHEMA='{"type":"object","properties":{"decision":{"type":"string","enum":["approve","revise"]},"summary":{"type":"string"},"blocking":{"type":"array","items":{"type":"string"}}},"required":["decision","summary"]}'

# =============================================================================
# Helpers
# =============================================================================

log() {
  echo "$@" | tee -a "$LOG_FILE"
}

# Use python for JSON parsing (no jq dependency)
json_get() {
  python -c "import sys,json; d=json.load(sys.stdin); print(d$1)" 2>/dev/null
}

pretty_json() {
  echo "$1" | python -c "import sys,json; print(json.dumps(json.load(sys.stdin),indent=2))" 2>/dev/null || echo "$1"
}

decision_of() {
  local decision
  decision=$(echo "$1" | python -c "
import sys, json
d = json.load(sys.stdin)
so = d.get('structured_output', {})
if isinstance(so, dict) and 'decision' in so:
    print(so['decision'])
elif 'decision' in d:
    print(d['decision'])
else:
    print('unknown')
" 2>/dev/null) || decision="unknown"
  echo "$decision"
}

blocking_of() {
  echo "$1" | python -c "
import sys, json
d = json.load(sys.stdin)
so = d.get('structured_output', d)
for b in so.get('blocking', []):
    print(b)
" 2>/dev/null || true
}

review() {
  local prompt="$1"
  claude -p "$prompt" --agent reviewer $AGENT_PERMS --output-format json --json-schema "$REVIEW_SCHEMA"
}

# =============================================================================
# State management
# =============================================================================

# Stage ordering (linear)
#   SPEC_APPROVED=0  TESTS_GENERATED=1  TESTS_FROZEN=2
#   CODE_IMPLEMENTED=3  CODE_VALIDATED=4  DONE=5
stage_order() {
  case "$1" in
    SPEC_APPROVED)     echo 0 ;;
    TESTS_GENERATED)   echo 1 ;;
    TESTS_FROZEN)      echo 2 ;;
    CODE_IMPLEMENTED)  echo 3 ;;
    CODE_VALIDATED)    echo 4 ;;
    DONE)              echo 5 ;;
    *)                 echo -1 ;;
  esac
}

save_state() {
  local stage="$1" iteration="${2:-0}"
  python -c "
import json, pathlib
pathlib.Path('${STATE_DIR}').mkdir(exist_ok=True)
with open('${STATE_FILE}', 'w') as f:
    json.dump({'task':'${TASK}','stage':'${stage}','iteration':${iteration}}, f)
"
  log "[state] ${stage} (iteration=${iteration})"
}

load_state() {
  if [[ -f "$STATE_FILE" ]]; then
    RESUME_STAGE=$(python -c "import json; d=json.load(open('${STATE_FILE}')); print(d['stage'])")
    RESUME_ITERATION=$(python -c "import json; d=json.load(open('${STATE_FILE}')); print(d['iteration'])")
    log "[resume] Continuing from stage=${RESUME_STAGE} iteration=${RESUME_ITERATION}"
  else
    RESUME_STAGE="SPEC_APPROVED"
    RESUME_ITERATION=0
  fi
}

# Returns 0 (true) if the given stage has already been completed
past() {
  local resume_ord target_ord
  resume_ord=$(stage_order "$RESUME_STAGE")
  target_ord=$(stage_order "$1")
  (( resume_ord >= target_ord ))
}

# Returns the starting iteration for a review loop.
# Uses saved iteration only when resuming INTO that loop's entry stage.
start_iter() {
  local loop_entry_stage="$1"
  if [[ "$RESUME_STAGE" == "$loop_entry_stage" ]]; then
    echo "$RESUME_ITERATION"
  else
    echo 0
  fi
}

# =============================================================================
# Guardrails
# =============================================================================

enforce_test_freeze() {
  local changed
  changed=$(git diff --name-only -- 'tests/') || true
  local staged
  staged=$(git diff --cached --name-only -- 'tests/') || true
  if [[ -n "$changed" || -n "$staged" ]]; then
    log "FAIL: frozen test files were modified: ${changed}${staged}"
    exit 4
  fi
}

# =============================================================================
# Pre-checks
# =============================================================================

[[ -f "$SPEC" ]] || { echo "ERROR: Spec not found: $SPEC"; exit 1; }
mkdir -p "$STATE_DIR"
log ""
log "=== Pipeline: ${TASK} ($(date -Iseconds)) ==="
log "=== Spec: ${SPEC} ==="

# Load resume state
load_state

# =============================================================================
# Stage 1: Test Generation
# =============================================================================

if ! past TESTS_GENERATED; then
  log ""
  log "========================================="
  log "=== Stage 1: Test Generation (sonnet) ==="
  log "========================================="
  claude -p "Read ${SPEC} and AGENTS.md. Write tests for task '${TASK}' covering all acceptance criteria. Create test files in tests/. Run 'uv run python -m pytest' to confirm tests fail for unimplemented code. Do NOT write production code." \
    --agent test-writer $AGENT_PERMS 2>&1 | tee -a "$LOG_FILE"
  save_state TESTS_GENERATED
fi

# =============================================================================
# Stage 2: Test Review (bounded loop)
# =============================================================================

if ! past TESTS_FROZEN; then
  START=$(start_iter TESTS_GENERATED)
  for ((i=START; i<=MAX_REVISIONS; i++)); do
    log ""
    log "============================================="
    log "=== Stage 2: Test Review (opus, iter $i) ==="
    log "============================================="
    RESULT=$(review "Read ${SPEC} and AGENTS.md. Review the test files in tests/ for task '${TASK}'. Check that all acceptance criteria have corresponding tests. Return JSON with your decision.")
    pretty_json "$RESULT" | tee -a "$LOG_FILE"

    DECISION=$(decision_of "$RESULT")
    log "Decision: $DECISION"

    if [[ "$DECISION" == "approve" ]]; then
      save_state TESTS_FROZEN
      break
    fi

    if ((i == MAX_REVISIONS)); then
      log "FAIL: test revision cap reached after $MAX_REVISIONS iterations"
      exit 2
    fi

    BLOCKING=$(blocking_of "$RESULT")
    log ""
    log "=== Stage 2b: Test Revision (sonnet, iter $i) ==="
    claude -p "Revise tests for '${TASK}'. Reviewer feedback: ${BLOCKING}. Read ${SPEC} for context. Do NOT touch production code. Run 'uv run python -m pytest' after revisions." \
      --agent test-writer $AGENT_PERMS 2>&1 | tee -a "$LOG_FILE"
    save_state TESTS_GENERATED $((i + 1))
  done
  log ""
  log ">>> Tests frozen <<<"
fi

# =============================================================================
# Stage 3: Implementation
# =============================================================================

if ! past CODE_IMPLEMENTED; then
  log ""
  log "============================================="
  log "=== Stage 3: Implementation (sonnet)      ==="
  log "============================================="
  claude -p "Read ${SPEC} and AGENTS.md. Read the frozen tests in tests/. Implement the minimal production code for task '${TASK}' to make all tests pass. Do NOT modify any test files. Run 'uv run python -m pytest' to verify. Keep iterating until all tests pass." \
    --agent implementer $AGENT_PERMS 2>&1 | tee -a "$LOG_FILE"
  enforce_test_freeze
  save_state CODE_IMPLEMENTED
fi

# =============================================================================
# Stage 4: Validation
# =============================================================================

if ! past CODE_VALIDATED; then
  log ""
  log "============================================="
  log "=== Stage 4: Validation (sonnet)           ==="
  log "============================================="
  claude -p "Read ${SPEC} and AGENTS.md. All tests pass. Now validate the implementation end-to-end: if the spec defines any runnable scripts or CLI commands, run them to verify they actually work. For example, if the spec defines a rollout script, run it with default/minimal arguments. If the spec has no runnable scripts, run 'uv run python -c ...' to import and exercise the main functions. Fix any issues you find. Do NOT modify test files. Run 'uv run python -m pytest' at the end to confirm tests still pass." \
    --agent implementer $AGENT_PERMS 2>&1 | tee -a "$LOG_FILE"
  enforce_test_freeze
  save_state CODE_VALIDATED
fi

# =============================================================================
# Stage 5: Code Review (bounded loop)
# =============================================================================

if ! past DONE; then
  START=$(start_iter CODE_VALIDATED)
  for ((i=START; i<=MAX_REVISIONS; i++)); do
    log ""
    log "============================================="
    log "=== Stage 5: Code Review (opus, iter $i) ==="
    log "============================================="
    RESULT=$(review "Read ${SPEC} and AGENTS.md. Review the implementation for task '${TASK}' against the spec and frozen tests. Run 'uv run python -m pytest' to check status. Return JSON with your decision.")
    pretty_json "$RESULT" | tee -a "$LOG_FILE"

    DECISION=$(decision_of "$RESULT")
    log "Decision: $DECISION"

    if [[ "$DECISION" == "approve" ]]; then
      save_state DONE
      break
    fi

    if ((i == MAX_REVISIONS)); then
      log "FAIL: code revision cap reached after $MAX_REVISIONS iterations"
      exit 3
    fi

    BLOCKING=$(blocking_of "$RESULT")
    log ""
    log "=== Stage 5b: Code Revision (sonnet, iter $i) ==="
    claude -p "Revise implementation for '${TASK}'. Reviewer feedback: ${BLOCKING}. Read ${SPEC} for context. Do NOT modify frozen tests. Run 'uv run python -m pytest' after revisions." \
      --agent implementer $AGENT_PERMS 2>&1 | tee -a "$LOG_FILE"
    enforce_test_freeze

    # Hard gate: tests must still pass after revision
    log ""
    log "=== Gate: pytest after code revision ==="
    uv run python -m pytest 2>&1 | tee -a "$LOG_FILE" || { log "FAIL: tests broke after code revision"; exit 5; }

    save_state CODE_VALIDATED $((i + 1))
  done
fi

log ""
log "========================================="
log "=== Pipeline COMPLETE: ${TASK}        ==="
log "========================================="
