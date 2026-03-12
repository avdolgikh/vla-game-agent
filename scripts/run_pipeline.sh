#!/usr/bin/env bash
set -euo pipefail

# Allow running from within a Claude Code session
unset CLAUDECODE 2>/dev/null || true

TASK="${1:?Usage: ./scripts/run_pipeline.sh <task-id>}"
SPEC="specs/${TASK}-spec.md"
MAX_REVISIONS=2

# Permission mode: agents that write code get acceptEdits, reviewer is read-only
WRITE_PERMS="--permission-mode acceptEdits"
READ_PERMS="--permission-mode default"

REVIEW_SCHEMA='{"type":"object","properties":{"decision":{"type":"string","enum":["approve","revise"]},"summary":{"type":"string"},"blocking":{"type":"array","items":{"type":"string"}}},"required":["decision","summary"]}'

# Use python for JSON parsing (no jq dependency)
json_get() {
  python -c "import sys,json; d=json.load(sys.stdin); print(d$1)" 2>/dev/null
}

review() {
  local prompt="$1"
  claude -p "$prompt" --agent reviewer $READ_PERMS \
    --output-format json --json-schema "$REVIEW_SCHEMA"
}

decision_of() {
  local decision
  # Try structured_output.decision first, then top-level .decision
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

pretty_json() {
  echo "$1" | python -c "import sys,json; print(json.dumps(json.load(sys.stdin),indent=2))" 2>/dev/null || echo "$1"
}

# --- Pre-check ---
[[ -f "$SPEC" ]] || { echo "ERROR: Spec not found: $SPEC"; exit 1; }
echo "=== Pipeline: ${TASK} ==="
echo "=== Spec: ${SPEC} ==="
echo ""

# --- 1. Generate tests ---
echo "========================================="
echo "=== Stage 1: Test Generation (sonnet) ==="
echo "========================================="
claude -p "Read ${SPEC} and AGENTS.md. Write tests for task '${TASK}' covering all acceptance criteria. Create test files in tests/. Run 'uv run pytest' to confirm tests fail (red) for unimplemented code. Do NOT write production code." \
  --agent test-writer $WRITE_PERMS
echo ""

# --- 2. Review tests (bounded loop) ---
for ((i=0; i<=MAX_REVISIONS; i++)); do
  echo "============================================="
  echo "=== Stage 2: Test Review (opus, iter $i) ==="
  echo "============================================="
  RESULT=$(review "Read ${SPEC} and AGENTS.md. Review the test files in tests/ for task '${TASK}'. Check that all acceptance criteria have corresponding tests. Return JSON with your decision.")
  pretty_json "$RESULT"

  DECISION=$(decision_of "$RESULT")
  echo "Decision: $DECISION"

  [[ "$DECISION" == "approve" ]] && break

  if ((i == MAX_REVISIONS)); then
    echo "FAIL: test revision cap reached after $MAX_REVISIONS iterations"
    exit 2
  fi

  echo ""
  echo "=== Stage 2b: Test Revision (sonnet, iter $i) ==="
  BLOCKING=$(echo "$RESULT" | python -c "
import sys, json
d = json.load(sys.stdin)
so = d.get('structured_output', d)
for b in so.get('blocking', []):
    print(b)
" 2>/dev/null) || BLOCKING=""
  claude -p "Revise tests for '${TASK}'. Reviewer feedback: ${BLOCKING}. Read ${SPEC} for context. Do NOT touch production code. Run 'uv run pytest' after revisions." \
    --agent test-writer $WRITE_PERMS
done
echo ""
echo ">>> Tests frozen <<<"
echo ""

# --- 3. Implement ---
echo "============================================="
echo "=== Stage 3: Implementation (sonnet)      ==="
echo "============================================="
claude -p "Read ${SPEC} and AGENTS.md. Read the frozen tests in tests/. Implement the minimal production code for task '${TASK}' to make all tests pass. Do NOT modify any test files. Run 'uv run pytest' to verify. Keep iterating until all tests pass." \
  --agent implementer $WRITE_PERMS
echo ""

# --- 4. Review code (bounded loop) ---
for ((i=0; i<=MAX_REVISIONS; i++)); do
  echo "============================================="
  echo "=== Stage 4: Code Review (opus, iter $i) ==="
  echo "============================================="
  RESULT=$(review "Read ${SPEC} and AGENTS.md. Review the implementation for task '${TASK}' against the spec and frozen tests. Run 'uv run pytest' to check status. Return JSON with your decision.")
  pretty_json "$RESULT"

  DECISION=$(decision_of "$RESULT")
  echo "Decision: $DECISION"

  [[ "$DECISION" == "approve" ]] && break

  if ((i == MAX_REVISIONS)); then
    echo "FAIL: code revision cap reached after $MAX_REVISIONS iterations"
    exit 3
  fi

  echo ""
  echo "=== Stage 4b: Code Revision (sonnet, iter $i) ==="
  BLOCKING=$(echo "$RESULT" | python -c "
import sys, json
d = json.load(sys.stdin)
so = d.get('structured_output', d)
for b in so.get('blocking', []):
    print(b)
" 2>/dev/null) || BLOCKING=""
  claude -p "Revise implementation for '${TASK}'. Reviewer feedback: ${BLOCKING}. Read ${SPEC} for context. Do NOT modify frozen tests. Run 'uv run pytest' after revisions." \
    --agent implementer $WRITE_PERMS
done

echo ""
echo "========================================="
echo "=== Pipeline COMPLETE: ${TASK}        ==="
echo "========================================="
