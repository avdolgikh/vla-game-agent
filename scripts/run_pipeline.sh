#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper that keeps the original Claude-first entrypoint available.
TASK="${1:?Usage: ./scripts/run_pipeline.sh <task-id> [extra args...] }"
shift
exec uv run python scripts/run_pipeline.py "$TASK" --provider claude "$@"
