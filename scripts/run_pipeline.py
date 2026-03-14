"""Run the provider-generalized autonomous pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from vla_agent.pipeline.core import run_from_cli
from vla_agent.pipeline.providers.claude import ClaudeProvider
from vla_agent.pipeline.providers.codex import CodexProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous TDD pipeline.")
    parser.add_argument("task", help="Task id. Spec is resolved as specs/<task>-spec.md.")
    parser.add_argument(
        "--provider",
        choices=["claude", "codex"],
        required=True,
        help="Provider runtime to execute pipeline stages with.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=4,
        help="Maximum revision attempts per review loop.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    providers = {
        "claude": ClaudeProvider,
        "codex": CodexProvider,
    }
    provider = providers[args.provider]()
    return run_from_cli(
        task=args.task,
        provider=provider,
        repo_root=repo_root,
        max_revisions=args.max_revisions,
    )


if __name__ == "__main__":
    sys.exit(main())
