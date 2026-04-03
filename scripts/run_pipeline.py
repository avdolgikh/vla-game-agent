"""Run the provider-generalized autonomous pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

from vla_agent.pipeline.core import PipelineConfig, run_from_cli
from vla_agent.pipeline.providers.claude import ClaudeProvider
from vla_agent.pipeline.providers.codex import CodexProvider
from vla_agent.pipeline.providers.gemini import GeminiProvider
from vla_agent.pipeline.providers.opencode import OpenCodeProvider


def load_config(path: Path) -> PipelineConfig:
    """Load a PipelineConfig from a TOML file. Unspecified fields use defaults."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return PipelineConfig(
        **{k: v for k, v in data.items() if k in PipelineConfig.__dataclass_fields__}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous TDD pipeline.")
    parser.add_argument("task", help="Task id. Spec is resolved as specs/<task>-spec.md.")
    parser.add_argument(
        "--provider",
        choices=["claude", "codex", "gemini", "opencode"],
        required=True,
        help="Provider runtime to execute pipeline stages with.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML config file. Fields map to PipelineConfig.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory. Defaults to the parent of scripts/.",
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
    repo_root = (
        args.repo_root.resolve() if args.repo_root else Path(__file__).resolve().parent.parent
    )
    config = load_config(args.config) if args.config else PipelineConfig()
    providers = {
        "claude": ClaudeProvider,
        "codex": CodexProvider,
        "gemini": GeminiProvider,
        "opencode": OpenCodeProvider,
    }
    provider = providers[args.provider]()
    return run_from_cli(
        task=args.task,
        provider=provider,
        repo_root=repo_root,
        max_revisions=args.max_revisions,
        config=config,
    )


if __name__ == "__main__":
    sys.exit(main())
