"""Integration tests for the trajectory collection script and outputs."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vla_agent.envs.crafter_env import CrafterEnv

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "collect_trajectories.py"
BASE_SEED = 42
NUM_EPISODES = 3
MAX_STEPS = 300

_POLICY_THRESHOLDS: list[tuple[str, str, float]] = [
    ("collect_wood", "collect wood", 0.9),
    ("place_table", "place table", 0.8),
    ("collect_stone", "collect stone", 0.7),
]


def _run_collection_script(policy_name: str, output_dir: Path) -> subprocess.CompletedProcess:
    output_dir = Path(output_dir)
    cmd = [
        "uv",
        "run",
        "python",
        str(SCRIPT_PATH),
        "--policy",
        policy_name,
        "--num-episodes",
        str(NUM_EPISODES),
        "--base-seed",
        str(BASE_SEED),
        "--max-steps",
        str(MAX_STEPS),
        "--output-dir",
        str(output_dir),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)


@pytest.mark.integration
@pytest.mark.parametrize("policy_name,instruction,min_success_rate", _POLICY_THRESHOLDS)
def test_collect_trajectories_manifest_and_files(
    policy_name: str, instruction: str, min_success_rate: float, tmp_path: Path
) -> None:
    output_dir = tmp_path / policy_name
    _run_collection_script(policy_name, output_dir)

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists(), "manifest.json must exist after running the collection script"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["policy"] == policy_name
    assert manifest["instruction"] == instruction
    assert manifest["action_space_size"] == CrafterEnv.num_actions
    assert manifest["observation_shape"] == [64, 64, 3]
    assert manifest["num_episodes"] == NUM_EPISODES
    assert manifest["base_seed"] == BASE_SEED

    episodes = manifest["episodes"]
    assert isinstance(episodes, list)
    assert len(episodes) == NUM_EPISODES
    assert manifest["num_episodes"] == len(episodes)

    success_count = 0
    for idx, episode in enumerate(episodes):
        for field in ("file", "seed", "success", "num_steps", "total_reward"):
            assert field in episode, f"Episode {idx} must include '{field}'"

        expected_file = f"episode_{idx:03d}.npz"
        assert episode["file"] == expected_file
        assert episode["seed"] == BASE_SEED + idx
        assert isinstance(episode["success"], bool)
        assert isinstance(episode["num_steps"], int)
        assert 0 <= episode["num_steps"] <= MAX_STEPS
        assert isinstance(episode["total_reward"], float)

        success_count += int(episode["success"])

        episode_path = output_dir / episode["file"]
        assert episode_path.exists(), f"Episode file {episode_path} must exist"

        with np.load(episode_path, allow_pickle=False) as archive:
            observations = archive["observations"]
            actions = archive["actions"]
            rewards = archive["rewards"]

        assert actions.shape == (episode["num_steps"],)
        assert rewards.shape == (episode["num_steps"],)
        assert observations.shape[0] == actions.shape[0] + 1
        assert observations.shape[1:] == (64, 64, 3)
        assert observations.dtype == np.uint8
        assert actions.dtype == np.int32
        assert rewards.dtype == np.float32
        if actions.size:
            assert actions.min() >= 0
            assert actions.max() < CrafterEnv.num_actions

    assert manifest["success_count"] == success_count
    required_successes = math.ceil(min_success_rate * NUM_EPISODES)
    assert success_count >= required_successes
