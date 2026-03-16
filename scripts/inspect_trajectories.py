"""Inspect collected trajectory data: reward distributions, step counts, success rates."""

import json
import numpy as np
import os
import sys


def inspect_policy(policy_dir: str) -> None:
    mpath = os.path.join(policy_dir, "manifest.json")
    if not os.path.exists(mpath):
        print(f"  No manifest.json found")
        return

    with open(mpath) as f:
        manifest = json.load(f)

    episodes = manifest["episodes"]
    n = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    rewards = [e["total_reward"] for e in episodes]
    steps = [e["num_steps"] for e in episodes]

    print(f"  Episodes: {n}")
    print(f"  Success:  {successes}/{n} ({100 * successes / n:.1f}%)")
    print(
        f"  Reward:   min={min(rewards):.1f}  max={max(rewards):.1f}  mean={np.mean(rewards):.2f}  std={np.std(rewards):.2f}"
    )
    print(
        f"  Steps:    min={min(steps)}  max={max(steps)}  mean={np.mean(steps):.1f}  std={np.std(steps):.1f}"
    )

    # Reward histogram
    unique, counts = np.unique(rewards, return_counts=True)
    print(f"  Reward distribution:")
    for v, c in zip(unique, counts):
        print(f"    {v:.1f}: {c} episodes ({100 * c / n:.1f}%)")

    # Spot-check a few .npz files
    npz_files = [e["file"] for e in episodes[:3]]
    print(f"  Sample .npz check ({len(npz_files)} files):")
    for fname in npz_files:
        fpath = os.path.join(policy_dir, fname)
        data = np.load(fpath)
        obs_shape = data["observations"].shape
        act_shape = data["actions"].shape
        rew = data["rewards"]
        print(
            f"    {fname}: obs={obs_shape}, act={act_shape}, reward_sum={rew.sum():.2f}, per-step rewards: {rew.tolist()}"
        )


def main():
    base = os.path.join("artifacts", "trajectories")
    if not os.path.isdir(base):
        print(f"No trajectory directory at {base}")
        sys.exit(1)

    for policy in sorted(os.listdir(base)):
        policy_dir = os.path.join(base, policy)
        if not os.path.isdir(policy_dir):
            continue
        print(f"\n=== {policy} ===")
        inspect_policy(policy_dir)


if __name__ == "__main__":
    main()
