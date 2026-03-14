"""Collect scripted policy trajectories for Crafter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Type

import numpy as np

from vla_agent.envs.crafter_env import CrafterEnv
from vla_agent.policies import (
    CollectStonePolicy,
    CollectWoodPolicy,
    PlaceTablePolicy,
    ScriptedPolicy,
)

POLICY_REGISTRY: dict[str, Type[ScriptedPolicy]] = {
    "collect_wood": CollectWoodPolicy,
    "place_table": PlaceTablePolicy,
    "collect_stone": CollectStonePolicy,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted Crafter trajectories.")
    parser.add_argument("--policy", choices=sorted(POLICY_REGISTRY.keys()), required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for manifest and episode files. Defaults to artifacts/trajectories/<policy>",
    )
    return parser.parse_args()


def _init_output_dir(policy_name: str, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return Path("artifacts") / "trajectories" / policy_name


def _stack_observations(frames: list[np.ndarray]) -> np.ndarray:
    return np.asarray(frames, dtype=np.uint8)


def _save_episode(
    episode_path: Path,
    observations: list[np.ndarray],
    actions: list[int],
    rewards: list[float],
) -> None:
    np.savez_compressed(
        episode_path,
        observations=_stack_observations(observations),
        actions=np.asarray(actions, dtype=np.int32),
        rewards=np.asarray(rewards, dtype=np.float32),
    )


def _run_episode(
    policy_cls: Type[ScriptedPolicy],
    seed: int,
    max_steps: int,
    episode_idx: int,
    output_dir: Path,
) -> tuple[dict, bool, str]:
    env = CrafterEnv(seed=seed)
    policy = policy_cls(env)
    try:
        obs, info = env.reset()
        policy.reset()
        observations: list[np.ndarray] = [obs.copy()]
        actions: list[int] = []
        rewards: list[float] = []
        total_reward = 0.0
        success = policy.succeeded(info)
        for _ in range(max_steps):
            action = int(policy.act(obs, info))
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs.copy())
            actions.append(action)
            rewards.append(float(reward))
            total_reward += float(reward)
            if policy.succeeded(info) or terminated or truncated:
                success = policy.succeeded(info)
                break
        else:
            success = policy.succeeded(info)
        num_steps = len(actions)
        episode_name = f"episode_{episode_idx:03d}.npz"
        episode_path = output_dir / episode_name
        _save_episode(episode_path, observations, actions, rewards)
        episode_meta = {
            "file": episode_name,
            "seed": seed,
            "success": bool(success),
            "num_steps": num_steps,
            "total_reward": float(total_reward),
        }
        return episode_meta, success, policy.instruction
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    policy_cls = POLICY_REGISTRY[args.policy]
    output_dir = _init_output_dir(args.policy, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    success_count = 0
    instruction: str | None = None

    for idx in range(args.num_episodes):
        seed = args.base_seed + idx
        episode_meta, success, policy_instruction = _run_episode(
            policy_cls, seed, args.max_steps, idx, output_dir
        )
        if instruction is None:
            instruction = policy_instruction
        success_count += int(success)
        episodes.append(episode_meta)
        print(
            "Episode {cur:03d}/{total:03d} | seed={seed} | steps={steps:03d} | reward={reward:.1f} | success={success_str}".format(
                cur=idx + 1,
                total=args.num_episodes,
                seed=seed,
                steps=episode_meta["num_steps"],
                reward=episode_meta["total_reward"],
                success_str="true" if success else "false",
            )
        )

    if instruction is None:
        temp_env = CrafterEnv(seed=args.base_seed)
        temp_policy = policy_cls(temp_env)
        instruction = temp_policy.instruction
        temp_env.close()

    action_space_size = CrafterEnv.num_actions
    manifest = {
        "policy": args.policy,
        "instruction": instruction,
        "action_space_size": action_space_size,
        "observation_shape": [64, 64, 3],
        "num_episodes": args.num_episodes,
        "base_seed": args.base_seed,
        "success_count": success_count,
        "episodes": episodes,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    success_rate = (success_count / args.num_episodes * 100.0) if args.num_episodes else 0.0
    print(f"Done. Saved {args.num_episodes} episodes to {output_dir}/")
    print(f"Success rate: {success_count}/{args.num_episodes} ({success_rate:.1f}%)")


if __name__ == "__main__":
    main()
