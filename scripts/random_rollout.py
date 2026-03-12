"""Random-policy rollout for Crafter.

Runs one episode with a uniform-random policy over the reduced action space and
saves frames and episode metadata to an output directory.
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from vla_agent.envs.crafter_env import CrafterEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-policy Crafter rollout.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for env and action sampling.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/rollouts/run",
        help="Directory for output files.",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--save-video", action="store_true", help="If set, also save an mp4 video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = CrafterEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    obs, _ = env.reset()

    actions_taken: list[int] = []
    action_names_taken: list[str] = []
    frames: list[np.ndarray] = []
    total_reward = 0.0
    final_info: dict = {}

    for step in range(args.max_steps):
        action = int(rng.integers(0, env.num_actions))
        obs, reward, terminated, truncated, info = env.step(action)

        actions_taken.append(action)
        action_names_taken.append(env.action_names[action])
        frames.append(obs)
        total_reward += reward
        final_info = info

        print(
            f"Step {step + 1:03d} | action: {env.action_names[action]} ({action}) "
            f"| reward: {reward:.1f} | total: {total_reward:.1f}"
        )

        if terminated or truncated:
            break

    env.close()

    num_steps = len(frames)

    # Save frames
    for i, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{i:03d}.png"
        iio.imwrite(str(frame_path), frame)

    # Save episode metadata
    episode_data = {
        "seed": args.seed,
        "num_steps": num_steps,
        "total_reward": total_reward,
        "actions_taken": actions_taken,
        "action_names_taken": action_names_taken,
        "final_inventory": final_info.get("inventory", {}),
        "achievements": final_info.get("achievements", {}),
    }
    episode_json_path = output_dir / "episode.json"
    with open(episode_json_path, "w") as f:
        json.dump(episode_data, f, indent=2)

    # Save video if requested
    if args.save_video:
        video_path = output_dir / "rollout.mp4"
        iio.imwrite(str(video_path), np.array(frames), fps=10, plugin="pyav", codec="libx264")
        print(f"Saved video to {video_path}")

    print(f"Episode done: {num_steps} steps, total reward {total_reward:.1f}")
    print(f"Saved {num_steps} frames to {frames_dir}/")
    print(f"Saved episode metadata to {episode_json_path}")


if __name__ == "__main__":
    main()
