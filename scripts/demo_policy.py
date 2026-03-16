"""Record video of a trained policy playing Crafter."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from vla_agent.envs.crafter_env import CrafterEnv
from vla_agent.models import CrafterCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a trained Crafter policy.")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="artifacts/demo")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    model = CrafterCNN(num_actions=CrafterEnv.num_actions)
    model.load_state_dict(torch.load(Path(args.model), map_location=device))
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.num_episodes):
        seed = args.base_seed + ep
        env = CrafterEnv(seed=seed)
        obs, info = env.reset()
        frames = [obs]
        total_reward = 0.0

        for step in range(args.max_steps):
            tensor = torch.from_numpy(obs).permute(2, 0, 1).to(torch.float32) / 255.0
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(model(tensor).argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(obs)
            total_reward += float(reward)
            if terminated or truncated:
                break

        env.close()

        inventory = info.get("inventory", {})
        achievements = info.get("achievements", {})
        wood = inventory.get("wood", 0)
        table = achievements.get("place_table", 0)
        stone = inventory.get("stone", 0)

        video_path = output_dir / f"episode_{ep:02d}_seed{seed}.mp4"
        iio.imwrite(
            str(video_path),
            np.array(frames),
            fps=args.fps,
            plugin="pyav",
            codec="libx264",
        )
        print(
            f"Episode {ep + 1}/{args.num_episodes} | seed={seed} | "
            f"steps={len(frames) - 1} | reward={total_reward:.1f} | "
            f"wood={wood} table={table} stone={stone} | {video_path}"
        )

    print(f"Done. Videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
