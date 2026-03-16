"""Evaluate a trained Crafter policy via environment rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vla_agent.envs.crafter_env import CrafterEnv
from vla_agent.models import CrafterCNN

TASK_NAMES = ("collect_wood", "place_table", "collect_stone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an imitation policy in Crafter.")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--policy-type", choices=["cnn"], default="cnn")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="artifacts/eval/mvp1")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    return parser.parse_args()


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    if arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device option: {arg}")


def _load_policy(policy_type: str, model_path: Path, device: torch.device) -> CrafterCNN:
    if policy_type != "cnn":
        raise ValueError(f"Unsupported policy type: {policy_type}")
    model = CrafterCNN(num_actions=CrafterEnv.num_actions)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(obs).permute(2, 0, 1).to(torch.float32) / 255.0
    return tensor.unsqueeze(0).to(device)


def _success_flags(info: dict) -> dict[str, bool]:
    inventory = info.get("inventory", {})
    achievements = info.get("achievements", {})
    return {
        "collect_wood": inventory.get("wood", 0) >= 1,
        "place_table": achievements.get("place_table", 0) >= 1,
        "collect_stone": inventory.get("stone", 0) >= 1,
    }


def _run_episode(
    model: CrafterCNN,
    seed: int,
    max_steps: int,
    device: torch.device,
) -> tuple[int, float, dict[str, bool]]:
    env = CrafterEnv(seed=seed)
    try:
        obs, info = env.reset()
        total_reward = 0.0
        num_steps = 0
        for _ in range(max_steps):
            obs_tensor = _obs_to_tensor(obs, device)
            with torch.no_grad():
                logits = model(obs_tensor)
            action = int(logits.argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            num_steps += 1
            if terminated or truncated:
                break
        success = _success_flags(info)
        return num_steps, total_reward, success
    finally:
        env.close()


def evaluate() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model = _load_policy(args.policy_type, model_path, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"

    success_counts = {task: 0 for task in TASK_NAMES}
    episodes: list[dict[str, object]] = []

    for idx in range(args.num_episodes):
        seed = args.base_seed + idx
        num_steps, total_reward, success = _run_episode(model, seed, args.max_steps, device)
        episodes.append(
            {
                "seed": seed,
                "num_steps": num_steps,
                "total_reward": total_reward,
                "successes": success,
            }
        )
        for task, succeeded in success.items():
            success_counts[task] += int(bool(succeeded))

        print(
            "Episode {ep:02d}/{total:02d} | seed={seed} | steps={steps:03d} | reward={reward:.1f} | "
            "wood={wood} table={table} stone={stone}".format(
                ep=idx + 1,
                total=args.num_episodes,
                seed=seed,
                steps=num_steps,
                reward=total_reward,
                wood=int(success["collect_wood"]),
                table=int(success["place_table"]),
                stone=int(success["collect_stone"]),
            )
        )

    success_rates = {
        task: (success_counts[task] / args.num_episodes) if args.num_episodes else 0.0
        for task in TASK_NAMES
    }
    results = {
        "model": str(model_path),
        "num_episodes": args.num_episodes,
        "base_seed": args.base_seed,
        "max_steps": args.max_steps,
        "success_rates": success_rates,
        "episodes": episodes,
    }
    results_path.write_text(json.dumps(results, indent=2))

    print(f"Done. {args.num_episodes} episodes evaluated.")
    for task in TASK_NAMES:
        count = success_counts[task]
        rate = success_rates[task] * 100.0
        print(f"  {task}:  {count}/{args.num_episodes} ({rate:.1f}%)")
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    evaluate()
