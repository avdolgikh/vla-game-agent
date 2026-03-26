"""Evaluate trained Crafter policies (CNN baseline or VLA) via environment rollouts."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import torch

from vla_agent.envs.crafter_env import CrafterEnv
from vla_agent.models import CrafterCNN, CrafterVLA, InstructionEncoder

TASK_NAMES = ("collect_wood", "place_table", "collect_stone")
INSTRUCTION_TASK_MAP = {
    "collect wood": "collect_wood",
    "place table": "place_table",
    "collect stone": "collect_stone",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an imitation policy in Crafter.")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--policy-type", choices=["cnn", "vla", "vla-cnn"], default="cnn")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="artifacts/eval/mvp1")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of stacked frames per observation (VLA variants only)",
    )
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


def _load_checkpoint(model_path: Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = dict(checkpoint.get("metadata") or {})
    else:
        state_dict = checkpoint
        metadata = {}
    return state_dict, metadata


def _load_policy(
    policy_type: str,
    model_path: Path,
    device: torch.device,
    requested_num_frames: int,
) -> tuple[torch.nn.Module, dict, int]:
    state_dict, metadata = _load_checkpoint(model_path, device)
    if policy_type == "cnn":
        num_frames = 1
        model = CrafterCNN(num_actions=CrafterEnv.num_actions)
    elif policy_type in {"vla", "vla-cnn"}:
        metadata_frames = metadata.get("num_frames")
        num_frames = int(metadata_frames) if metadata_frames is not None else requested_num_frames
        metadata_vision = metadata.get("vision_type")
        if isinstance(metadata_vision, str):
            vision_type = metadata_vision.lower()
        else:
            vision_type = "cnn" if policy_type == "vla-cnn" else "convnext"
        model = CrafterVLA(
            num_actions=CrafterEnv.num_actions,
            pretrained=False,
            num_frames=num_frames,
            vision_type=vision_type,
        )
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata, num_frames


def _obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(obs).permute(2, 0, 1).to(torch.float32) / 255.0
    return tensor.unsqueeze(0).to(device)


def _init_frame_buffer(num_frames: int, device: torch.device) -> deque[torch.Tensor]:
    frame = torch.zeros(3, 64, 64, dtype=torch.float32, device=device)
    return deque([frame.clone() for _ in range(num_frames)], maxlen=num_frames)


def _success_flags(info: dict) -> dict[str, bool]:
    inventory = info.get("inventory", {})
    achievements = info.get("achievements", {})
    return {
        "collect_wood": inventory.get("wood", 0) >= 1,
        "place_table": achievements.get("place_table", 0) >= 1,
        "collect_stone": inventory.get("stone", 0) >= 1,
    }


def _run_episode(
    model: torch.nn.Module,
    seed: int,
    max_steps: int,
    device: torch.device,
    policy_type: str,
    *,
    num_frames: int = 1,
    text_embed: torch.Tensor | None = None,
) -> tuple[int, float, dict[str, bool]]:
    env = CrafterEnv(seed=seed)
    try:
        obs, info = env.reset()
        total_reward = 0.0
        num_steps = 0
        frame_buffer: deque[torch.Tensor] | None = None
        is_vla_policy = policy_type in {"vla", "vla-cnn"}
        if is_vla_policy and num_frames > 1:
            frame_buffer = _init_frame_buffer(num_frames, device)

        for _ in range(max_steps):
            obs_tensor = _obs_to_tensor(obs, device)
            model_obs = obs_tensor
            if frame_buffer is not None:
                frame_buffer.append(obs_tensor.squeeze(0))
                stacked = torch.stack(list(frame_buffer), dim=0).unsqueeze(0)
                model_obs = stacked

            with torch.no_grad():
                if policy_type == "cnn":
                    logits = model(model_obs)
                elif is_vla_policy:
                    if text_embed is None:
                        raise RuntimeError("VLA policies require a text embedding.")
                    logits = model(model_obs, text_embed)
                else:
                    raise ValueError(f"Unsupported policy type: {policy_type}")
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
    if args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1.")
    device = _resolve_device(args.device)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model, metadata, effective_num_frames = _load_policy(
        args.policy_type,
        model_path,
        device,
        args.num_frames,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "eval_results.json"

    if args.policy_type == "cnn":
        success_counts = {task: 0 for task in TASK_NAMES}
        episodes: list[dict[str, object]] = []
        for idx in range(args.num_episodes):
            seed = args.base_seed + idx
            num_steps, total_reward, success = _run_episode(
                model,
                seed,
                args.max_steps,
                device,
                "cnn",
                num_frames=1,
            )
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
            "model_type": "cnn",
            "policy_type": "cnn",
            "num_frames": 1,
            "num_episodes": args.num_episodes,
            "base_seed": args.base_seed,
            "max_steps": args.max_steps,
            "success_rates": success_rates,
            "episodes": episodes,
        }
        print(f"Done. {args.num_episodes} episodes evaluated.")
        for task in TASK_NAMES:
            count = success_counts[task]
            rate = success_rates[task] * 100.0
            print(f"  {task}:  {count}/{args.num_episodes} ({rate:.1f}%)")
    else:
        encoder = InstructionEncoder(device=device)
        seeds = [args.base_seed + idx for idx in range(args.num_episodes)]
        instruction_results: dict[str, dict[str, object]] = {}
        success_rates: dict[str, float] = {}
        episodes: list[dict[str, object]] = []
        total_episodes = len(seeds) * len(INSTRUCTION_TASK_MAP)

        for instruction, task in INSTRUCTION_TASK_MAP.items():
            print(f'Evaluating instruction: "{instruction}"')
            text_embed = encoder.encode(instruction).unsqueeze(0)
            text_embed = text_embed.to(device)
            instruction_episodes: list[dict[str, object]] = []
            successes = 0
            for idx, seed in enumerate(seeds):
                num_steps, total_reward, success = _run_episode(
                    model,
                    seed,
                    args.max_steps,
                    device,
                    args.policy_type,
                    num_frames=effective_num_frames,
                    text_embed=text_embed,
                )
                task_success = bool(success.get(task, False))
                instruction_episodes.append(
                    {
                        "seed": seed,
                        "num_steps": num_steps,
                        "total_reward": total_reward,
                        "success": task_success,
                    }
                )
                episodes.append(
                    {
                        "instruction": instruction,
                        "task": task,
                        "seed": seed,
                        "num_steps": num_steps,
                        "total_reward": total_reward,
                        "success": task_success,
                        "successes": success,
                    }
                )
                successes += int(task_success)
                print(
                    f"  Episode {idx + 1:02d}/{len(seeds):02d} | seed={seed} | steps={num_steps:03d} | "
                    f"reward={total_reward:.1f} | success={int(task_success)}"
                )
            success_rate = successes / len(seeds) if seeds else 0.0
            instruction_results[instruction] = {
                "task": task,
                "success_rate": success_rate,
                "successes": successes,
                "episodes": instruction_episodes,
            }
            success_rates[task] = success_rate

        results = {
            "model": str(model_path),
            "model_type": metadata.get("model_type", args.policy_type),
            "policy_type": args.policy_type,
            "num_frames": effective_num_frames,
            "num_episodes": total_episodes,
            "num_episodes_per_instruction": args.num_episodes,
            "base_seed": args.base_seed,
            "max_steps": args.max_steps,
            "success_rates": success_rates,
            "instructions": instruction_results,
            "episodes": episodes,
            "checkpoint_metadata": metadata,
        }
        print(f"Done. {total_episodes} episodes evaluated ({args.num_episodes} per instruction).")
        for instruction, entry in instruction_results.items():
            rate = entry["success_rate"] * 100.0
            print(f'  "{instruction}": {entry["successes"]}/{args.num_episodes} ({rate:.1f}%)')

    results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    evaluate()
