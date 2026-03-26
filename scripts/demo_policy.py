"""Record video of a trained policy playing Crafter."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from vla_agent.envs.crafter_env import CrafterEnv
from vla_agent.models import CrafterCNN, CrafterVLA, InstructionEncoder

INSTRUCTION_TASK_MAP = {
    "collect wood": "collect_wood",
    "place table": "place_table",
    "collect stone": "collect_stone",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a trained Crafter policy.")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--policy-type", choices=["cnn", "vla", "vla-cnn"], default="cnn")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="artifacts/demo")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Comma-separated instructions (VLA only; defaults to all 3)",
    )
    return parser.parse_args()


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_policy(
    policy_type: str,
    model_path: Path,
    device: torch.device,
    requested_num_frames: int,
) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = dict(checkpoint.get("metadata") or {})
    else:
        state_dict = checkpoint
        metadata = {}

    if policy_type == "cnn":
        model = CrafterCNN(num_actions=CrafterEnv.num_actions)
        num_frames = 1
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
    return model, num_frames


def _obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(obs).permute(2, 0, 1).to(torch.float32) / 255.0
    return tensor.unsqueeze(0).to(device)


def _init_frame_buffer(num_frames: int, device: torch.device) -> deque[torch.Tensor]:
    frame = torch.zeros(3, 64, 64, dtype=torch.float32, device=device)
    return deque([frame.clone() for _ in range(num_frames)], maxlen=num_frames)


def _record_episode(
    model: torch.nn.Module,
    seed: int,
    max_steps: int,
    device: torch.device,
    policy_type: str,
    num_frames: int,
    text_embed: torch.Tensor | None = None,
) -> tuple[list[np.ndarray], int, float, dict]:
    env = CrafterEnv(seed=seed)
    try:
        obs, info = env.reset()
        frames = [obs]
        total_reward = 0.0
        num_steps = 0

        frame_buffer: deque[torch.Tensor] | None = None
        is_vla = policy_type in {"vla", "vla-cnn"}
        if is_vla and num_frames > 1:
            frame_buffer = _init_frame_buffer(num_frames, device)

        for _ in range(max_steps):
            obs_tensor = _obs_to_tensor(obs, device)
            model_obs = obs_tensor

            if frame_buffer is not None:
                frame_buffer.append(obs_tensor.squeeze(0))
                model_obs = torch.stack(list(frame_buffer), dim=0).unsqueeze(0)

            with torch.no_grad():
                if is_vla:
                    logits = model(model_obs, text_embed)
                else:
                    logits = model(model_obs)

            action = int(logits.argmax(dim=1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(obs)
            total_reward += float(reward)
            num_steps += 1
            if terminated or truncated:
                break

        return frames, num_steps, total_reward, info
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    model_path = Path(args.model)
    model, effective_num_frames = _load_policy(
        args.policy_type, model_path, device, args.num_frames
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_vla = args.policy_type in {"vla", "vla-cnn"}

    if is_vla:
        if args.instructions:
            instructions = [s.strip() for s in args.instructions.split(",")]
        else:
            instructions = list(INSTRUCTION_TASK_MAP.keys())

        encoder = InstructionEncoder(device=device)

        for instruction in instructions:
            task = INSTRUCTION_TASK_MAP.get(instruction, instruction.replace(" ", "_"))
            text_embed = encoder.encode(instruction).unsqueeze(0).to(device)
            print(f'Recording instruction: "{instruction}"')

            for ep in range(args.num_episodes):
                seed = args.base_seed + ep
                frames, num_steps, total_reward, info = _record_episode(
                    model,
                    seed,
                    args.max_steps,
                    device,
                    args.policy_type,
                    effective_num_frames,
                    text_embed,
                )

                inventory = info.get("inventory", {})
                achievements = info.get("achievements", {})
                video_path = output_dir / f"{task}_seed{seed}.mp4"
                iio.imwrite(
                    str(video_path),
                    np.array(frames),
                    fps=args.fps,
                    plugin="pyav",
                    codec="libx264",
                )
                print(
                    f"  Episode {ep + 1}/{args.num_episodes} | seed={seed} | "
                    f"steps={num_steps} | reward={total_reward:.1f} | "
                    f"wood={inventory.get('wood', 0)} "
                    f"table={achievements.get('place_table', 0)} "
                    f"stone={inventory.get('stone', 0)} | {video_path}"
                )
    else:
        for ep in range(args.num_episodes):
            seed = args.base_seed + ep
            frames, num_steps, total_reward, info = _record_episode(
                model,
                seed,
                args.max_steps,
                device,
                args.policy_type,
                1,
            )

            inventory = info.get("inventory", {})
            achievements = info.get("achievements", {})
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
                f"steps={num_steps} | reward={total_reward:.1f} | "
                f"wood={inventory.get('wood', 0)} "
                f"table={achievements.get('place_table', 0)} "
                f"stone={inventory.get('stone', 0)} | {video_path}"
            )

    print(f"Done. Videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
