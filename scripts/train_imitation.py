"""Train Crafter imitation policies (vision-only CNN or instruction-conditioned VLA)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vla_agent.data import TrajectoryDataset, train_val_split
from vla_agent.models import CrafterCNN, CrafterVLA, InstructionEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Crafter imitation policy.")
    parser.add_argument(
        "--data-dirs", nargs="+", required=True, help="Trajectory directories to load"
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/models/mvp1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--experiment-name", type=str, default="mvp1")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "vla", "vla-cnn"],
        default="cnn",
        help="Model architecture to train",
    )
    parser.add_argument(
        "--class-weights", action="store_true", help="Use inverse-frequency class weights"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of stacked frames per sample (VLA variants only)",
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


def _checkpoint_payload(model: nn.Module, metadata: dict[str, Any] | None) -> dict[str, Any] | dict:
    """Wrap a model state dict with metadata when provided."""
    state_dict = model.state_dict()
    if not metadata:
        return state_dict
    return {"state_dict": state_dict, "metadata": dict(metadata)}


def _set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _make_dataloaders(
    dataset: TrajectoryDataset,
    val_fraction: float,
    seed: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    train_subset, val_subset = train_val_split(dataset, val_fraction=val_fraction, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, len(train_subset), len(val_subset)


def _make_loss(
    dataset: TrajectoryDataset,
    device: torch.device,
    use_class_weights: bool,
) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()
    counts = dataset.action_counts().astype(np.float32)
    weights = np.ones_like(counts, dtype=np.float32)
    mask = counts > 0
    if mask.any():
        mean_count = counts[mask].mean()
        weights[mask] = mean_count / counts[mask]
    return nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device=device, dtype=torch.float32)
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
    instruction_cache: dict[str, torch.Tensor] | None,
    instruction_encoder: InstructionEncoder | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        obs = batch["observation"].to(device)
        actions = batch["action"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_model(
            model_type,
            model,
            obs,
            batch,
            instruction_cache,
            instruction_encoder,
            device,
        )
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()
        batch_size = actions.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
    instruction_cache: dict[str, torch.Tensor] | None,
    instruction_encoder: InstructionEncoder | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            obs = batch["observation"].to(device)
            actions = batch["action"].to(device)
            logits = _forward_model(
                model_type,
                model,
                obs,
                batch,
                instruction_cache,
                instruction_encoder,
                device,
            )
            loss = criterion(logits, actions)
            batch_size = actions.shape[0]
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            correct += int((preds == actions).sum().item())
            total += batch_size
    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def _maybe_start_mlflow(experiment_name: str, disabled: bool):
    if disabled:
        return None
    import mlflow

    tracking_dir = Path("mlruns").resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=experiment_name)
    return mlflow


def _log_mlflow_params(mlflow, params: dict[str, Any]) -> None:
    if mlflow is None:
        return
    mlflow.log_params(params)


def _log_mlflow_metrics(mlflow, metrics: dict[str, float], step: int | None = None) -> None:
    if mlflow is None:
        return
    for key, value in metrics.items():
        if step is None:
            mlflow.log_metric(key, value)
        else:
            mlflow.log_metric(key, value, step=step)


def _log_mlflow_artifacts(mlflow, output_dir: Path) -> None:
    if mlflow is None:
        return
    mlflow.log_artifacts(str(output_dir))


def _end_mlflow_run(mlflow) -> None:
    if mlflow is None:
        return
    mlflow.end_run()


def _initialize_model(
    model_type: str,
    num_actions: int,
    device: torch.device,
    lr: float,
    num_frames: int,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    if model_type == "cnn":
        model = CrafterCNN(num_actions=num_actions).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif model_type == "vla":
        model = CrafterVLA(num_actions=num_actions, pretrained=True, num_frames=num_frames).to(
            device
        )
        optimizer = torch.optim.Adam(model.action_head.parameters(), lr=lr)
    elif model_type == "vla-cnn":
        model = CrafterVLA(
            num_actions=num_actions,
            pretrained=False,
            num_frames=num_frames,
            vision_type="cnn",
        ).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model, optimizer


def _build_instruction_support(
    model_type: str,
    device: torch.device,
    dataset: TrajectoryDataset,
) -> tuple[InstructionEncoder | None, dict[str, torch.Tensor] | None]:
    if model_type not in {"vla", "vla-cnn"}:
        return None, None
    encoder = InstructionEncoder(device=device)
    cache: dict[str, torch.Tensor] = {}
    unique_instructions = dataset.unique_instructions()
    if unique_instructions:
        embeddings = encoder.encode_batch(unique_instructions)
        for idx, text in enumerate(unique_instructions):
            cache[text] = embeddings[idx].detach().clone()
    return encoder, cache


def _batch_text_embeddings(
    instructions: Sequence[str] | str,
    cache: dict[str, torch.Tensor] | None,
    encoder: InstructionEncoder | None,
    device: torch.device,
) -> torch.Tensor:
    if cache is None or encoder is None:
        raise RuntimeError("Instruction encoder not initialized.")
    texts = [instructions] if isinstance(instructions, str) else list(instructions)
    if not texts:
        return torch.empty((0, encoder.embed_dim), dtype=torch.float32, device=device)
    tensors: list[torch.Tensor] = []
    for text in texts:
        tensor = cache.get(text)
        if tensor is None:
            tensor = encoder.encode(text).detach().clone()
            cache[text] = tensor
        tensors.append(tensor)
    stacked = torch.stack(tensors, dim=0).to(device)
    return stacked


def _forward_model(
    model_type: str,
    model: nn.Module,
    obs: torch.Tensor,
    batch: dict[str, Any],
    instruction_cache: dict[str, torch.Tensor] | None,
    instruction_encoder: InstructionEncoder | None,
    device: torch.device,
) -> torch.Tensor:
    if model_type == "cnn":
        return model(obs)
    if model_type in {"vla", "vla-cnn"}:
        instructions = batch.get("instruction")
        if instructions is None:
            raise RuntimeError("Batch missing 'instruction' field required for VLA training.")
        text_embeddings = _batch_text_embeddings(
            instructions,
            instruction_cache,
            instruction_encoder,
            device,
        )
        return model(obs, text_embeddings)
    raise ValueError(f"Unsupported model type: {model_type}")


def train() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed, device)

    if args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1.")
    vla_like = args.model_type in {"vla", "vla-cnn"}
    if not vla_like and args.num_frames != 1:
        raise ValueError("--num-frames > 1 is only supported for VLA model types.")

    dataset_num_frames = args.num_frames if vla_like else 1
    dataset = TrajectoryDataset(data_dirs=args.data_dirs, num_frames=dataset_num_frames)
    train_loader, val_loader, num_train, num_val = _make_dataloaders(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    model, optimizer = _initialize_model(
        args.model_type,
        dataset.num_actions,
        device,
        args.lr,
        dataset_num_frames,
    )
    criterion = _make_loss(dataset, device, args.class_weights)
    instruction_encoder, instruction_cache = _build_instruction_support(
        args.model_type,
        device,
        dataset,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"
    final_model_path = output_dir / "final_model.pt"
    log_path = output_dir / "train_log.json"

    mlflow = _maybe_start_mlflow(args.experiment_name, args.no_mlflow)
    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": device.type,
        "model_type": args.model_type,
        "class_weights": "inverse_frequency" if args.class_weights else "none",
        "num_train_samples": num_train,
        "num_val_samples": num_val,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "data_dirs": json.dumps([str(Path(d)) for d in args.data_dirs]),
        "num_frames": dataset_num_frames,
    }
    if args.model_type == "vla":
        params["vision_type"] = "convnext"
    elif args.model_type == "vla-cnn":
        params["vision_type"] = "cnn"
    best_val_acc = -1.0
    best_epoch = 0
    epoch_records: list[dict[str, Any]] = []
    checkpoint_metadata: dict[str, Any] | None = None
    if vla_like:
        checkpoint_metadata = {
            "model_type": args.model_type,
            "num_actions": dataset.num_actions,
            "num_frames": dataset_num_frames,
            "vision_type": "cnn" if args.model_type == "vla-cnn" else "convnext",
        }

    try:
        _log_mlflow_params(mlflow, params)

        for epoch in range(1, args.epochs + 1):
            train_loss = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                args.model_type,
                instruction_cache,
                instruction_encoder,
            )
            val_loss, val_acc = _evaluate(
                model,
                val_loader,
                criterion,
                device,
                args.model_type,
                instruction_cache,
                instruction_encoder,
            )
            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(_checkpoint_payload(model, checkpoint_metadata), best_model_path)
            epoch_records.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            _log_mlflow_metrics(mlflow, metrics, step=epoch)

            line = (
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"train_loss={train_loss:.3f} | val_loss={val_loss:.3f} | val_acc={val_acc:.3f}"
            )
            if improved:
                line += " | best=true"
            print(line)

        torch.save(_checkpoint_payload(model, checkpoint_metadata), final_model_path)

        config = dict(params)
        config["data_dirs"] = [str(Path(d)) for d in args.data_dirs]
        config["class_weights"] = "inverse_frequency" if args.class_weights else "none"
        config["model_type"] = args.model_type
        config["num_frames"] = dataset_num_frames
        train_log = {
            "config": config,
            "epochs": epoch_records,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
        }
        log_path.write_text(json.dumps(train_log, indent=2))

        _log_mlflow_metrics(mlflow, {"best_val_acc": best_val_acc, "best_epoch": best_epoch})
        _log_mlflow_artifacts(mlflow, output_dir)
    finally:
        _end_mlflow_run(mlflow)

    print(f"Done. Best val_acc={best_val_acc:.3f} at epoch {best_epoch}.")
    print(f"Saved: {best_model_path}")
    print(f"       {final_model_path}")
    print(f"       {log_path}")


if __name__ == "__main__":
    train()
