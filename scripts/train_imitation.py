"""Train the MVP-1 vision-only Crafter policy via behavioral cloning."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vla_agent.data import TrajectoryDataset, train_val_split
from vla_agent.models import CrafterCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a vision-only Crafter imitation policy.")
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
        "--class-weights", action="store_true", help="Use inverse-frequency class weights"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
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
    model: CrafterCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        obs = batch["observation"].to(device)
        actions = batch["action"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(obs)
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
    model: CrafterCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            obs = batch["observation"].to(device)
            actions = batch["action"].to(device)
            logits = model(obs)
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


def train() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    _set_seed(args.seed, device)

    dataset = TrajectoryDataset(data_dirs=args.data_dirs)
    train_loader, val_loader, num_train, num_val = _make_dataloaders(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    model = CrafterCNN(num_actions=dataset.num_actions).to(device)
    criterion = _make_loss(dataset, device, args.class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
        "class_weights": "inverse_frequency" if args.class_weights else "none",
        "num_train_samples": num_train,
        "num_val_samples": num_val,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "data_dirs": json.dumps([str(Path(d)) for d in args.data_dirs]),
    }
    best_val_acc = -1.0
    best_epoch = 0
    epoch_records: list[dict[str, Any]] = []

    try:
        _log_mlflow_params(mlflow, params)

        for epoch in range(1, args.epochs + 1):
            train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
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

        torch.save(model.state_dict(), final_model_path)

        config = dict(params)
        config["data_dirs"] = [str(Path(d)) for d in args.data_dirs]
        config["class_weights"] = "inverse_frequency" if args.class_weights else "none"
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
