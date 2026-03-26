"""Generate comparison plots from training logs and evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MILESTONES = [
    ("MVP-1: CNN baseline", "mvp1"),
    ("MVP-2: + text (ConvNeXt)", "mvp2"),
    ("MVP-2.1: + 224x224 resize", "mvp2.1"),
    ("MVP-2.2: + frame stacking", "mvp2.2"),
    ("MVP-2.3: trainable CNN", "mvp2.3"),
]

TASKS = ["collect_wood", "place_table", "collect_stone"]
TASK_LABELS = ["collect wood", "place table", "collect stone"]

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_training_curves(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (label, key), color in zip(MILESTONES, colors):
        log_path = ARTIFACTS_ROOT / "models" / key / "train_log.json"
        if not log_path.exists():
            continue
        log = _load_json(log_path)
        epochs = [e["epoch"] for e in log["epochs"]]
        val_acc = [e["val_acc"] for e in log["epochs"]]
        ax.plot(epochs, val_acc, label=label, color=color, linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("Training Progress Across Milestones", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0.3, 0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_task_success_rates(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    rates_per_milestone: list[tuple[str, dict[str, float]]] = []
    for label, key in MILESTONES:
        eval_path = ARTIFACTS_ROOT / "eval" / key / "eval_results.json"
        if not eval_path.exists():
            continue
        results = _load_json(eval_path)
        rates_per_milestone.append((label, results["success_rates"]))

    n_milestones = len(rates_per_milestone)
    n_tasks = len(TASKS)
    bar_width = 0.15
    group_width = n_milestones * bar_width

    for i, (label, rates) in enumerate(rates_per_milestone):
        positions = [j - group_width / 2 + i * bar_width + bar_width / 2 for j in range(n_tasks)]
        values = [rates.get(task, 0.0) * 100 for task in TASKS]
        bars = ax.bar(positions, values, bar_width, label=label, color=colors[i], edgecolor="white")
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Per-Task Success Rates Across Milestones", fontsize=14)
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(TASK_LABELS, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    output_dir = ARTIFACTS_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(output_dir / "training_curves.png")
    plot_task_success_rates(output_dir / "task_success_rates.png")
    print("Done.")


if __name__ == "__main__":
    main()
