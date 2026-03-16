"""Data utilities for MVP-1 vision-only imitation learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class TrajectoryDataset(Dataset):
    """In-memory dataset of (observation, action) pairs loaded from Crafter trajectories."""

    def __init__(self, data_dirs: Sequence[str | Path], transform: Callable | None = None) -> None:
        if not data_dirs:
            raise ValueError("TrajectoryDataset requires at least one data directory.")
        self._transform = transform
        self._episode_slices: list[tuple[int, int]] = []
        self._num_actions: int | None = None
        self._observations: np.ndarray
        self._actions: np.ndarray
        self._action_counts: np.ndarray | None = None
        self._load_directories(data_dirs)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._actions.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        obs = self._observations[index]
        action = int(self._actions[index])
        obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).to(torch.float32) / 255.0
        if self._transform is not None:
            obs_tensor = self._transform(obs_tensor)
        action_tensor = torch.tensor(action, dtype=torch.long)
        return {"observation": obs_tensor, "action": action_tensor}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def episode_slices(self) -> list[tuple[int, int]]:
        """Return a copy of the (start, end) index ranges for each loaded episode."""
        return list(self._episode_slices)

    @property
    def num_actions(self) -> int:
        if self._num_actions is None:
            raise RuntimeError("TrajectoryDataset has no recorded action_space_size.")
        return self._num_actions

    def action_counts(self) -> np.ndarray:
        """Return per-action sample counts."""
        if self._action_counts is None:
            if self.num_actions == 0:
                self._action_counts = np.zeros(0, dtype=np.int64)
            else:
                counts = np.bincount(
                    self._actions.astype(np.int64, copy=False),
                    minlength=self.num_actions,
                )
                if counts.shape[0] < self.num_actions:
                    padded = np.zeros(self.num_actions, dtype=np.int64)
                    padded[: counts.shape[0]] = counts
                    counts = padded
                self._action_counts = counts.astype(np.int64, copy=False)
        return self._action_counts.copy()

    # ------------------------------------------------------------------
    # Internal loading helpers
    # ------------------------------------------------------------------

    def _load_directories(self, data_dirs: Sequence[str | Path]) -> None:
        obs_chunks: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        total_samples = 0

        for directory in data_dirs:
            dir_path = Path(directory)
            if not dir_path.exists():
                raise FileNotFoundError(f"Trajectory directory '{dir_path}' does not exist.")
            manifest_path = dir_path / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found at '{manifest_path}'.")

            manifest = json.loads(manifest_path.read_text())
            action_space_size = int(manifest.get("action_space_size", 0))
            if action_space_size <= 0:
                raise ValueError(f"Manifest at '{manifest_path}' is missing action_space_size.")
            if self._num_actions is None:
                self._num_actions = action_space_size
            elif self._num_actions != action_space_size:
                raise ValueError(
                    f"Inconsistent action_space_size across datasets: {self._num_actions} vs "
                    f"{action_space_size}"
                )

            episodes = manifest.get("episodes", [])
            for episode_meta in episodes:
                npz_path = dir_path / episode_meta["file"]
                if not npz_path.exists():
                    raise FileNotFoundError(f"Episode file '{npz_path}' not found.")
                data = np.load(str(npz_path), allow_pickle=False)
                observations = data["observations"]
                actions = data["actions"]
                if observations.shape[0] != actions.shape[0] + 1:
                    raise ValueError(
                        f"Episode '{npz_path}' has mismatched observations/actions lengths "
                        f"({observations.shape[0]} vs {actions.shape[0]})."
                    )
                obs_chunks.append(observations[:-1].astype(np.uint8, copy=False))
                action_chunks.append(actions.astype(np.int64, copy=False))
                start = total_samples
                total_samples += actions.shape[0]
                self._episode_slices.append((start, total_samples))

        if not obs_chunks:
            self._observations = np.empty((0, 64, 64, 3), dtype=np.uint8)
            self._actions = np.empty((0,), dtype=np.int64)
            if self._num_actions is None:
                self._num_actions = 0
        else:
            self._observations = np.concatenate(obs_chunks, axis=0)
            self._actions = np.concatenate(action_chunks, axis=0)


def train_val_split(
    dataset: TrajectoryDataset,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split a TrajectoryDataset into train/val Subsets grouped by episode."""
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be in [0, 1].")

    episode_slices = dataset.episode_slices
    num_episodes = len(episode_slices)
    if num_episodes == 0:
        empty = Subset(dataset, [])
        return empty, empty

    rng = np.random.default_rng(seed)
    episode_indices = np.arange(num_episodes)
    rng.shuffle(episode_indices)

    if val_fraction == 0.0:
        num_val = 0
    elif val_fraction == 1.0:
        num_val = num_episodes
    else:
        num_val = max(1, int(round(num_episodes * val_fraction)))
        num_val = min(num_val, num_episodes)

    val_episode_ids = set(episode_indices[:num_val])
    train_indices: list[int] = []
    val_indices: list[int] = []

    for episode_id, (start, end) in enumerate(episode_slices):
        target = val_indices if episode_id in val_episode_ids else train_indices
        target.extend(range(start, end))

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
