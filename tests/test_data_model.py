"""Unit tests for vla_agent.data and vla_agent.models — MVP-1 AC-1 through AC-4, AC-9, AC-11."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

NUM_ACTIONS = 8


def _make_episode_npz(
    tmp_dir: Path,
    filename: str,
    num_steps: int,
    rng: np.random.Generator,
    action_override: np.ndarray | None = None,
) -> int:
    """Write a synthetic episode .npz file and return num_steps."""
    observations = rng.integers(0, 256, size=(num_steps + 1, 64, 64, 3), dtype=np.uint8)
    if action_override is not None:
        actions = action_override
    else:
        actions = rng.integers(0, NUM_ACTIONS, size=(num_steps,), dtype=np.int32)
    rewards = rng.random(num_steps).astype(np.float32)
    path = tmp_dir / filename
    np.savez(str(path), observations=observations, actions=actions, rewards=rewards)
    return num_steps


def _make_policy_dir(
    tmp_path: Path,
    policy_name: str,
    num_episodes: int,
    steps_per_episode: list[int] | None = None,
    seed: int = 0,
) -> Path:
    """
    Create a trajectory directory with manifest.json and episode_NNN.npz files.

    Returns the directory path.
    """
    rng = np.random.default_rng(seed)
    policy_dir = tmp_path / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)

    if steps_per_episode is None:
        steps_per_episode = [20] * num_episodes

    episodes_meta = []
    for i, num_steps in enumerate(steps_per_episode):
        filename = f"episode_{i:03d}.npz"
        _make_episode_npz(policy_dir, filename, num_steps, rng)
        episodes_meta.append(
            {
                "file": filename,
                "seed": 42 + i,
                "success": True,
                "num_steps": num_steps,
                "total_reward": float(rng.random()),
            }
        )

    manifest = {
        "policy": policy_name,
        "instruction": f"do {policy_name}",
        "action_space_size": NUM_ACTIONS,
        "observation_shape": [64, 64, 3],
        "num_episodes": num_episodes,
        "base_seed": 42,
        "success_count": num_episodes,
        "episodes": episodes_meta,
    }
    (policy_dir / "manifest.json").write_text(json.dumps(manifest))
    return policy_dir


# ---------------------------------------------------------------------------
# AC-1: TrajectoryDataset loads data correctly
# ---------------------------------------------------------------------------


class TestTrajectoryDatasetLoading:
    """AC-1: TrajectoryDataset loads data correctly."""

    def test_import_trajectory_dataset(self):
        """TrajectoryDataset must be importable from vla_agent.data."""
        from vla_agent.data import TrajectoryDataset  # noqa: F401

    def test_len_nonzero_single_dir(self, tmp_path):
        """Loading one directory produces a dataset with len > 0."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=2, steps_per_episode=[10, 15]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        assert len(ds) > 0

    def test_len_equals_total_steps_single_dir(self, tmp_path):
        """Dataset length equals sum of num_steps across all episodes (terminal obs excluded)."""
        from vla_agent.data import TrajectoryDataset

        steps = [10, 20, 15]
        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=3, steps_per_episode=steps
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        assert len(ds) == sum(steps)

    def test_len_equals_total_steps_multiple_dirs(self, tmp_path):
        """Dataset length equals combined step counts across multiple directories."""
        from vla_agent.data import TrajectoryDataset

        steps_a = [10, 20]
        steps_b = [15, 25]
        steps_c = [30]
        dir_a = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=2, steps_per_episode=steps_a, seed=0
        )
        dir_b = _make_policy_dir(
            tmp_path, "place_table", num_episodes=2, steps_per_episode=steps_b, seed=1
        )
        dir_c = _make_policy_dir(
            tmp_path, "collect_stone", num_episodes=1, steps_per_episode=steps_c, seed=2
        )
        ds = TrajectoryDataset(data_dirs=[str(dir_a), str(dir_b), str(dir_c)])
        expected = sum(steps_a) + sum(steps_b) + sum(steps_c)
        assert len(ds) == expected

    def test_observation_shape(self, tmp_path):
        """Each sample's observation must have shape (3, 64, 64) — channels-first."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        sample = ds[0]
        assert sample["observation"].shape == (3, 64, 64), (
            f"Expected observation shape (3, 64, 64), got {sample['observation'].shape}"
        )

    def test_observation_dtype_float32(self, tmp_path):
        """Each sample's observation must be float32."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        sample = ds[0]
        assert sample["observation"].dtype == torch.float32, (
            f"Expected float32 observation, got {sample['observation'].dtype}"
        )

    def test_observation_range_zero_to_one(self, tmp_path):
        """Observation values must be in [0, 1] after uint8 → float32 / 255 conversion."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[10]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        for i in range(len(ds)):
            obs = ds[i]["observation"]
            assert float(obs.min()) >= 0.0, f"Sample {i}: observation min < 0"
            assert float(obs.max()) <= 1.0, f"Sample {i}: observation max > 1"

    def test_action_dtype_int64(self, tmp_path):
        """Each sample's action must be a torch.int64 (LongTensor) scalar."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        sample = ds[0]
        assert sample["action"].dtype == torch.int64, (
            f"Expected int64 action dtype, got {sample['action'].dtype}"
        )

    def test_action_is_scalar(self, tmp_path):
        """Action must be a scalar tensor (0-dimensional or shape ())."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        sample = ds[0]
        assert sample["action"].ndim == 0, (
            f"Expected scalar action tensor, got shape {sample['action'].shape}"
        )

    def test_action_range_zero_to_seven(self, tmp_path):
        """Every action must be in [0, 7]."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=2, steps_per_episode=[10, 10]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        for i in range(len(ds)):
            action_val = int(ds[i]["action"])
            assert 0 <= action_val < NUM_ACTIONS, (
                f"Sample {i}: action {action_val} is out of range [0, 7]"
            )

    def test_terminal_observation_excluded(self, tmp_path):
        """Terminal observation (obs[T]) must not appear as a sample."""
        from vla_agent.data import TrajectoryDataset

        num_steps = 5
        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[num_steps]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        # Dataset length must equal num_steps, not num_steps + 1
        assert len(ds) == num_steps, (
            f"Expected {num_steps} samples (terminal obs excluded), got {len(ds)}"
        )

    def test_sample_returns_dict_with_required_keys(self, tmp_path):
        """__getitem__ must return a dict with 'observation' and 'action' keys."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        sample = ds[0]
        assert "observation" in sample, "Sample dict must contain 'observation'"
        assert "action" in sample, "Sample dict must contain 'action'"

    def test_observation_is_float_tensor(self, tmp_path):
        """Observation must be a torch.Tensor (FloatTensor)."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        obs = ds[0]["observation"]
        assert isinstance(obs, torch.Tensor), f"Expected torch.Tensor, got {type(obs)}"

    def test_action_is_long_tensor(self, tmp_path):
        """Action must be a torch.Tensor of long dtype."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        action = ds[0]["action"]
        assert isinstance(action, torch.Tensor), f"Expected torch.Tensor, got {type(action)}"

    def test_transform_applied_to_observation(self, tmp_path):
        """If transform is provided, it must be applied to the observation tensor."""
        from vla_agent.data import TrajectoryDataset

        # Transform that multiplies by 2 — result will be outside [0,1] if transform applied
        def double(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds_no_transform = TrajectoryDataset(data_dirs=[str(policy_dir)])
        ds_with_transform = TrajectoryDataset(data_dirs=[str(policy_dir)], transform=double)

        obs_plain = ds_no_transform[0]["observation"]
        obs_transformed = ds_with_transform[0]["observation"]

        torch.testing.assert_close(obs_transformed, obs_plain * 2.0)

    def test_correct_action_action_pairing(self, tmp_path):
        """obs[t] must be paired with action[t], not action[t+1]."""
        from vla_agent.data import TrajectoryDataset

        rng = np.random.default_rng(99)
        num_steps = 8
        policy_dir = tmp_path / "test_policy"
        policy_dir.mkdir()

        # Create a known action sequence
        known_actions = np.arange(num_steps, dtype=np.int32) % NUM_ACTIONS
        observations = rng.integers(0, 256, size=(num_steps + 1, 64, 64, 3), dtype=np.uint8)
        rewards = np.ones(num_steps, dtype=np.float32)
        np.savez(
            str(policy_dir / "episode_000.npz"),
            observations=observations,
            actions=known_actions,
            rewards=rewards,
        )

        manifest = {
            "policy": "test_policy",
            "instruction": "test",
            "action_space_size": NUM_ACTIONS,
            "observation_shape": [64, 64, 3],
            "num_episodes": 1,
            "base_seed": 0,
            "success_count": 1,
            "episodes": [
                {
                    "file": "episode_000.npz",
                    "seed": 0,
                    "success": True,
                    "num_steps": num_steps,
                    "total_reward": 1.0,
                }
            ],
        }
        (policy_dir / "manifest.json").write_text(json.dumps(manifest))

        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        assert len(ds) == num_steps
        for t in range(num_steps):
            sample = ds[t]
            expected_action = int(known_actions[t])
            actual_action = int(sample["action"])
            assert actual_action == expected_action, (
                f"At step {t}: expected action {expected_action}, got {actual_action}"
            )


# ---------------------------------------------------------------------------
# AC-1: action_counts method
# ---------------------------------------------------------------------------


class TestActionCounts:
    """AC-1 (extended) / spec §1.3: action_counts() returns per-action sample counts."""

    def test_import_action_counts(self, tmp_path):
        """TrajectoryDataset must have an action_counts method."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=1, steps_per_episode=[5]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        assert hasattr(ds, "action_counts"), "TrajectoryDataset must have an action_counts() method"

    def test_action_counts_shape(self, tmp_path):
        """action_counts() must return an ndarray of shape (8,)."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=2, steps_per_episode=[10, 10]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        counts = ds.action_counts()
        assert isinstance(counts, np.ndarray), f"Expected np.ndarray, got {type(counts)}"
        assert counts.shape == (NUM_ACTIONS,), (
            f"Expected shape ({NUM_ACTIONS},), got {counts.shape}"
        )

    def test_action_counts_sum_equals_dataset_length(self, tmp_path):
        """Sum of action_counts() must equal len(dataset)."""
        from vla_agent.data import TrajectoryDataset

        steps = [10, 15, 8]
        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=3, steps_per_episode=steps
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        assert ds.action_counts().sum() == len(ds)

    def test_action_counts_nonnegative(self, tmp_path):
        """All counts in action_counts() must be >= 0."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=2, steps_per_episode=[20, 20]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        assert (ds.action_counts() >= 0).all()

    def test_action_counts_integer_dtype(self, tmp_path):
        """action_counts() must return an integer-type ndarray."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=2, steps_per_episode=[8, 12]
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        counts = ds.action_counts()
        assert counts.dtype.kind in {"i", "u"}, (
            f"Expected integer dtype for action_counts(), got {counts.dtype}"
        )

    def test_action_counts_correct_values(self, tmp_path):
        """action_counts() must match the known distribution of a synthetic episode."""
        from vla_agent.data import TrajectoryDataset

        rng = np.random.default_rng(7)
        policy_dir = tmp_path / "known_dist"
        policy_dir.mkdir()

        # Create actions with a known distribution: all action 3
        num_steps = 12
        fixed_actions = np.full(num_steps, fill_value=3, dtype=np.int32)
        observations = rng.integers(0, 256, size=(num_steps + 1, 64, 64, 3), dtype=np.uint8)
        rewards = np.ones(num_steps, dtype=np.float32)
        np.savez(
            str(policy_dir / "episode_000.npz"),
            observations=observations,
            actions=fixed_actions,
            rewards=rewards,
        )

        manifest = {
            "policy": "known_dist",
            "instruction": "test",
            "action_space_size": NUM_ACTIONS,
            "observation_shape": [64, 64, 3],
            "num_episodes": 1,
            "base_seed": 0,
            "success_count": 1,
            "episodes": [
                {
                    "file": "episode_000.npz",
                    "seed": 0,
                    "success": True,
                    "num_steps": num_steps,
                    "total_reward": 1.0,
                }
            ],
        }
        (policy_dir / "manifest.json").write_text(json.dumps(manifest))

        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        counts = ds.action_counts()

        expected = np.zeros(NUM_ACTIONS, dtype=np.int64)
        expected[3] = num_steps
        np.testing.assert_array_equal(counts, expected)


# ---------------------------------------------------------------------------
# AC-2: Train/val split is episode-level
# ---------------------------------------------------------------------------


class TestTrainValSplit:
    """AC-2: train_val_split is episode-level, deterministic, and respects the fraction."""

    def test_import_train_val_split(self):
        """train_val_split must be importable from vla_agent.data."""
        from vla_agent.data import train_val_split  # noqa: F401

    def test_returns_two_subsets(self, tmp_path):
        """train_val_split must return a 2-tuple of Subset objects."""
        from torch.utils.data import Subset

        from vla_agent.data import TrajectoryDataset, train_val_split

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=10, steps_per_episode=[10] * 10
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        train_sub, val_sub = train_val_split(ds, val_fraction=0.2, seed=42)
        assert isinstance(train_sub, Subset)
        assert isinstance(val_sub, Subset)

    def test_split_covers_all_samples(self, tmp_path):
        """train + val indices must cover all samples exactly once."""
        from vla_agent.data import TrajectoryDataset, train_val_split

        steps = [10] * 10
        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=10, steps_per_episode=steps
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        train_sub, val_sub = train_val_split(ds, val_fraction=0.2, seed=42)

        train_indices = set(train_sub.indices)
        val_indices = set(val_sub.indices)

        assert train_indices.isdisjoint(val_indices), "Train and val indices must not overlap"
        assert train_indices | val_indices == set(range(len(ds))), (
            "Train + val indices must cover all samples"
        )

    def test_no_episode_overlap_between_train_and_val(self, tmp_path):
        """No frame index from a train episode should appear in the val set, and vice versa."""
        from vla_agent.data import TrajectoryDataset, train_val_split

        # Build multiple directories with known episode lengths
        directory_configs = [
            ("collect_wood", [5, 7, 3]),
            ("place_table", [8, 6]),
            ("collect_stone", [4, 9]),
        ]

        data_dirs = []
        all_steps = []
        for idx, (policy_name, steps_list) in enumerate(directory_configs):
            dir_path = _make_policy_dir(
                tmp_path / f"dir_{idx}",
                policy_name,
                num_episodes=len(steps_list),
                steps_per_episode=steps_list,
                seed=idx,
            )
            data_dirs.append(str(dir_path))
            all_steps.extend(steps_list)

        ds = TrajectoryDataset(data_dirs=data_dirs)
        train_sub, val_sub = train_val_split(ds, val_fraction=0.3, seed=42)

        # Reconstruct which global indices belong to each episode
        episode_index_ranges = []
        start = 0
        for s in all_steps:
            episode_index_ranges.append(set(range(start, start + s)))
            start += s

        train_set = set(train_sub.indices)
        val_set = set(val_sub.indices)

        for ep_idx, ep_range in enumerate(episode_index_ranges):
            in_train = bool(ep_range & train_set)
            in_val = bool(ep_range & val_set)
            assert not (in_train and in_val), (
                f"Episode {ep_idx} has frames in both train and val — split is not episode-level"
            )

    def test_split_is_deterministic(self, tmp_path):
        """Same seed must produce identical train/val splits."""
        from vla_agent.data import TrajectoryDataset, train_val_split

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=10, steps_per_episode=[10] * 10
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])

        train_a, val_a = train_val_split(ds, val_fraction=0.2, seed=42)
        train_b, val_b = train_val_split(ds, val_fraction=0.2, seed=42)

        assert train_a.indices == train_b.indices, "Train indices must be identical for same seed"
        assert val_a.indices == val_b.indices, "Val indices must be identical for same seed"

    def test_different_seeds_produce_different_splits(self, tmp_path):
        """Different seeds should (with very high probability) produce different splits."""
        from vla_agent.data import TrajectoryDataset, train_val_split

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=20, steps_per_episode=[10] * 20
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])

        _, val_a = train_val_split(ds, val_fraction=0.2, seed=42)
        _, val_b = train_val_split(ds, val_fraction=0.2, seed=99)

        assert val_a.indices != val_b.indices, (
            "Different seeds should produce different splits (with overwhelming probability)"
        )

    def test_val_fraction_approximately_respected(self, tmp_path):
        """Val fraction should be approximately correct within ±5% tolerance."""
        from vla_agent.data import TrajectoryDataset, train_val_split

        n_episodes = 20
        steps_per_ep = [10] * n_episodes
        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=n_episodes, steps_per_episode=steps_per_ep
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])

        val_fraction = 0.2
        _, val_sub = train_val_split(ds, val_fraction=val_fraction, seed=42)

        actual_val_fraction = len(val_sub) / len(ds)
        assert abs(actual_val_fraction - val_fraction) <= 0.05, (
            f"Val fraction {actual_val_fraction:.3f} deviates from target {val_fraction} by more than 5%"
        )

    def test_train_larger_than_val(self, tmp_path):
        """With val_fraction=0.15, train set must be larger than val set."""
        from vla_agent.data import TrajectoryDataset, train_val_split

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=10, steps_per_episode=[10] * 10
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])
        train_sub, val_sub = train_val_split(ds, val_fraction=0.15, seed=42)

        assert len(train_sub) > len(val_sub), (
            f"Train ({len(train_sub)}) should be larger than val ({len(val_sub)})"
        )


# ---------------------------------------------------------------------------
# AC-3: CrafterCNN forward pass
# ---------------------------------------------------------------------------


class TestCrafterCNNForwardPass:
    """AC-3: CrafterCNN forward pass produces correct shapes and valid action indices."""

    def test_import_crafter_cnn(self):
        """CrafterCNN must be importable from vla_agent.models."""
        from vla_agent.models import CrafterCNN  # noqa: F401

    def test_forward_pass_output_shape(self):
        """Forward pass on (B, 3, 64, 64) input must return (B, 8) logits."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(0)
        model = CrafterCNN(num_actions=8)
        model.eval()
        x = torch.zeros(4, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (4, 8), f"Expected shape (4, 8), got {logits.shape}"

    def test_forward_pass_batch_size_one(self):
        """Forward pass on (1, 3, 64, 64) must return (1, 8)."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(0)
        model = CrafterCNN(num_actions=8)
        model.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 8), f"Expected shape (1, 8), got {logits.shape}"

    def test_forward_pass_output_dtype_float32(self):
        """Output logits must be float32."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(0)
        model = CrafterCNN(num_actions=8)
        model.eval()
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        assert logits.dtype == torch.float32, f"Expected float32 logits, got {logits.dtype}"

    def test_argmax_gives_valid_action(self):
        """argmax of output logits must be in [0, 7]."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(0)
        model = CrafterCNN(num_actions=8)
        model.eval()
        x = torch.rand(8, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        actions = logits.argmax(dim=1)
        assert actions.shape == (8,)
        for i, a in enumerate(actions.tolist()):
            assert 0 <= a < NUM_ACTIONS, f"Batch item {i}: argmax action {a} out of range [0, 7]"

    def test_parameter_count_in_range(self):
        """Parameter count must be between 100K and 1M."""
        from vla_agent.models import CrafterCNN

        model = CrafterCNN(num_actions=8)
        num_params = sum(p.numel() for p in model.parameters())
        assert 100_000 <= num_params <= 1_000_000, (
            f"Expected parameter count in [100K, 1M], got {num_params}"
        )

    def test_custom_num_actions(self):
        """CrafterCNN with num_actions=4 must output shape (B, 4)."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(0)
        model = CrafterCNN(num_actions=4)
        model.eval()
        x = torch.rand(3, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (3, 4), f"Expected shape (3, 4), got {logits.shape}"

    def test_model_is_nn_module(self):
        """CrafterCNN must be a subclass of torch.nn.Module."""
        from vla_agent.models import CrafterCNN

        model = CrafterCNN()
        assert isinstance(model, torch.nn.Module)

    def test_no_activation_after_fc2(self):
        """Output logits must be raw (no softmax/sigmoid applied) — values can be negative."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(0)
        model = CrafterCNN(num_actions=8)
        model.eval()
        # Use random input — raw logits can be negative, but softmax output cannot
        x = torch.randn(16, 3, 64, 64)
        with torch.no_grad():
            logits = model(x)
        # If softmax were applied, all values would be in (0,1). Raw logits can be negative.
        has_negative = (logits < 0).any().item()
        # With random weights and inputs, it is overwhelmingly likely some logit is negative
        assert has_negative, (
            "Expected some negative logit values (raw logits), but all were non-negative — "
            "check that no activation is applied after fc2"
        )


# ---------------------------------------------------------------------------
# AC-4: CrafterCNN is deterministic
# ---------------------------------------------------------------------------


class TestCrafterCNNDeterminism:
    """AC-4: CrafterCNN produces deterministic outputs with same weights and same input."""

    def test_same_input_same_output(self):
        """Same input tensor + same model weights must produce identical output."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(7)
        model = CrafterCNN(num_actions=8)
        model.eval()

        torch.manual_seed(0)
        x = torch.rand(4, 3, 64, 64)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        torch.testing.assert_close(
            out1, out2, msg="Repeated forward passes on same input must be identical"
        )

    def test_same_seed_same_parameters(self):
        """Two CrafterCNN instances initialized with the same torch seed must have identical parameters."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(42)
        model_a = CrafterCNN(num_actions=8)

        torch.manual_seed(42)
        model_b = CrafterCNN(num_actions=8)

        for (name_a, param_a), (name_b, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            torch.testing.assert_close(
                param_a,
                param_b,
                msg=f"Parameter '{name_a}' differs between two models with same seed",
            )

    def test_same_seed_same_forward_output(self):
        """Two models with identical weights produce identical forward outputs on the same input."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(42)
        model_a = CrafterCNN(num_actions=8)
        model_a.eval()

        torch.manual_seed(42)
        model_b = CrafterCNN(num_actions=8)
        model_b.eval()

        torch.manual_seed(0)
        x = torch.rand(4, 3, 64, 64)

        with torch.no_grad():
            out_a = model_a(x)
            out_b = model_b(x)

        torch.testing.assert_close(
            out_a, out_b, msg="Models with same seed must produce identical outputs"
        )

    def test_different_seeds_different_parameters(self):
        """Two CrafterCNN instances with different seeds must have different parameters (with high probability)."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(1)
        model_a = CrafterCNN(num_actions=8)

        torch.manual_seed(2)
        model_b = CrafterCNN(num_actions=8)

        # Check that at least one parameter tensor differs
        any_different = False
        for (_, param_a), (_, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            if not torch.equal(param_a, param_b):
                any_different = True
                break

        assert any_different, "Models with different seeds should have different parameters"


# ---------------------------------------------------------------------------
# AC-9: Saved model is loadable
# ---------------------------------------------------------------------------


class TestModelSaveLoad:
    """AC-9: CrafterCNN can be saved and loaded, producing identical outputs."""

    def test_state_dict_save_and_load(self, tmp_path):
        """Saved state dict can be loaded into a fresh CrafterCNN."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(42)
        model = CrafterCNN(num_actions=8)
        model.eval()

        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(save_path))

        loaded_model = CrafterCNN(num_actions=8)
        loaded_model.load_state_dict(torch.load(str(save_path), map_location="cpu"))
        loaded_model.eval()

        torch.manual_seed(0)
        x = torch.rand(4, 3, 64, 64)
        with torch.no_grad():
            out_original = model(x)
            out_loaded = loaded_model(x)

        torch.testing.assert_close(
            out_original, out_loaded, msg="Loaded model must produce same output as original"
        )

    def test_loaded_model_parameter_count(self, tmp_path):
        """Loaded model must have the same parameter count as the saved model."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(42)
        model = CrafterCNN(num_actions=8)
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(save_path))

        loaded_model = CrafterCNN(num_actions=8)
        loaded_model.load_state_dict(torch.load(str(save_path), map_location="cpu"))

        original_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for p in loaded_model.parameters())
        assert original_params == loaded_params

    def test_loaded_model_state_dict_keys_match(self, tmp_path):
        """Loaded state dict must have identical keys to the original."""
        from vla_agent.models import CrafterCNN

        torch.manual_seed(42)
        model = CrafterCNN(num_actions=8)
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(save_path))

        loaded_state = torch.load(str(save_path), map_location="cpu")
        assert set(loaded_state.keys()) == set(model.state_dict().keys())


# ---------------------------------------------------------------------------
# AC-11: Reproducibility of dataset loading
# ---------------------------------------------------------------------------


class TestDataReproducibility:
    """AC-11 (partial): Dataset loading is deterministic — same files produce same samples."""

    def test_same_dataset_same_samples_at_index(self, tmp_path):
        """Loading the same directory twice produces identical samples at each index."""
        from vla_agent.data import TrajectoryDataset

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=3, steps_per_episode=[10, 10, 10]
        )
        ds1 = TrajectoryDataset(data_dirs=[str(policy_dir)])
        ds2 = TrajectoryDataset(data_dirs=[str(policy_dir)])

        for i in range(len(ds1)):
            torch.testing.assert_close(
                ds1[i]["observation"],
                ds2[i]["observation"],
                msg=f"Sample {i} observation differs between two loads of the same data",
            )
            assert int(ds1[i]["action"]) == int(ds2[i]["action"]), (
                f"Sample {i} action differs between two loads of the same data"
            )

    def test_train_val_split_same_seed_same_indices(self, tmp_path):
        """train_val_split with the same seed always produces the same index sets."""
        from vla_agent.data import TrajectoryDataset, train_val_split

        policy_dir = _make_policy_dir(
            tmp_path, "collect_wood", num_episodes=10, steps_per_episode=[10] * 10
        )
        ds = TrajectoryDataset(data_dirs=[str(policy_dir)])

        results = []
        for _ in range(3):
            train_sub, val_sub = train_val_split(ds, val_fraction=0.2, seed=42)
            results.append((train_sub.indices, val_sub.indices))

        for i in range(1, len(results)):
            assert results[0][0] == results[i][0], (
                "Train indices differ across repeated calls with same seed"
            )
            assert results[0][1] == results[i][1], (
                "Val indices differ across repeated calls with same seed"
            )


# ---------------------------------------------------------------------------
# AC-1 (integration): TrajectoryDataset loads real trajectory files
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrajectoryDatasetIntegration:
    """AC-1 integration: load real artifact files if present."""

    def test_loads_real_trajectories(self):
        """TrajectoryDataset loads real artifact directories when they exist."""
        from vla_agent.data import TrajectoryDataset

        artifact_dirs = [
            "artifacts/trajectories/collect_wood",
            "artifacts/trajectories/place_table",
            "artifacts/trajectories/collect_stone",
        ]
        existing = [d for d in artifact_dirs if Path(d).exists()]
        if not existing:
            pytest.skip("No real trajectory artifacts found — run collect_trajectories first")

        ds = TrajectoryDataset(data_dirs=existing)
        assert len(ds) > 0, "Real trajectory dataset must have at least one sample"

        sample = ds[0]
        assert sample["observation"].shape == (3, 64, 64)
        assert sample["observation"].dtype == torch.float32
        assert (
            0.0 <= float(sample["observation"].min()) <= float(sample["observation"].max()) <= 1.0
        )
        assert sample["action"].dtype == torch.int64
        assert 0 <= int(sample["action"]) < NUM_ACTIONS
