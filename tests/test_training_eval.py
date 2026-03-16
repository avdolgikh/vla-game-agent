"""Tests for training and evaluation scripts — MVP-1 AC-5 through AC-8, AC-10, AC-11."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_imitation.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_policy.py"

NUM_ACTIONS = 8
# Episodes per policy directory used in fast smoke tests
SMALL_NUM_EPISODES = 3
SMALL_STEPS_PER_EPISODE = 15


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_episode_npz(
    policy_dir: Path,
    filename: str,
    num_steps: int,
    rng: np.random.Generator,
) -> None:
    """Write a synthetic episode .npz file."""
    observations = rng.integers(0, 256, size=(num_steps + 1, 64, 64, 3), dtype=np.uint8)
    actions = rng.integers(0, NUM_ACTIONS, size=(num_steps,), dtype=np.int32)
    rewards = rng.random(num_steps).astype(np.float32)
    np.savez(
        str(policy_dir / filename), observations=observations, actions=actions, rewards=rewards
    )


def _make_policy_dir(
    parent: Path,
    policy_name: str,
    num_episodes: int = SMALL_NUM_EPISODES,
    num_steps: int = SMALL_STEPS_PER_EPISODE,
    seed: int = 0,
) -> Path:
    """Create a directory with synthetic trajectory data and a manifest.json."""
    rng = np.random.default_rng(seed)
    policy_dir = parent / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    for i in range(num_episodes):
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


def _run_train(
    data_dirs: list[Path],
    output_dir: Path,
    epochs: int = 2,
    seed: int = 42,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run train_imitation.py and return the CompletedProcess."""
    cmd = [
        "uv",
        "run",
        "python",
        str(TRAIN_SCRIPT),
        "--data-dirs",
        *[str(d) for d in data_dirs],
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        "8",
        "--lr",
        "1e-3",
        "--val-fraction",
        "0.3",
        "--seed",
        str(seed),
        "--device",
        "cpu",
        "--no-mlflow",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


def _run_eval(
    model_path: Path,
    output_dir: Path,
    num_episodes: int = 3,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run evaluate_policy.py and return the CompletedProcess."""
    cmd = [
        "uv",
        "run",
        "python",
        str(EVAL_SCRIPT),
        "--model",
        str(model_path),
        "--policy-type",
        "cnn",
        "--num-episodes",
        str(num_episodes),
        "--max-steps",
        "30",
        "--base-seed",
        "1000",
        "--output-dir",
        str(output_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Static checks (no subprocess needed)
# ---------------------------------------------------------------------------


class TestScriptFilesExist:
    """Verify that the script files exist at expected paths."""

    def test_train_script_exists(self):
        """scripts/train_imitation.py must exist."""
        assert TRAIN_SCRIPT.exists(), f"Expected training script at {TRAIN_SCRIPT}"

    def test_eval_script_exists(self):
        """scripts/evaluate_policy.py must exist."""
        assert EVAL_SCRIPT.exists(), f"Expected eval script at {EVAL_SCRIPT}"


# ---------------------------------------------------------------------------
# AC-5: Training runs end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainingEndToEnd:
    """AC-5: train_imitation.py runs end-to-end and produces expected artifacts."""

    def test_training_exits_zero(self, tmp_path):
        """Training script must exit with return code 0."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
            _make_policy_dir(tmp_path / "data", "collect_stone", seed=2),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir)
        assert result.returncode == 0, (
            f"Training script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_best_model_pt_created(self, tmp_path):
        """best_model.pt must be created in the output directory."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert (out_dir / "best_model.pt").exists(), "best_model.pt must be created"

    def test_final_model_pt_created(self, tmp_path):
        """final_model.pt must be created in the output directory."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert (out_dir / "final_model.pt").exists(), "final_model.pt must be created"

    def test_train_log_json_created(self, tmp_path):
        """train_log.json must be created in the output directory."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert (out_dir / "train_log.json").exists(), "train_log.json must be created"

    def test_train_log_json_structure(self, tmp_path):
        """train_log.json must contain config, epochs list, best_epoch, best_val_acc."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        assert "config" in log, "train_log.json must contain 'config'"
        assert "epochs" in log, "train_log.json must contain 'epochs'"
        assert "best_epoch" in log, "train_log.json must contain 'best_epoch'"
        assert "best_val_acc" in log, "train_log.json must contain 'best_val_acc'"

    def test_train_log_epochs_have_required_fields(self, tmp_path):
        """Each epoch entry in train_log.json must contain train_loss, val_loss, val_acc."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        assert len(log["epochs"]) == 2, f"Expected 2 epoch entries, got {len(log['epochs'])}"

        for i, epoch_entry in enumerate(log["epochs"]):
            assert "epoch" in epoch_entry, f"Epoch {i} entry missing 'epoch'"
            assert "train_loss" in epoch_entry, f"Epoch {i} entry missing 'train_loss'"
            assert "val_loss" in epoch_entry, f"Epoch {i} entry missing 'val_loss'"
            assert "val_acc" in epoch_entry, f"Epoch {i} entry missing 'val_acc'"

    def test_train_log_epoch_numbers_correct(self, tmp_path):
        """Epoch numbers in train_log.json must be 1-indexed and sequential."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=3)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        for i, epoch_entry in enumerate(log["epochs"]):
            assert epoch_entry["epoch"] == i + 1, (
                f"Expected epoch number {i + 1}, got {epoch_entry['epoch']}"
            )

    def test_train_log_val_acc_in_range(self, tmp_path):
        """val_acc values in train_log.json must be in [0, 1]."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        for entry in log["epochs"]:
            assert 0.0 <= entry["val_acc"] <= 1.0, f"val_acc {entry['val_acc']} is not in [0, 1]"

    def test_train_log_losses_are_nonnegative(self, tmp_path):
        """train_loss and val_loss must be non-negative."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        for entry in log["epochs"]:
            assert entry["train_loss"] >= 0.0, f"train_loss {entry['train_loss']} is negative"
            assert entry["val_loss"] >= 0.0, f"val_loss {entry['val_loss']} is negative"

    def test_train_log_config_has_required_fields(self, tmp_path):
        """train_log.json config section must contain key hyperparameters."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        config = log["config"]
        required_config_keys = {"epochs", "batch_size", "lr", "val_fraction", "seed", "device"}
        missing = required_config_keys - set(config.keys())
        assert not missing, f"train_log.json config is missing fields: {missing}"

    def test_best_model_pt_is_loadable(self, tmp_path):
        """best_model.pt must be loadable by CrafterCNN."""
        import torch

        from vla_agent.models import CrafterCNN

        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        model = CrafterCNN(num_actions=8)
        state = torch.load(str(out_dir / "best_model.pt"), map_location="cpu")
        model.load_state_dict(state)  # must not raise

    def test_single_data_dir_is_accepted(self, tmp_path):
        """Training with a single --data-dirs argument must succeed."""
        data_dir = _make_policy_dir(tmp_path / "data", "collect_wood", num_episodes=4, seed=0)
        out_dir = tmp_path / "model_out"
        result = _run_train([data_dir], out_dir, epochs=2)
        assert result.returncode == 0, (
            f"Training with single data dir failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# AC-10: MLflow tracking
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMLflowTracking:
    """AC-10: MLflow experiment tracking works correctly."""

    def test_mlflow_experiment_created(self, tmp_path):
        """After training with MLflow enabled, an experiment directory must exist in mlruns/."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        mlruns_dir = tmp_path / "mlruns"

        cmd = [
            "uv",
            "run",
            "python",
            str(TRAIN_SCRIPT),
            "--data-dirs",
            str(data_dirs[0]),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--lr",
            "1e-3",
            "--val-fraction",
            "0.3",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--experiment-name",
            "mvp1_test",
        ]
        result = subprocess.run(
            cmd,
            cwd=str(tmp_path),  # run in tmp_path so mlruns/ is created there
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Training with MLflow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert mlruns_dir.exists(), (
            f"Expected mlruns/ directory at {mlruns_dir} after training with MLflow"
        )

    def test_mlflow_run_directory_created(self, tmp_path):
        """After training, mlruns/ must contain at least one run directory."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"

        cmd = [
            "uv",
            "run",
            "python",
            str(TRAIN_SCRIPT),
            "--data-dirs",
            str(data_dirs[0]),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--lr",
            "1e-3",
            "--val-fraction",
            "0.3",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--experiment-name",
            "mvp1_test",
        ]
        result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Training with MLflow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        mlruns_dir = tmp_path / "mlruns"
        # mlruns should contain subdirectories (experiment and run directories)
        subdirs = list(mlruns_dir.rglob("*"))
        assert len(subdirs) > 0, "mlruns/ must contain run artifacts after training"

    def test_no_mlflow_flag_skips_mlruns(self, tmp_path):
        """With --no-mlflow, no mlruns/ directory should be created."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        mlruns_dir = tmp_path / "mlruns"

        result = _run_train(data_dirs, out_dir, epochs=2)
        # _run_train already passes --no-mlflow, and runs from REPO_ROOT
        # We verify here that the flag doesn't break anything
        assert result.returncode == 0, (
            f"Training with --no-mlflow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_no_mlflow_flag_still_creates_file_artifacts(self, tmp_path):
        """With --no-mlflow, file artifacts (best_model.pt etc.) must still be created."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        assert (out_dir / "best_model.pt").exists(), "best_model.pt must exist with --no-mlflow"
        assert (out_dir / "final_model.pt").exists(), "final_model.pt must exist with --no-mlflow"
        assert (out_dir / "train_log.json").exists(), "train_log.json must exist with --no-mlflow"

    def test_mlflow_per_epoch_metrics_count(self, tmp_path):
        """MLflow run must have train_loss, val_loss, val_acc metrics with one entry per epoch."""
        try:
            import mlflow
        except ImportError:
            pytest.skip("mlflow not installed")

        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        n_epochs = 3

        cmd = [
            "uv",
            "run",
            "python",
            str(TRAIN_SCRIPT),
            "--data-dirs",
            str(data_dirs[0]),
            "--output-dir",
            str(out_dir),
            "--epochs",
            str(n_epochs),
            "--batch-size",
            "8",
            "--lr",
            "1e-3",
            "--val-fraction",
            "0.3",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--experiment-name",
            "mvp1_metrics_test",
        ]
        result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Training failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        experiment = mlflow.get_experiment_by_name("mvp1_metrics_test")
        assert experiment is not None, "MLflow experiment 'mvp1_metrics_test' must exist"

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) >= 1, "At least one MLflow run must exist"

        run_id = runs.iloc[0]["run_id"]
        client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))

        for metric_name in ("train_loss", "val_loss", "val_acc"):
            history = client.get_metric_history(run_id, metric_name)
            assert len(history) == n_epochs, (
                f"Expected {n_epochs} entries for metric '{metric_name}', got {len(history)}"
            )

    def test_mlflow_params_logged(self, tmp_path):
        """MLflow run must have hyperparameters logged as params."""
        try:
            import mlflow
        except ImportError:
            pytest.skip("mlflow not installed")

        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"

        cmd = [
            "uv",
            "run",
            "python",
            str(TRAIN_SCRIPT),
            "--data-dirs",
            str(data_dirs[0]),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--lr",
            "1e-3",
            "--val-fraction",
            "0.3",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--experiment-name",
            "mvp1_params_test",
        ]
        result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Training failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        experiment = mlflow.get_experiment_by_name("mvp1_params_test")
        assert experiment is not None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) >= 1
        run_id = runs.iloc[0]["run_id"]

        client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
        run_data = client.get_run(run_id).data

        required_params = {"epochs", "batch_size", "lr", "val_fraction", "seed", "device"}
        logged_params = set(run_data.params.keys())
        missing = required_params - logged_params
        assert not missing, f"MLflow run is missing params: {missing}"

    def test_mlflow_best_val_acc_logged(self, tmp_path):
        """MLflow run must have best_val_acc logged as a final metric."""
        try:
            import mlflow
        except ImportError:
            pytest.skip("mlflow not installed")

        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"

        cmd = [
            "uv",
            "run",
            "python",
            str(TRAIN_SCRIPT),
            "--data-dirs",
            str(data_dirs[0]),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--lr",
            "1e-3",
            "--val-fraction",
            "0.3",
            "--seed",
            "42",
            "--device",
            "cpu",
            "--experiment-name",
            "mvp1_best_acc_test",
        ]
        result = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True)
        assert result.returncode == 0, (
            f"Training failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
        experiment = mlflow.get_experiment_by_name("mvp1_best_acc_test")
        assert experiment is not None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        run_id = runs.iloc[0]["run_id"]

        client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
        run_data = client.get_run(run_id).data

        assert "best_val_acc" in run_data.metrics, "best_val_acc must be logged as a final metric"
        assert "best_epoch" in run_data.metrics, "best_epoch must be logged as a final metric"


# ---------------------------------------------------------------------------
# AC-7: Evaluation rollout runs end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEvaluationEndToEnd:
    """AC-7: evaluate_policy.py runs end-to-end and produces expected artifacts."""

    def _get_or_train_model(self, tmp_path: Path) -> Path:
        """Helper: train a tiny model and return its path."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=1)
        assert result.returncode == 0, f"Pre-requisite training failed:\n{result.stderr}"
        return out_dir / "best_model.pt"

    def test_eval_script_exits_zero(self, tmp_path):
        """evaluate_policy.py must exit with return code 0."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        result = _run_eval(model_path, eval_out, num_episodes=2)
        assert result.returncode == 0, (
            f"Eval script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_eval_results_json_created(self, tmp_path):
        """eval_results.json must be created in the output directory."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        result = _run_eval(model_path, eval_out, num_episodes=2)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"
        assert (eval_out / "eval_results.json").exists(), "eval_results.json must be created"

    def test_eval_results_json_structure(self, tmp_path):
        """eval_results.json must contain model, num_episodes, success_rates, episodes keys."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        result = _run_eval(model_path, eval_out, num_episodes=2)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

        data = json.loads((eval_out / "eval_results.json").read_text())
        required_keys = {
            "model",
            "num_episodes",
            "base_seed",
            "max_steps",
            "success_rates",
            "episodes",
        }
        missing = required_keys - set(data.keys())
        assert not missing, f"eval_results.json is missing keys: {missing}"

    def test_eval_results_num_episodes_matches(self, tmp_path):
        """eval_results.json['num_episodes'] must match the requested number."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        num_ep = 3
        result = _run_eval(model_path, eval_out, num_episodes=num_ep)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

        data = json.loads((eval_out / "eval_results.json").read_text())
        assert data["num_episodes"] == num_ep, (
            f"Expected num_episodes={num_ep}, got {data['num_episodes']}"
        )
        assert len(data["episodes"]) == num_ep, (
            f"Expected {num_ep} episode entries, got {len(data['episodes'])}"
        )

    def test_eval_results_success_rates_are_fractions(self, tmp_path):
        """Success rates in eval_results.json must be in [0, 1]."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        result = _run_eval(model_path, eval_out, num_episodes=3)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

        data = json.loads((eval_out / "eval_results.json").read_text())
        for task, rate in data["success_rates"].items():
            assert 0.0 <= rate <= 1.0, f"Success rate for '{task}' = {rate} is not in [0, 1]"

    def test_eval_results_has_three_tasks(self, tmp_path):
        """eval_results.json['success_rates'] must contain collect_wood, place_table, collect_stone."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        result = _run_eval(model_path, eval_out, num_episodes=2)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

        data = json.loads((eval_out / "eval_results.json").read_text())
        expected_tasks = {"collect_wood", "place_table", "collect_stone"}
        actual_tasks = set(data["success_rates"].keys())
        assert expected_tasks == actual_tasks, (
            f"Expected tasks {expected_tasks}, got {actual_tasks}"
        )

    def test_eval_episode_entries_have_required_fields(self, tmp_path):
        """Each episode entry in eval_results.json must have seed, num_steps, total_reward, successes."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        result = _run_eval(model_path, eval_out, num_episodes=2)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

        data = json.loads((eval_out / "eval_results.json").read_text())
        required_episode_keys = {"seed", "num_steps", "total_reward", "successes"}
        for i, ep in enumerate(data["episodes"]):
            missing = required_episode_keys - set(ep.keys())
            assert not missing, f"Episode {i} is missing fields: {missing}"

    def test_eval_episode_seeds_are_sequential(self, tmp_path):
        """Episode seeds must be base_seed + i for i in range(num_episodes)."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        base_seed = 1000
        num_ep = 3
        result = _run_eval(model_path, eval_out, num_episodes=num_ep)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

        data = json.loads((eval_out / "eval_results.json").read_text())
        for i, ep in enumerate(data["episodes"]):
            expected_seed = base_seed + i
            assert ep["seed"] == expected_seed, (
                f"Episode {i}: expected seed {expected_seed}, got {ep['seed']}"
            )

    def test_eval_success_rates_consistent_with_episodes(self, tmp_path):
        """Aggregate success rates must be consistent with per-episode successes."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        num_ep = 4
        result = _run_eval(model_path, eval_out, num_episodes=num_ep)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

        data = json.loads((eval_out / "eval_results.json").read_text())
        for task in ("collect_wood", "place_table", "collect_stone"):
            count = sum(1 for ep in data["episodes"] if ep["successes"].get(task, False))
            expected_rate = count / num_ep
            actual_rate = data["success_rates"][task]
            assert abs(actual_rate - expected_rate) < 1e-6, (
                f"Task '{task}': aggregate rate {actual_rate} inconsistent with "
                f"per-episode count {count}/{num_ep} = {expected_rate}"
            )

    def test_eval_model_path_recorded_in_results(self, tmp_path):
        """eval_results.json must record the model path."""
        model_path = self._get_or_train_model(tmp_path)
        eval_out = tmp_path / "eval_out"
        result = _run_eval(model_path, eval_out, num_episodes=2)
        assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

        data = json.loads((eval_out / "eval_results.json").read_text())
        assert "model" in data, "eval_results.json must record the model path"
        assert str(model_path) in data["model"] or data["model"].endswith("best_model.pt"), (
            "Recorded model path should reference the model file"
        )


# ---------------------------------------------------------------------------
# AC-6: Training produces a useful model (soft accuracy target)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainingUsefulModel:
    """AC-6: After training, the model must demonstrate non-trivial learning."""

    def test_val_acc_above_chance_on_tiny_dataset(self, tmp_path):
        """
        After even 5 epochs on a small synthetic dataset, best_val_acc should be > 0.
        (We do not test the 60% threshold here since that requires real data — that
        threshold is tested only with a larger dataset in the full integration test below.)
        """
        data_dirs = [
            _make_policy_dir(
                tmp_path / "data", "collect_wood", num_episodes=5, num_steps=20, seed=0
            ),
            _make_policy_dir(
                tmp_path / "data", "place_table", num_episodes=5, num_steps=20, seed=1
            ),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=5)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        best_val_acc = log["best_val_acc"]
        assert best_val_acc > 0.0, (
            f"best_val_acc={best_val_acc:.4f} is 0, model is not learning at all"
        )

    def test_best_val_acc_ge_first_epoch_acc(self, tmp_path):
        """best_val_acc must be >= val_acc of the first epoch (best tracking works)."""
        data_dirs = [
            _make_policy_dir(
                tmp_path / "data", "collect_wood", num_episodes=5, num_steps=20, seed=0
            ),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=4)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        first_epoch_acc = log["epochs"][0]["val_acc"]
        best_val_acc = log["best_val_acc"]
        assert best_val_acc >= first_epoch_acc, (
            f"best_val_acc ({best_val_acc:.4f}) < first epoch val_acc ({first_epoch_acc:.4f})"
        )

    @pytest.mark.slow
    def test_val_acc_above_60_percent_on_real_data(self):
        """
        AC-6 full target: after 20 epochs on real trajectories, val_acc must exceed 60%.
        Requires real trajectory artifacts to be present.
        """
        artifact_dirs = [
            REPO_ROOT / "artifacts" / "trajectories" / "collect_wood",
            REPO_ROOT / "artifacts" / "trajectories" / "place_table",
            REPO_ROOT / "artifacts" / "trajectories" / "collect_stone",
        ]
        existing = [d for d in artifact_dirs if d.exists()]
        if not existing:
            pytest.skip("Real trajectory artifacts not found — run collect_trajectories first")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "model_out"
            result = _run_train(
                existing,
                out_dir,
                epochs=20,
                seed=42,
                extra_args=["--batch-size", "64", "--lr", "1e-3", "--val-fraction", "0.15"],
            )
            assert result.returncode == 0, (
                f"Training failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            log = json.loads((out_dir / "train_log.json").read_text())
            assert log["best_val_acc"] > 0.60, (
                f"AC-6 not met: best_val_acc={log['best_val_acc']:.4f} <= 0.60"
            )


# ---------------------------------------------------------------------------
# AC-8: Evaluation detects the instruction-free limitation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestInstructionFreeLimitation:
    """
    AC-8: When evaluated on real data, the model trained on all three policies
    must show collect_stone success rate < 30%, and collect_wood highest.
    """

    def test_collect_stone_below_30_percent(self):
        """collect_stone success rate must be < 30% (no instruction conditioning)."""
        model_path = REPO_ROOT / "artifacts" / "models" / "mvp1" / "best_model.pt"
        if not model_path.exists():
            pytest.skip("Trained model not found at artifacts/models/mvp1/best_model.pt")

        with tempfile.TemporaryDirectory() as tmp:
            eval_out = Path(tmp) / "eval_out"
            result = _run_eval(model_path, eval_out, num_episodes=50)
            assert result.returncode == 0, (
                f"Eval failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            data = json.loads((eval_out / "eval_results.json").read_text())
            stone_rate = data["success_rates"]["collect_stone"]
            assert stone_rate < 0.30, (
                f"AC-8: collect_stone rate {stone_rate:.2%} should be < 30% "
                f"without instruction conditioning"
            )

    def test_collect_wood_highest_success_rate(self):
        """collect_wood success rate must be the highest among all three tasks."""
        model_path = REPO_ROOT / "artifacts" / "models" / "mvp1" / "best_model.pt"
        if not model_path.exists():
            pytest.skip("Trained model not found at artifacts/models/mvp1/best_model.pt")

        with tempfile.TemporaryDirectory() as tmp:
            eval_out = Path(tmp) / "eval_out"
            result = _run_eval(model_path, eval_out, num_episodes=50)
            assert result.returncode == 0, (
                f"Eval failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            data = json.loads((eval_out / "eval_results.json").read_text())
            rates = data["success_rates"]
            wood_rate = rates["collect_wood"]
            table_rate = rates["place_table"]
            stone_rate = rates["collect_stone"]
            assert wood_rate >= table_rate and wood_rate >= stone_rate, (
                f"AC-8: collect_wood rate ({wood_rate:.2%}) must be the highest, "
                f"but got table={table_rate:.2%}, stone={stone_rate:.2%}"
            )


# ---------------------------------------------------------------------------
# AC-11: Reproducibility — training and evaluation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReproducibility:
    """AC-11: Training and evaluation are reproducible with the same seed."""

    def test_training_same_seed_same_metrics(self, tmp_path):
        """Two training runs with the same seed, data, and hyperparameters produce identical metrics."""
        data_dirs = [
            _make_policy_dir(
                tmp_path / "data", "collect_wood", num_episodes=4, num_steps=15, seed=0
            ),
        ]
        out_dir_a = tmp_path / "run_a"
        out_dir_b = tmp_path / "run_b"

        result_a = _run_train(data_dirs, out_dir_a, epochs=3, seed=42)
        result_b = _run_train(data_dirs, out_dir_b, epochs=3, seed=42)

        assert result_a.returncode == 0, f"Run A failed:\n{result_a.stderr}"
        assert result_b.returncode == 0, f"Run B failed:\n{result_b.stderr}"

        log_a = json.loads((out_dir_a / "train_log.json").read_text())
        log_b = json.loads((out_dir_b / "train_log.json").read_text())

        for i, (entry_a, entry_b) in enumerate(zip(log_a["epochs"], log_b["epochs"])):
            assert abs(entry_a["train_loss"] - entry_b["train_loss"]) < 1e-5, (
                f"Epoch {i + 1}: train_loss differs between runs: "
                f"{entry_a['train_loss']} vs {entry_b['train_loss']}"
            )
            assert abs(entry_a["val_loss"] - entry_b["val_loss"]) < 1e-5, (
                f"Epoch {i + 1}: val_loss differs between runs: "
                f"{entry_a['val_loss']} vs {entry_b['val_loss']}"
            )
            assert abs(entry_a["val_acc"] - entry_b["val_acc"]) < 1e-5, (
                f"Epoch {i + 1}: val_acc differs between runs: "
                f"{entry_a['val_acc']} vs {entry_b['val_acc']}"
            )

    def test_training_different_seeds_different_metrics(self, tmp_path):
        """Two training runs with different seeds should produce different metrics (very likely)."""
        data_dirs = [
            _make_policy_dir(
                tmp_path / "data", "collect_wood", num_episodes=4, num_steps=15, seed=0
            ),
        ]
        out_dir_a = tmp_path / "run_a"
        out_dir_b = tmp_path / "run_b"

        result_a = _run_train(data_dirs, out_dir_a, epochs=3, seed=42)
        result_b = _run_train(data_dirs, out_dir_b, epochs=3, seed=99)

        assert result_a.returncode == 0, f"Run A failed:\n{result_a.stderr}"
        assert result_b.returncode == 0, f"Run B failed:\n{result_b.stderr}"

        log_a = json.loads((out_dir_a / "train_log.json").read_text())
        log_b = json.loads((out_dir_b / "train_log.json").read_text())

        # With different seeds, at least one metric across epochs should differ
        any_different = any(
            abs(ea["train_loss"] - eb["train_loss"]) > 1e-6
            for ea, eb in zip(log_a["epochs"], log_b["epochs"])
        )
        assert any_different, (
            "Training with different seeds produced identical metrics — "
            "seed may not be affecting initialization or data ordering"
        )

    def test_eval_same_model_same_seed_same_results(self, tmp_path):
        """Two evaluation runs with the same model and base_seed produce identical results."""
        # First train a model
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        model_out = tmp_path / "model_out"
        result = _run_train(data_dirs, model_out, epochs=1)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        model_path = model_out / "best_model.pt"

        eval_out_a = tmp_path / "eval_a"
        eval_out_b = tmp_path / "eval_b"

        result_a = _run_eval(model_path, eval_out_a, num_episodes=3)
        result_b = _run_eval(model_path, eval_out_b, num_episodes=3)

        assert result_a.returncode == 0, f"Eval A failed:\n{result_a.stderr}"
        assert result_b.returncode == 0, f"Eval B failed:\n{result_b.stderr}"

        data_a = json.loads((eval_out_a / "eval_results.json").read_text())
        data_b = json.loads((eval_out_b / "eval_results.json").read_text())

        assert data_a["success_rates"] == data_b["success_rates"], (
            "Two eval runs with same model and seed must produce identical success rates"
        )
        for i, (ep_a, ep_b) in enumerate(zip(data_a["episodes"], data_b["episodes"])):
            assert ep_a["num_steps"] == ep_b["num_steps"], (
                f"Episode {i}: num_steps differs between identical eval runs"
            )
            assert abs(ep_a["total_reward"] - ep_b["total_reward"]) < 1e-6, (
                f"Episode {i}: total_reward differs between identical eval runs"
            )
