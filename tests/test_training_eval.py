"""Tests for training and evaluation scripts — MVP-1 AC-5 through AC-8, AC-10, AC-11."""

from __future__ import annotations

import json
import subprocess
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_imitation.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_policy.py"

NUM_ACTIONS = 8
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
    *,
    action_signal: bool = False,
) -> None:
    """Write a synthetic episode .npz file."""
    if action_signal:
        actions = np.arange(num_steps, dtype=np.int32) % NUM_ACTIONS
        observations = np.empty((num_steps + 1, 64, 64, 3), dtype=np.uint8)
        for step, action in enumerate(actions):
            intensity = (action * 32) % 256
            noise = rng.integers(0, 16, size=(64, 64, 3), dtype=np.int32)
            observations[step] = np.clip(intensity + noise, 0, 255).astype(np.uint8)
        observations[-1] = np.zeros((64, 64, 3), dtype=np.uint8)
    else:
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
    *,
    action_signal: bool = False,
) -> Path:
    """Create a directory with synthetic trajectory data and a manifest.json."""
    rng = np.random.default_rng(seed)
    policy_dir = parent / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    for i in range(num_episodes):
        filename = f"episode_{i:03d}.npz"
        _make_episode_npz(policy_dir, filename, num_steps, rng, action_signal=action_signal)
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


def _run_train_mlflow(
    data_dirs: list[Path],
    output_dir: Path,
    *,
    epochs: int = 2,
    seed: int = 42,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run train_imitation.py with MLflow enabled and return the CompletedProcess."""
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
# MLflow helper utilities
# ---------------------------------------------------------------------------


def _snapshot_mlruns() -> tuple[Path, bool, set[str]]:
    """Snapshot mlruns/ state so tests can clean up after themselves."""
    mlruns_dir = REPO_ROOT / "mlruns"
    existed_before = mlruns_dir.exists()
    existing_names = {entry.name for entry in mlruns_dir.iterdir()} if existed_before else set()
    return mlruns_dir, existed_before, existing_names


def _mlruns_new_entries(snapshot: tuple[Path, bool, set[str]]) -> set[str]:
    mlruns_dir, _, existing_names = snapshot
    if not mlruns_dir.exists():
        return set()
    return {entry.name for entry in mlruns_dir.iterdir()} - existing_names


def _cleanup_mlruns(snapshot: tuple[Path, bool, set[str]]) -> None:
    mlruns_dir, existed_before, existing_names = snapshot
    if not mlruns_dir.exists():
        return
    for entry in list(mlruns_dir.iterdir()):
        if entry.name in existing_names:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    if not existed_before and not any(mlruns_dir.iterdir()):
        mlruns_dir.rmdir()


def _mlruns_tracking_uri(mlruns_dir: Path) -> str:
    """Return a proper file URI for the mlruns directory (avoids bare Windows paths)."""
    return mlruns_dir.resolve().as_uri()


def _require_mlflow():
    """Fail fast if MLflow is unavailable for the integration tests."""
    try:
        import mlflow
    except ImportError:
        pytest.fail("mlflow must be installed to run MLflow tracking tests", pytrace=False)
    return mlflow


# ---------------------------------------------------------------------------
# Unit tests (no subprocess)
# ---------------------------------------------------------------------------


class TestScriptFilesExist:
    """Verify that the script files exist at expected paths."""

    def test_train_script_exists(self):
        assert TRAIN_SCRIPT.exists(), f"Expected training script at {TRAIN_SCRIPT}"

    def test_eval_script_exists(self):
        assert EVAL_SCRIPT.exists(), f"Expected eval script at {EVAL_SCRIPT}"


# ---------------------------------------------------------------------------
# AC-5: Training end-to-end (single consolidated test)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainingEndToEnd:
    """AC-5: train_imitation.py runs and produces correct artifacts."""

    def test_training_artifacts_and_log_structure(self, tmp_path):
        """Train 2 epochs, verify all output files, log structure, and config fields."""
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
            _make_policy_dir(tmp_path / "data", "collect_stone", seed=2),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, (
            f"Training script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        # AC-5: files exist
        assert (out_dir / "best_model.pt").exists()
        assert (out_dir / "final_model.pt").exists()
        assert (out_dir / "train_log.json").exists()

        # AC-5: log structure
        log = json.loads((out_dir / "train_log.json").read_text())
        assert "config" in log
        assert "epochs" in log
        assert "best_epoch" in log
        assert "best_val_acc" in log
        assert len(log["epochs"]) == 2

        # AC-5: epoch entries have required fields
        for i, entry in enumerate(log["epochs"]):
            assert entry["epoch"] == i + 1
            assert "train_loss" in entry and entry["train_loss"] >= 0
            assert "val_loss" in entry and entry["val_loss"] >= 0
            assert 0.0 <= entry["val_acc"] <= 1.0

        # AC-5: config has required fields (spec §3.4)
        config = log["config"]
        for key in (
            "epochs",
            "batch_size",
            "lr",
            "val_fraction",
            "seed",
            "device",
            "data_dirs",
            "num_train_samples",
            "num_val_samples",
            "num_parameters",
        ):
            assert key in config, f"train_log.json config missing '{key}'"

    def test_class_weights_flag(self, tmp_path):
        """--class-weights flag is accepted and recorded in config."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2, extra_args=["--class-weights"])
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        assert log["config"].get("class_weights") not in (None, False, "none")

    def test_best_model_is_loadable(self, tmp_path):
        """AC-9: best_model.pt loads into CrafterCNN without error."""
        import torch

        from vla_agent.models import CrafterCNN

        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=2)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        model = CrafterCNN(num_actions=8)
        state = torch.load(str(out_dir / "best_model.pt"), map_location="cpu")
        model.load_state_dict(state)


# ---------------------------------------------------------------------------
# AC-10: MLflow tracking (2 consolidated tests)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMLflowTracking:
    """AC-10: MLflow experiment tracking works correctly."""

    def test_mlflow_full_integration(self, tmp_path):
        """Training with MLflow logs experiment, params, per-epoch metrics, final metrics, and artifacts."""
        mlflow = _require_mlflow()

        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        n_epochs = 3
        snapshot = _snapshot_mlruns()
        mlruns_dir, _, _ = snapshot
        experiment_name = f"mvp1_test_{uuid.uuid4().hex[:8]}"

        try:
            result = _run_train_mlflow(
                data_dirs,
                out_dir,
                epochs=n_epochs,
                extra_args=["--experiment-name", experiment_name],
            )
            assert result.returncode == 0, (
                f"Training with MLflow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            assert mlruns_dir.exists()

            mlflow.set_tracking_uri(_mlruns_tracking_uri(mlruns_dir))
            experiment = mlflow.get_experiment_by_name(experiment_name)
            assert experiment is not None, f"MLflow experiment '{experiment_name}' must exist"

            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            assert len(runs) >= 1
            run_id = runs.iloc[0]["run_id"]
            client = mlflow.tracking.MlflowClient(tracking_uri=_mlruns_tracking_uri(mlruns_dir))
            run_data = client.get_run(run_id).data

            # Params
            required_params = {
                "epochs",
                "batch_size",
                "lr",
                "val_fraction",
                "seed",
                "device",
                "class_weights",
                "num_train_samples",
                "num_val_samples",
                "num_parameters",
                "data_dirs",
            }
            missing_params = required_params - set(run_data.params.keys())
            assert not missing_params, f"MLflow run missing params: {missing_params}"

            # Per-epoch metrics
            for metric_name in ("train_loss", "val_loss", "val_acc"):
                history = client.get_metric_history(run_id, metric_name)
                assert len(history) == n_epochs, (
                    f"Expected {n_epochs} entries for '{metric_name}', got {len(history)}"
                )

            # Final metrics
            assert "best_val_acc" in run_data.metrics
            assert "best_epoch" in run_data.metrics

            # Artifacts
            artifacts = {a.path for a in client.list_artifacts(run_id)}
            missing_artifacts = {"best_model.pt", "final_model.pt", "train_log.json"} - artifacts
            assert not missing_artifacts, f"MLflow run missing artifacts: {missing_artifacts}"
        finally:
            _cleanup_mlruns(snapshot)

    def test_no_mlflow_flag(self, tmp_path):
        """--no-mlflow produces no mlruns/ changes but still writes file artifacts."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        out_dir = tmp_path / "model_out"
        snapshot = _snapshot_mlruns()
        mlruns_dir, existed_before, _ = snapshot

        try:
            result = _run_train(data_dirs, out_dir, epochs=2)
            assert result.returncode == 0, f"Training with --no-mlflow failed:\n{result.stderr}"

            assert not _mlruns_new_entries(snapshot), "mlruns/ should not change with --no-mlflow"
            if not existed_before:
                assert not mlruns_dir.exists()

            assert (out_dir / "best_model.pt").exists()
            assert (out_dir / "final_model.pt").exists()
            assert (out_dir / "train_log.json").exists()
        finally:
            _cleanup_mlruns(snapshot)


# ---------------------------------------------------------------------------
# AC-7: Evaluation end-to-end (single consolidated test)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEvaluationEndToEnd:
    """AC-7: evaluate_policy.py runs and produces correct artifacts."""

    def test_eval_artifacts_and_structure(self, tmp_path):
        """Evaluate a tiny model, verify eval_results.json structure and consistency."""
        # Train a tiny model first
        data_dirs = [
            _make_policy_dir(tmp_path / "data", "collect_wood", seed=0),
            _make_policy_dir(tmp_path / "data", "place_table", seed=1),
        ]
        model_out = tmp_path / "model_out"
        result = _run_train(data_dirs, model_out, epochs=1)
        assert result.returncode == 0, f"Pre-requisite training failed:\n{result.stderr}"
        model_path = model_out / "best_model.pt"

        eval_out = tmp_path / "eval_out"
        num_ep = 3
        result = _run_eval(model_path, eval_out, num_episodes=num_ep)
        assert result.returncode == 0, (
            f"Eval script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        assert (eval_out / "eval_results.json").exists()
        data = json.loads((eval_out / "eval_results.json").read_text())

        # Top-level keys
        for key in ("model", "num_episodes", "base_seed", "max_steps", "success_rates", "episodes"):
            assert key in data, f"eval_results.json missing '{key}'"

        assert data["num_episodes"] == num_ep
        assert len(data["episodes"]) == num_ep

        # Success rates
        expected_tasks = {"collect_wood", "place_table", "collect_stone"}
        assert set(data["success_rates"].keys()) == expected_tasks
        for task, rate in data["success_rates"].items():
            assert 0.0 <= rate <= 1.0

        # Episode entries
        for i, ep in enumerate(data["episodes"]):
            for key in ("seed", "num_steps", "total_reward", "successes"):
                assert key in ep, f"Episode {i} missing '{key}'"
            assert ep["seed"] == 1000 + i

        # Consistency: aggregate rates match per-episode counts
        for task in expected_tasks:
            count = sum(1 for ep in data["episodes"] if ep["successes"].get(task, False))
            expected_rate = count / num_ep
            assert abs(data["success_rates"][task] - expected_rate) < 1e-6


# ---------------------------------------------------------------------------
# AC-6: Training produces a useful model
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainingUsefulModel:
    """AC-6: Model demonstrates learning."""

    def test_model_improves_on_learnable_data(self, tmp_path):
        """best_val_acc must exceed first-epoch val_acc on data with learnable signal."""
        data_dirs = [
            _make_policy_dir(
                tmp_path / "data",
                "collect_wood",
                num_episodes=5,
                num_steps=20,
                seed=0,
                action_signal=True,
            ),
        ]
        out_dir = tmp_path / "model_out"
        result = _run_train(data_dirs, out_dir, epochs=4)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        log = json.loads((out_dir / "train_log.json").read_text())
        first_epoch_acc = log["epochs"][0]["val_acc"]
        best_val_acc = log["best_val_acc"]
        assert best_val_acc > first_epoch_acc, (
            f"best_val_acc ({best_val_acc:.4f}) must exceed first epoch ({first_epoch_acc:.4f})"
        )

    @pytest.mark.slow
    def test_val_acc_above_60_on_real_data(self):
        """AC-6: 20 epochs on real trajectories must exceed 60% val_acc."""
        artifact_dirs = [
            REPO_ROOT / "artifacts" / "trajectories" / p
            for p in ("collect_wood", "place_table", "collect_stone")
        ]
        existing = [d for d in artifact_dirs if d.exists()]
        if not existing:
            pytest.skip("Real trajectory artifacts not found")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "model_out"
            result = _run_train(
                existing,
                out_dir,
                epochs=20,
                seed=42,
                extra_args=["--batch-size", "64", "--lr", "1e-3", "--val-fraction", "0.15"],
            )
            assert result.returncode == 0, f"Training failed:\n{result.stderr}"
            log = json.loads((out_dir / "train_log.json").read_text())
            assert log["best_val_acc"] > 0.60, (
                f"AC-6: best_val_acc={log['best_val_acc']:.4f} <= 0.60"
            )


# ---------------------------------------------------------------------------
# AC-8: Instruction-free limitation (slow, requires real artifacts)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestInstructionFreeLimitation:
    """AC-8: Without instructions, collect_stone < 30% and collect_wood is highest."""

    def test_asymmetric_success_rates(self):
        model_path = REPO_ROOT / "artifacts" / "models" / "mvp1" / "best_model.pt"
        if not model_path.exists():
            pytest.skip("Trained model not found at artifacts/models/mvp1/best_model.pt")

        with tempfile.TemporaryDirectory() as tmp:
            eval_out = Path(tmp) / "eval_out"
            result = _run_eval(model_path, eval_out, num_episodes=50)
            assert result.returncode == 0, f"Eval failed:\n{result.stderr}"

            data = json.loads((eval_out / "eval_results.json").read_text())
            rates = data["success_rates"]

            assert rates["collect_stone"] < 0.30, (
                f"collect_stone rate {rates['collect_stone']:.2%} should be < 30%"
            )
            assert rates["collect_wood"] >= rates["place_table"]
            assert rates["collect_wood"] >= rates["collect_stone"]


# ---------------------------------------------------------------------------
# AC-11: Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReproducibility:
    """AC-11: Same seed + data + hyperparams = identical results."""

    def test_training_reproducible(self, tmp_path):
        """Two training runs with same seed produce identical metrics."""
        data_dirs = [
            _make_policy_dir(
                tmp_path / "data", "collect_wood", num_episodes=4, num_steps=15, seed=0
            ),
        ]
        out_a = tmp_path / "run_a"
        out_b = tmp_path / "run_b"

        result_a = _run_train(data_dirs, out_a, epochs=3, seed=42)
        result_b = _run_train(data_dirs, out_b, epochs=3, seed=42)
        assert result_a.returncode == 0, f"Run A failed:\n{result_a.stderr}"
        assert result_b.returncode == 0, f"Run B failed:\n{result_b.stderr}"

        log_a = json.loads((out_a / "train_log.json").read_text())
        log_b = json.loads((out_b / "train_log.json").read_text())

        for i, (ea, eb) in enumerate(zip(log_a["epochs"], log_b["epochs"])):
            for key in ("train_loss", "val_loss", "val_acc"):
                assert abs(ea[key] - eb[key]) < 1e-5, (
                    f"Epoch {i + 1}: {key} differs: {ea[key]} vs {eb[key]}"
                )

    def test_eval_reproducible(self, tmp_path):
        """Two eval runs with same model and seed produce identical results."""
        data_dirs = [_make_policy_dir(tmp_path / "data", "collect_wood", seed=0)]
        model_out = tmp_path / "model_out"
        result = _run_train(data_dirs, model_out, epochs=1)
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"

        eval_a = tmp_path / "eval_a"
        eval_b = tmp_path / "eval_b"
        result_a = _run_eval(model_out / "best_model.pt", eval_a, num_episodes=3)
        result_b = _run_eval(model_out / "best_model.pt", eval_b, num_episodes=3)
        assert result_a.returncode == 0
        assert result_b.returncode == 0

        data_a = json.loads((eval_a / "eval_results.json").read_text())
        data_b = json.loads((eval_b / "eval_results.json").read_text())
        assert data_a["success_rates"] == data_b["success_rates"]
