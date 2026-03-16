# Spec: Vision-Only Imitation Baseline (MVP-1)

## Status

Approved


## Goal

Train a small CNN policy that maps a single 64×64 RGB frame to one of 8 actions via behavioral cloning on the expert trajectories collected in MVP-0b. This is the first learned component in the project and establishes the full train → evaluate → rollout loop. The model intentionally receives **no text instruction** — it learns a single mixed behavior from all three expert policies. MVP-2 will add instruction conditioning.

The key question MVP-1 answers: **can a small vision model reproduce scripted-expert behavior from pixels alone, and where does it break without task instructions?**

---

## Scope

### In scope

- Trajectory dataset loader (reads `.npz` files, produces PyTorch tensors)
- CNN policy network (`frame → action logits`)
- Training script with cross-entropy loss (behavioral cloning)
- Offline evaluation (action-prediction accuracy on held-out episodes)
- Online evaluation (rollout in CrafterEnv, measure task success rate)
- Training artifacts saved to `artifacts/models/mvp1/`
- MLflow experiment tracking (local file-based backend, no server)

### Out of scope

- Instruction conditioning, text encoder, multi-modal fusion (MVP-2)
- Recurrent / temporal models (single-frame only)
- Reward-based learning, RL fine-tuning
- Hyperparameter search, wandb
- Data augmentation
- Multi-GPU / distributed training

---

## 1. Trajectory Dataset

**File:** `src/vla_agent/data.py`

### 1.1 TrajectoryDataset

A `torch.utils.data.Dataset` that loads expert trajectories from disk and serves `(observation, action)` pairs for behavioral cloning.

```python
class TrajectoryDataset(Dataset):
    def __init__(self, data_dirs: list[str], transform: Callable | None = None):
        """
        Args:
            data_dirs: list of directories, each containing manifest.json + episode_NNN.npz
            transform: optional callable applied to each observation (e.g., normalization)
        """
```

**Loading behavior:**

1. For each directory in `data_dirs`, read `manifest.json`.
2. For each episode in the manifest, load the `.npz` file.
3. Flatten all `(observation[t], action[t])` pairs across all episodes and all directories into a single flat index. Convention: `observation[t]` is paired with `action[t]` (the action taken after seeing that frame). The terminal observation `observation[T]` has no corresponding action and is **excluded**.
4. Store in memory as arrays (the full dataset fits in RAM — 500 episodes × ~50 steps × 64×64×3 ≈ 300 MB).

**`__getitem__` returns:**

| Key | Type | Description |
|-----|------|-------------|
| `observation` | `torch.FloatTensor` shape `(3, 64, 64)` | Channels-first, float32, scaled to `[0, 1]` by dividing by 255 |
| `action` | `torch.LongTensor` scalar | Action index in `[0, 7]` |

If `transform` is provided, it is applied to the observation tensor after the default uint8→float conversion but before returning.

**`__len__`** returns the total number of `(obs, action)` pairs across all loaded episodes.

### 1.2 Dataset Splitting

The dataset supports train/val splitting by episode, not by frame. This prevents data leakage from temporally adjacent frames.

```python
def train_val_split(
    dataset: TrajectoryDataset,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset]:
```

- Groups frame indices by their source episode.
- Shuffles episodes (deterministically via `seed`).
- Assigns the first `(1 - val_fraction)` episodes to train, the rest to val.
- Returns `(train_subset, val_subset)` as `torch.utils.data.Subset` objects.

### 1.3 Action Distribution

Since the three policies have very different episode lengths and action distributions, the dataset loader must also expose per-action sample counts so the training script can compute class weights if needed.

```python
def action_counts(self) -> np.ndarray:
    """Return array of shape (8,) with per-action sample counts."""
```

---

## 2. CNN Policy Network

**File:** `src/vla_agent/models.py`

### 2.1 Architecture: `CrafterCNN`

A minimal CNN suitable for 64×64 inputs. No bells and whistles — just enough capacity to memorize the expert policy.

```python
class CrafterCNN(nn.Module):
    def __init__(self, num_actions: int = 8):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 64, 64) float32, values in [0, 1]
        Returns:
            logits: (B, num_actions) float32
        """
```

**Layer structure:**

| Layer | Type | Details |
|-------|------|---------|
| conv1 | Conv2d | 3 → 32, kernel 8, stride 4, padding 0 → output 15×15 |
| conv2 | Conv2d | 32 → 64, kernel 4, stride 2, padding 0 → output 6×6 |
| conv3 | Conv2d | 64 → 64, kernel 3, stride 1, padding 0 → output 4×4 |
| flatten | Flatten | 64 × 4 × 4 = 1024 |
| fc1 | Linear | 1024 → 256 |
| fc2 | Linear | 256 → num_actions |

Activation: ReLU after each conv and after fc1. No activation after fc2 (raw logits).

**Why this architecture?** It is the standard Nature DQN encoder scaled for 64×64 input. Small enough to train in minutes on CPU, large enough to represent the three scripted behaviors.

### 2.2 Parameter Count

Expected: ~350K parameters. The spec does not prescribe an exact count, but tests will verify order-of-magnitude (100K–1M range).

---

## 3. Training Script

**File:** `scripts/train_imitation.py`

### 3.1 CLI

```bash
uv run python scripts/train_imitation.py \
    --data-dirs artifacts/trajectories/collect_wood artifacts/trajectories/place_table artifacts/trajectories/collect_stone \
    --output-dir artifacts/models/mvp1 \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3 \
    --val-fraction 0.15 \
    --seed 42 \
    --device auto
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dirs` | `str` (nargs=+) | required | Trajectory directories to load |
| `--output-dir` | `str` | `artifacts/models/mvp1` | Where to save model and logs |
| `--epochs` | `int` | `20` | Training epochs |
| `--batch-size` | `int` | `64` | Mini-batch size |
| `--lr` | `float` | `1e-3` | Learning rate (Adam) |
| `--val-fraction` | `float` | `0.15` | Fraction of episodes for validation |
| `--seed` | `int` | `42` | Random seed for splitting and initialization |
| `--device` | `str` | `auto` | `cpu`, `cuda`, or `auto` (pick cuda if available) |
| `--experiment-name` | `str` | `mvp1` | MLflow experiment name |
| `--no-mlflow` | flag | off | Disable MLflow tracking |

### 3.2 Training Loop

1. Load all trajectories into `TrajectoryDataset`.
2. Split into train/val by episode via `train_val_split`.
3. Initialize `CrafterCNN` with the given seed (`torch.manual_seed`).
4. Start an MLflow run (see §3.5).
5. Optimizer: `Adam(lr=lr)`.
6. Loss: `CrossEntropyLoss` (optionally weighted by inverse action frequency if class imbalance is severe — decision left to implementer, but the option must exist via a `--class-weights` flag, default off).
7. For each epoch:
   a. Train: iterate batches, compute loss, backprop, step.
   b. Validate: compute loss and top-1 accuracy on val set.
   c. Log epoch metrics to MLflow.
   d. Print epoch summary (see 3.3).
   e. If val accuracy is best so far, save checkpoint as `best_model.pt`.
8. After all epochs: save final checkpoint as `final_model.pt`.
9. Log final artifacts to MLflow and end the run.

### 3.3 Console Output

Per-epoch line:

```
Epoch 05/20 | train_loss=0.842 | val_loss=1.023 | val_acc=0.671 | best=true
```

Final summary:

```
Done. Best val_acc=0.671 at epoch 5.
Saved: artifacts/models/mvp1/best_model.pt
       artifacts/models/mvp1/final_model.pt
       artifacts/models/mvp1/train_log.json
```

### 3.4 Saved Artifacts

| File | Description |
|------|-------------|
| `best_model.pt` | State dict of best model (by val accuracy) |
| `final_model.pt` | State dict after last epoch |
| `train_log.json` | Per-epoch metrics: `epoch`, `train_loss`, `val_loss`, `val_acc` |

`train_log.json` structure:

```json
{
  "config": {
    "data_dirs": ["..."],
    "epochs": 20,
    "batch_size": 64,
    "lr": 0.001,
    "val_fraction": 0.15,
    "seed": 42,
    "device": "cpu",
    "num_train_samples": 12000,
    "num_val_samples": 2000,
    "num_parameters": 350000
  },
  "epochs": [
    {"epoch": 1, "train_loss": 1.5, "val_loss": 1.4, "val_acc": 0.45},
    ...
  ],
  "best_epoch": 5,
  "best_val_acc": 0.671
}
```

### 3.5 MLflow Integration

The training script uses MLflow for experiment tracking with a **local file-based backend** (no server required). The tracking URI defaults to `mlruns/` in the project root.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--experiment-name` | `str` | `mvp1` | MLflow experiment name |
| `--no-mlflow` | flag | off | Disable MLflow tracking entirely (for tests or quick runs) |

**What is logged:**

| Category | Items |
|----------|-------|
| Parameters | `epochs`, `batch_size`, `lr`, `val_fraction`, `seed`, `device`, `class_weights`, `num_train_samples`, `num_val_samples`, `num_parameters`, `data_dirs` |
| Metrics (per epoch) | `train_loss`, `val_loss`, `val_acc` (logged with `step=epoch`) |
| Metrics (final) | `best_val_acc`, `best_epoch` |
| Artifacts | `best_model.pt`, `final_model.pt`, `train_log.json` |

**Behavior:**

1. Set tracking URI to `mlruns/` relative to the current working directory.
2. Set or create the experiment by name (`--experiment-name`).
3. Start a run at the beginning of training.
4. Log all hyperparameters as MLflow params.
5. Log `train_loss`, `val_loss`, `val_acc` as MLflow metrics at each epoch step.
6. At the end of training, log `best_val_acc` and `best_epoch` as final metrics.
7. Log the output directory contents (`best_model.pt`, `final_model.pt`, `train_log.json`) as MLflow artifacts.
8. End the run.

If `--no-mlflow` is passed, skip all MLflow calls. The rest of the training script behavior (console output, file artifacts) is unchanged regardless of this flag.

---

## 4. Evaluation Script

**File:** `scripts/evaluate_policy.py`

This script rolls out the trained CNN policy in the actual CrafterEnv and measures task success rates — the ground-truth measure of whether imitation learning works.

### 4.1 CLI

```bash
uv run python scripts/evaluate_policy.py \
    --model artifacts/models/mvp1/best_model.pt \
    --policy-type cnn \
    --num-episodes 50 \
    --max-steps 300 \
    --base-seed 1000 \
    --output-dir artifacts/eval/mvp1
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | `str` | required | Path to saved model state dict |
| `--policy-type` | `str` | `cnn` | Policy type (`cnn` for MVP-1) |
| `--num-episodes` | `int` | `50` | Episodes to evaluate |
| `--max-steps` | `int` | `300` | Max steps per episode |
| `--base-seed` | `int` | `1000` | Base seed (offset from training seeds to avoid overlap) |
| `--output-dir` | `str` | `artifacts/eval/mvp1` | Output directory |

### 4.2 Rollout Behavior

For each episode:

1. Create `CrafterEnv(seed=base_seed + i)`.
2. `obs, info = env.reset()`
3. At each step:
   - Convert `obs` to tensor: `(64,64,3) uint8 → (1,3,64,64) float32 / 255`.
   - Forward through model, get logits.
   - Select action: `argmax(logits)` (greedy, deterministic).
   - `obs, reward, terminated, truncated, info = env.step(action)`
   - Break if `terminated` or `truncated` or `steps >= max_steps`.
4. After episode: check each task's success condition against final `info`:
   - `collect_wood`: `info["inventory"]["wood"] >= 1`
   - `place_table`: `info["achievements"].get("place_table", 0) >= 1`
   - `collect_stone`: `info["inventory"]["stone"] >= 1`

### 4.3 Console Output

Per-episode line:

```
Episode 01/50 | seed=1000 | steps=87 | reward=1.0 | wood=1 table=0 stone=0
```

Final summary:

```
Done. 50 episodes evaluated.
  collect_wood:  34/50 (68.0%)
  place_table:   12/50 (24.0%)
  collect_stone:  3/50 (6.0%)
Saved: artifacts/eval/mvp1/eval_results.json
```

### 4.4 Saved Artifacts

`eval_results.json`:

```json
{
  "model": "artifacts/models/mvp1/best_model.pt",
  "num_episodes": 50,
  "base_seed": 1000,
  "max_steps": 300,
  "success_rates": {
    "collect_wood": 0.68,
    "place_table": 0.24,
    "collect_stone": 0.06
  },
  "episodes": [
    {
      "seed": 1000,
      "num_steps": 87,
      "total_reward": 1.0,
      "successes": {"collect_wood": true, "place_table": false, "collect_stone": false}
    }
  ]
}
```

---

## 5. Package Layout

New and modified files:

```
src/vla_agent/
  data.py              # NEW: TrajectoryDataset, train_val_split
  models.py            # NEW: CrafterCNN
scripts/
  train_imitation.py   # NEW: training script
  evaluate_policy.py   # NEW: rollout evaluation
```

New dependency: `mlflow`. Add via `uv add mlflow`. PyTorch (`torch>=2.0`) is already in `pyproject.toml`.

---

## 6. Acceptance Criteria

### AC-1: TrajectoryDataset loads data correctly

- Loading 3 policy directories produces a dataset with `len > 0`.
- Each sample has `observation` of shape `(3, 64, 64)` dtype `float32` in `[0, 1]`.
- Each sample has `action` of dtype `int64` (long) in `[0, 7]`.
- Total sample count equals the sum of all episode step counts across all loaded directories.
- Terminal observations (last frame of each episode) are excluded.

### AC-2: Train/val split is episode-level

- No frame from a train episode appears in the val set.
- No frame from a val episode appears in the train set.
- Split is deterministic: same seed produces identical splits.
- Val fraction is approximately respected (±5% due to episode granularity).

### AC-3: CrafterCNN forward pass

- Input `(B, 3, 64, 64)` float32 → output `(B, 8)` float32.
- Model parameter count is between 100K and 1M.
- `argmax` of output gives a valid action index in `[0, 7]`.

### AC-4: CrafterCNN is deterministic

- Same input tensor + same model weights → identical output (on same device).
- Two models initialized with the same seed have identical parameters.

### AC-5: Training runs end-to-end

- `scripts/train_imitation.py` completes without error on a small subset (e.g., 2 epochs, 3 episodes per policy).
- `best_model.pt`, `final_model.pt`, and `train_log.json` are created in the output directory.
- `train_log.json` contains per-epoch entries with `train_loss`, `val_loss`, `val_acc`.
- Val accuracy increases from epoch 1 to the best epoch (model learns something).

### AC-6: Training produces a useful model

- After training on full 500-episode datasets for 20 epochs, val action-prediction accuracy is **above 60%**.
- This is a soft target — the model may not reach expert performance without instruction conditioning, but it must demonstrably learn from the data.

### AC-7: Evaluation rollout runs end-to-end

- `scripts/evaluate_policy.py` completes without error for 10 episodes.
- `eval_results.json` is created with per-episode results and aggregate success rates.
- Success rates are plausible (the model does something, not random-level 0%).

### AC-8: Evaluation detects the instruction-free limitation

- When trained on all three policies mixed, the model's `collect_stone` success rate is expected to be **low** (<30%), because without an instruction signal the model cannot distinguish which task to pursue.
- `collect_wood` success rate should be highest (trees are common, wood-chopping dominates the action distribution).
- This asymmetry is the motivating result for MVP-2 (instruction conditioning).

### AC-9: Saved model is loadable

- `CrafterCNN` can be instantiated and loaded from `best_model.pt` via `model.load_state_dict(torch.load(...))`.
- The loaded model produces the same outputs as the model at save time.

### AC-10: MLflow tracking works

- After a training run, an MLflow experiment named `mvp1` exists in `mlruns/`.
- The run contains all hyperparameters logged as params.
- The run contains `train_loss`, `val_loss`, `val_acc` metrics with per-epoch step history.
- The run contains `best_val_acc` and `best_epoch` as final metrics.
- The run contains `best_model.pt`, `final_model.pt`, `train_log.json` as artifacts.
- `--no-mlflow` flag disables all MLflow logging without affecting other training behavior.

### AC-11: Reproducibility

- Training with the same seed, data, and hyperparameters produces identical `train_log.json` metrics (on the same device).
- Evaluation with the same seed and model produces identical `eval_results.json`.

---

## 7. Testing Notes

### Unit tests (no Crafter, no GPU)

- `TrajectoryDataset`: load synthetic `.npz` files (create small ones in test fixtures), verify shapes, dtypes, length, exclusion of terminal frames.
- `train_val_split`: verify episode-level separation, determinism, approximate fraction.
- `action_counts`: verify counts match known distribution from fixtures.
- `CrafterCNN`: forward pass shape, parameter count range, deterministic initialization.
- Model save/load roundtrip: save state dict, load into fresh model, verify identical output.

### MLflow tests (no Crafter, no GPU, no server)

- Training with MLflow enabled creates an experiment and run with expected params, metrics, and artifacts.
- Training with `--no-mlflow` produces no `mlruns/` directory changes but still writes file artifacts normally.
- Per-epoch metrics have correct step indices.

### Integration tests (`@pytest.mark.integration`)

- `TrajectoryDataset` loads real trajectory files from `artifacts/trajectories/` (if present).
- Training script end-to-end: 2 epochs on a small subset, verify output files exist and are well-formed.
- Evaluation script end-to-end: load a (randomly initialized) model, run 3 episodes, verify `eval_results.json`.

---

## 8. Expected Results & Interpretation

This section documents what we expect to see and why — it serves as the interpretive frame for the results.

**Offline accuracy (val set):** 60–80%. The three tasks have overlapping visual contexts but different optimal actions. Without an instruction signal, the model learns a compromise policy weighted by the action distribution in the training data.

**Online success rates (rollout):**

| Task | Expected range | Why |
|------|---------------|-----|
| collect_wood | 40–70% | Trees are everywhere; moving toward trees and chopping is the dominant pattern in training data |
| place_table | 10–30% | Requires a multi-step sequence (chop 2 trees, then place); model has no memory across frames |
| collect_stone | 0–15% | Longest dependency chain; near-impossible without task-specific intent |

The key takeaway: **a vision-only single-frame model partially works for simple tasks but fundamentally cannot handle multi-step goals or task disambiguation.** This motivates MVP-2 (instruction conditioning).

---

## 9. Non-Functional Requirements

- **UV only** — `uv sync`, `uv run pytest`, `uv run python`. Never pip.
- **No global state.** All randomness flows from explicit seeds.
- **No print in library code** (`src/`). Only scripts print.
- **One new dependency:** `mlflow` for experiment tracking. PyTorch and NumPy are already available.
- **Seed reproducibility.** Same seed + same data + same device = identical results.
- **Imports:** absolute from `vla_agent` (e.g., `from vla_agent.models import CrafterCNN`).
- **Artifacts directory:** `artifacts/models/mvp1/` for checkpoints, `artifacts/eval/mvp1/` for evaluation results, `mlruns/` for MLflow tracking data.
