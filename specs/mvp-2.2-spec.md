# Spec: MVP-2.2 — Frame Stacking (Temporal Context)

## Status

Approved

## Goal

Add 4-frame temporal context to `CrafterVLA` using per-frame encoding + mean pooling (Option B from the research spec). The model currently sees a single frame and cannot distinguish states that look similar but differ in history (e.g., "approaching tree" vs "just chopped tree"). Frame stacking gives the action head access to recent visual history, enabling temporal reasoning for multi-step tasks like `place_table` and `collect_stone`.

---

## Problem Statement

Single-frame input limits performance on multi-step tasks. `place_table` requires: collect wood → approach placement location → place. Without temporal context, the model cannot know it already has wood. `collect_stone` requires finding and approaching stone — motion direction is invisible from one frame. MVP-2.1 achieved val_acc=60.9% with collect_wood=48%, place_table=12%, collect_stone=0%. Temporal context should unlock meaningful improvement on the latter two tasks.

---

## Design

### Architecture: Per-Frame Encoding + Mean Pooling

Encode each of the last N=4 frames independently through the frozen ConvNeXt backbone, producing N vectors of 768-d. Mean-pool across the temporal dimension to get a single 768-d representation. Concatenate with the 384-d text embedding (unchanged) → 1152-d input to a wider action head.

This preserves the frozen ConvNeXt contract — no architecture changes to the vision encoder.

### Wider Action Head

Upgrade from `1152→256→8` (~297K params) to `1152→512→256→8` (~460K params). The richer temporal input benefits from a slightly larger head.

### Data Pipeline: Frame Sequences

The dataset currently returns single `(observation, action, instruction)` tuples. Change it to return a sequence of N=4 consecutive frames for each timestep, with zero-padding at episode boundaries (first few timesteps of each episode get padded with black frames).

### Evaluation: Rolling Frame Buffer

The evaluation loop currently feeds one frame per step. Add a rolling buffer of the last 4 frames, initialized with zeros on episode reset.

---

## Files to Modify

### `src/vla_agent/data.py`

1. Add `num_frames` parameter to `TrajectoryDataset.__init__()` (default=1 for backward compatibility).
2. Modify `__getitem__()`: when `num_frames > 1`, return `observation` as shape `(N, 3, 64, 64)` instead of `(3, 64, 64)`. Look back `num_frames-1` steps within the same episode; zero-pad if near episode start.
3. Use `_episode_slices` to determine episode boundaries for correct padding.

### `src/vla_agent/models.py`

1. Add `num_frames` parameter to `CrafterVLA.__init__()` (default=1).
2. Store `self.num_frames = num_frames`.
3. Widen action head: `Linear(1152, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 8)`.
4. Modify `forward()`:
   - Accept `image` as either `(B, 3, H, W)` (single frame, backward compat) or `(B, N, 3, H, W)` (frame stack).
   - When 5D: reshape to `(B*N, 3, H, W)`, run through resize + normalize + backbone, reshape back to `(B, N, 768)`, mean-pool to `(B, 768)`.
   - When 4D and `num_frames == 1`: existing behavior (backward compat).

### `scripts/train_imitation.py`

1. Add `--num-frames` CLI arg (default=1).
2. Pass `num_frames` to `TrajectoryDataset` constructor.
3. Pass `num_frames` to `CrafterVLA` constructor.

### `scripts/evaluate_policy.py`

1. Add `--num-frames` CLI arg (default=1).
2. Read `num_frames` from saved model checkpoint metadata (fallback to CLI arg).
3. Maintain a rolling frame buffer (deque of length N). On reset: fill with zeros. On step: append new frame, drop oldest.
4. Stack buffer into `(1, N, 3, H, W)` tensor for model input.

### What NOT to Change

- `CrafterEnv` — observation production unchanged.
- `InstructionEncoder` — text encoding unchanged.
- ConvNeXt backbone — remains frozen, same weights.
- ImageNet normalization — applied per-frame inside `forward()`.
- `CrafterCNN` model and `--model-type cnn` path — unaffected.

---

## Acceptance Criteria

### AC-1: Frame stack shape

`TrajectoryDataset` with `num_frames=4` returns `observation` tensor of shape `(4, 3, 64, 64)`.

### AC-2: Episode boundary padding

For timesteps near episode start (index < 3 within episode), the returned frame stack is zero-padded for missing history frames.

### AC-3: Model accepts stacked frames

`CrafterVLA(num_frames=4).forward(image, text)` accepts `image` of shape `(B, 4, 3, 64, 64)` and produces `(B, 8)` logits.

### AC-4: Backward compatibility

`CrafterVLA(num_frames=1)` still accepts `(B, 3, 64, 64)` single-frame input and works identically to MVP-2.1.

### AC-5: Wider action head

Action head has 3 linear layers: `1152→512→256→8`.

### AC-6: Validation accuracy

Training with `num_frames=4` achieves val_acc > 55% (MVP-2.1 baseline: 60.9%, conservative threshold allows for variance with new architecture).

### AC-7: Evaluation improvement

At least one task achieves a higher success rate than MVP-2.1 baselines (collect_wood: 48%, place_table: 12%, collect_stone: 0%).

---

## Artifact Pipeline

### Training
command: C:/Users/alexe/AppData/Local/Programs/Python/Python311/python.exe scripts/train_imitation.py --model-type vla --data-dirs artifacts/trajectories/collect_wood artifacts/trajectories/place_table artifacts/trajectories/collect_stone --output-dir artifacts/models/mvp2.2 --experiment-name mvp2.2 --epochs 20 --batch-size 64 --lr 1e-3 --seed 42 --device cuda --no-mlflow --num-frames 4
required_files:
  - artifacts/models/mvp2.2/best_model.pt
  - artifacts/models/mvp2.2/train_log.json
metrics_file: artifacts/models/mvp2.2/train_log.json
metrics_checks:
  - path: best_val_acc
    op: ">"
    value: 0.55
    label: "AC-6: val_acc above 55%"

### Evaluation
command: C:/Users/alexe/AppData/Local/Programs/Python/Python311/python.exe scripts/evaluate_policy.py --model artifacts/models/mvp2.2/best_model.pt --policy-type vla --num-episodes 50 --base-seed 1000 --output-dir artifacts/eval/mvp2.2 --device cuda --num-frames 4
required_files:
  - artifacts/eval/mvp2.2/eval_results.json
metrics_file: artifacts/eval/mvp2.2/eval_results.json
metrics_checks:
  - path: success_rates.collect_wood
    op: ">"
    value: 0.08
    label: "AC-7: collect_wood exceeds MVP-1 baseline"

### Acceptance
summary_file: artifacts/eval/mvp2.2/eval_results.json
all_checks_must_pass: false
min_checks_pass: 1
