# Spec: MVP-2.3 — Trainable Vision Encoder (Domain Adaptation)

## Status

Approved

## Goal

Replace the frozen ConvNeXt vision backbone with a trainable lightweight CNN in `CrafterVLA`. The frozen ConvNeXt was pretrained on ImageNet (natural photos) and operates on a fundamentally different visual domain than Crafter (2D pixel art). MVP-1's trainable CNN achieved 71.6% val_acc vs the best frozen ConvNeXt configuration at 60.9% — the domain gap is the primary bottleneck. A trainable CNN learns Crafter-specific visual features, operates at native 64×64 (no resize), and trains faster.

---

## Problem Statement

Frozen ConvNeXt features are suboptimal for Crafter's pixel art domain:

| Config | Vision | val_acc | collect_wood | place_table |
|--------|--------|---------|-------------|-------------|
| MVP-1 | Trainable CNN, 64×64 | 71.6% | 8% | 84% |
| MVP-2 | Frozen ConvNeXt, 64×64 | 51.6% | 72% | 22% |
| MVP-2.1 | Frozen ConvNeXt, 224×224 | 60.9% | 48% | 12% |
| MVP-2.2 | Frozen ConvNeXt, 224×224, 4-frame | 55.0% | 58% | 22% |

MVP-1 proved a trainable CNN learns better Crafter features (71.6% val_acc). MVP-2 proved text conditioning enables task-specific behavior. MVP-2.2 proved frame stacking helps multi-step tasks. MVP-2.3 combines all three: trainable vision + text conditioning + temporal context.

---

## Design

### New Vision Backbone: Trainable CNN

Use the same 3-layer convolutional architecture as `CrafterCNN` (proven in MVP-1), followed by a linear projection to a 256-d vision feature vector:

```
Conv2d(3, 32, kernel_size=8, stride=4) → ReLU       # 64×64 → 15×15
Conv2d(32, 64, kernel_size=4, stride=2) → ReLU      # 15×15 → 6×6
Conv2d(64, 64, kernel_size=3, stride=1) → ReLU      # 6×6 → 4×4
Flatten                                               # → 1024
Linear(1024, 256) → ReLU                             # → 256-d vision feature
```

This produces a 256-d vision feature vector per frame. All CNN parameters are **trainable** (not frozen).

### No Resize, No ImageNet Normalization

The CNN operates at native 64×64. No `TF.resize()` to 224×224 and no ImageNet mean/std normalization — these were only needed for the frozen ConvNeXt.

### Fusion and Action Head

Concatenate the 256-d vision feature with the 384-d text embedding → 640-d input to a 2-layer action head:

```
Linear(640, 256) → ReLU → Linear(256, 8)
```

The 2-layer head (vs 3-layer in MVP-2.2) is sufficient for the smaller fusion dimension.

### Frame Stacking (Retained from MVP-2.2)

Same temporal processing: encode each of N=4 frames independently through the CNN, producing N vectors of 256-d. Mean-pool across the temporal dimension → single 256-d representation. This preserves the temporal context that improved place_table from 12% to 22%.

### Text Encoder (Unchanged)

Frozen `all-MiniLM-L6-v2` producing 384-d embeddings. No changes.

### Parameterization via `vision_type`

Add a `vision_type` parameter to `CrafterVLA.__init__()`:
- `vision_type="convnext"` (default): existing frozen ConvNeXt behavior (backward compatible)
- `vision_type="cnn"`: new trainable CNN backbone

This keeps one model class with configurable vision, avoiding class proliferation.

### New Model Type: `vla-cnn`

Add `--model-type vla-cnn` to training and evaluation scripts. This selects `CrafterVLA(vision_type="cnn", ...)`. The optimizer trains **all parameters with `requires_grad=True`** (CNN + action head), not just the action head.

### Trainable Parameter Count

| Component | Params |
|-----------|--------|
| Conv layers (3) | ~76K |
| Linear(1024→256) | ~262K |
| Action head (640→256→8) | ~166K |
| **Total trainable** | **~504K** |
| Text encoder (frozen) | ~22M |

Compare: MVP-2.2 had ~723K trainable (action head only) + ~28M frozen ConvNeXt + ~22M frozen text.

---

## Files to Modify

### `src/vla_agent/models.py`

1. Add `vision_type` parameter to `CrafterVLA.__init__()` (default `"convnext"` for backward compatibility).
2. When `vision_type == "cnn"`:
   - Build a `nn.Sequential` CNN backbone: 3 conv layers (same architecture as `CrafterCNN.conv1/conv2/conv3`) + Flatten + Linear(1024, 256) + ReLU.
   - Store as `self.vision_cnn`.
   - Set `self.vision_dim = 256`.
   - Do NOT register ImageNet normalization buffers.
   - Do NOT freeze the CNN (it must be trainable).
   - Action head: `Linear(640, 256) → ReLU → Linear(256, 8)`.
3. When `vision_type == "convnext"`:
   - Existing behavior unchanged (frozen ConvNeXt, 224×224 resize, ImageNet normalization, 3-layer action head).
4. Store `self.vision_type = vision_type`.
5. Modify `forward()`:
   - When `vision_type == "cnn"`: skip resize and ImageNet normalization. Run frames through `self.vision_cnn` (without `torch.no_grad()`). Mean-pool if frame stacking.
   - When `vision_type == "convnext"`: existing behavior (resize, normalize, frozen backbone under `torch.no_grad()`).
6. Modify `train()` override: when `vision_type == "cnn"`, the CNN should follow standard train/eval mode (no special handling needed — StochasticDepth was a ConvNeXt issue).

### `scripts/train_imitation.py`

1. Add `vla-cnn` to `--model-type` choices.
2. In model construction: when `model_type == "vla-cnn"`, create `CrafterVLA(vision_type="cnn", num_frames=num_frames, ...)`.
3. Optimizer: when `model_type == "vla-cnn"`, optimize all parameters with `requires_grad=True` (not just `action_head.parameters()`).
4. Allow `--num-frames` for `vla-cnn` model type (same as `vla`).
5. Checkpoint metadata: include `vision_type: "cnn"` alongside existing `model_type`, `num_actions`, `num_frames`.

### `scripts/evaluate_policy.py`

1. Add `vla-cnn` to `--policy-type` choices.
2. In model loading: when `policy_type == "vla-cnn"`, create `CrafterVLA(vision_type="cnn", pretrained=False, ...)`.
3. Read `vision_type` from checkpoint metadata if available (fallback to CLI inference from `policy_type`).
4. Frame buffer handling: same as `vla` — use `num_frames` from metadata or CLI.

### What NOT to Change

- `CrafterCNN` — the standalone CNN model, unaffected.
- `CrafterEnv` — observation production unchanged.
- `InstructionEncoder` — text encoding unchanged.
- `TrajectoryDataset` — data pipeline unchanged (frame stacking already works generically).
- Existing `--model-type vla` and `--model-type cnn` paths — fully backward compatible.

---

## Acceptance Criteria

### AC-1: Vision type parameter

`CrafterVLA(vision_type="cnn")` constructs the trainable CNN backbone with `vision_dim=256`.

### AC-2: CNN architecture

The CNN backbone has 3 conv layers (same kernel sizes and strides as `CrafterCNN`) followed by Flatten + Linear(1024, 256) + ReLU.

### AC-3: No resize for CNN

`CrafterVLA(vision_type="cnn").forward(image, text)` does NOT resize the input to 224×224.

### AC-4: CNN parameters are trainable

When `vision_type="cnn"`, the CNN backbone parameters have `requires_grad=True`. The text encoder remains frozen.

### AC-5: Action head for CNN

Action head when `vision_type="cnn"`: `Linear(640, 256) → ReLU → Linear(256, 8)`.

### AC-6: Frame stacking works with CNN

`CrafterVLA(vision_type="cnn", num_frames=4).forward(image, text)` accepts `image` of shape `(B, 4, 3, 64, 64)` and produces `(B, 8)` logits with mean-pooled vision features.

### AC-7: Backward compatibility

`CrafterVLA()` and `CrafterVLA(vision_type="convnext")` behave identically to MVP-2.2 (frozen ConvNeXt, 224×224 resize, 3-layer action head).

### AC-8: Model type CLI

`--model-type vla-cnn` is accepted by `train_imitation.py` and `--policy-type vla-cnn` by `evaluate_policy.py`.

### AC-9: Validation accuracy

Training with `--model-type vla-cnn --num-frames 4` achieves val_acc > 60% (MVP-2.2 baseline: 55.0%, MVP-1 CNN baseline: 71.6%).

### AC-10: Evaluation improvement

At least one task achieves a higher success rate than MVP-2.2 baselines (collect_wood: 58%, place_table: 22%, collect_stone: 0%).

---

## Artifact Pipeline

### Training
command: uv run python scripts/train_imitation.py --model-type vla-cnn --data-dirs artifacts/trajectories/collect_wood artifacts/trajectories/place_table artifacts/trajectories/collect_stone --output-dir artifacts/models/mvp2.3 --experiment-name mvp2.3 --epochs 20 --batch-size 64 --lr 1e-3 --seed 42 --device cuda --no-mlflow --num-frames 4
required_files:
  - artifacts/models/mvp2.3/best_model.pt
  - artifacts/models/mvp2.3/train_log.json
metrics_file: artifacts/models/mvp2.3/train_log.json
metrics_checks:
  - path: best_val_acc
    op: ">"
    value: 0.60
    label: "AC-9: val_acc above 60%"

### Evaluation
command: uv run python scripts/evaluate_policy.py --model artifacts/models/mvp2.3/best_model.pt --policy-type vla-cnn --num-episodes 50 --base-seed 1000 --output-dir artifacts/eval/mvp2.3 --device cuda --num-frames 4
required_files:
  - artifacts/eval/mvp2.3/eval_results.json
metrics_file: artifacts/eval/mvp2.3/eval_results.json
metrics_checks:
  - path: success_rates.collect_wood
    op: ">"
    value: 0.08
    label: "AC-10: collect_wood exceeds MVP-1 baseline"

### Acceptance
summary_file: artifacts/eval/mvp2.3/eval_results.json
all_checks_must_pass: false
min_checks_pass: 1
