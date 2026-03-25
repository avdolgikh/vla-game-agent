# Spec: MVP-2.1 — 224×224 Input Resize

## Status

Approved

## Goal

Resize input frames from 64×64 to 224×224 before feeding them to ConvNeXt-Tiny in `CrafterVLA`. This aligns input resolution with ConvNeXt's ImageNet training resolution, expanding spatial feature maps from 2×2 to 7×7 before pooling. Already proven in a background experiment to improve val_acc from 51.6% to 61.0%.

---

## Problem Statement

ConvNeXt-Tiny was pretrained on 224×224 ImageNet images. When fed 64×64 Crafter frames, the progressive downsampling collapses spatial dimensions to 2×2 before global average pooling — only 4 spatial positions. At 224×224 this becomes 7×7 (49 positions), giving the frozen backbone meaningful spatial structure to work with. This is not a model capacity issue — it's a resolution alignment issue.

---

## Design

### Change

Add `torchvision.transforms.functional.resize` in `CrafterVLA.forward()` to upscale input images from 64×64 to 224×224 before normalization and backbone forward pass. Use bilinear interpolation (PyTorch default for `resize`).

### File to Modify

`src/vla_agent/models.py` — `CrafterVLA.forward()` method only.

### What NOT to Change

- No changes to `CrafterVLA.__init__()`, `train()`, or any other method.
- No changes to `data.py`, `train_imitation.py`, or `evaluate_policy.py`.
- No changes to `InstructionEncoder`.
- No new dependencies (torchvision is already installed).

---

## Acceptance Criteria

### AC-1: Forward pass resizes to 224×224

`CrafterVLA.forward()` resizes the input image tensor to 224×224 before passing to the vision backbone. The resize uses `torchvision.transforms.functional.resize`.

### AC-2: Input/output shapes unchanged

The model still accepts `(B, 3, 64, 64)` images and `(B, 384)` text embeddings, and outputs `(B, 8)` action logits. The resize is internal — callers see no difference.

### AC-3: Validation accuracy improves

Training on the same data with the same hyperparameters achieves val_acc > 55% (MVP-2 baseline: 51.6%, expected: ~61%).

### AC-4: Evaluation success rates

At least one task achieves a higher success rate than MVP-2's collect_wood baseline (72%).

### AC-5: Backward compatibility

Existing `CrafterCNN` model and `--model-type cnn` path are unaffected. Only the VLA path changes.

---

## Artifact Pipeline

### Training
command: C:/Users/alexe/AppData/Local/Programs/Python/Python311/python.exe scripts/train_imitation.py --model-type vla --data-dirs artifacts/trajectories/collect_wood artifacts/trajectories/place_table artifacts/trajectories/collect_stone --output-dir artifacts/models/mvp2.1 --experiment-name mvp2.1 --epochs 20 --batch-size 64 --lr 1e-3 --seed 42 --device cuda --no-mlflow
required_files:
  - artifacts/models/mvp2.1/best_model.pt
  - artifacts/models/mvp2.1/train_log.json
metrics_file: artifacts/models/mvp2.1/train_log.json
metrics_checks:
  - path: best_val_acc
    op: ">"
    value: 0.55
    label: "AC-3: val_acc above 55%"

### Evaluation
command: C:/Users/alexe/AppData/Local/Programs/Python/Python311/python.exe scripts/evaluate_policy.py --model artifacts/models/mvp2.1/best_model.pt --policy-type vla --num-episodes 50 --base-seed 1000 --output-dir artifacts/eval/mvp2.1 --device cuda
required_files:
  - artifacts/eval/mvp2.1/eval_results.json
metrics_file: artifacts/eval/mvp2.1/eval_results.json
metrics_checks:
  - path: success_rates.collect_wood
    op: ">"
    value: 0.08
    label: "AC-4: collect_wood exceeds MVP-1 baseline"

### Acceptance
summary_file: artifacts/eval/mvp2.1/eval_results.json
all_checks_must_pass: false
min_checks_pass: 1
