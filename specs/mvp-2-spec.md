# Spec: Instruction-Conditioned VLA Policy (MVP-2)

## Status

Approved

## Goal

Add instruction conditioning to the imitation learning pipeline: the model receives both a game frame and a text instruction, and must predict an action that pursues the instructed task. This is the core VLA (Vision-Language-Action) result: same observations, different instructions → different behavior.

The key question MVP-2 answers: **does instruction conditioning enable task-specific behavior that the vision-only baseline (MVP-1) cannot achieve?**

---

## Scope

### In scope

- Instruction text returned by `TrajectoryDataset` (loaded from manifest `"instruction"` field)
- Text encoder: frozen `all-MiniLM-L6-v2` (384-d sentence embeddings) with caching
- Vision encoder: frozen ConvNeXt-Tiny with ImageNet-1K weights (768-d features)
- VLA model class: frozen encoders + trainable MLP action head
- Training script extended with `--model-type vla`
- Evaluation script extended with `--policy-type vla` (per-instruction evaluation)
- MLflow tracking under experiment name `mvp2`
- Artifacts saved to `artifacts/models/mvp2/` and `artifacts/eval/mvp2/`

### Out of scope

- Recurrent / temporal models (still single-frame)
- Reward-based learning, RL fine-tuning
- Hyperparameter search
- Data augmentation (aside from ImageNet normalization for ConvNeXt)
- Multi-GPU / distributed training
- LoRA or partial encoder fine-tuning (fully frozen encoders only)
- New trajectory collection (reuse MVP-0b's 500-episode datasets)

---

## Component Selection Rationale

Based on the pretrained components research (`specs/pretrained-components-for-vla-classifier-research.pdf`):

**Vision encoder: ConvNeXt-Tiny (ImageNet-1K V1)**

- 28.6M params, 768-d output, BSD-3-Clause (TorchVision)
- Native 64×64 support (min_size 32×32) — no resizing, preserves pixel art structure
- CNN architecture — no positional embedding constraints
- TorchVision model zoo — `torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)`

**Text encoder: all-MiniLM-L6-v2**

- 22.7M params, 384-d output, Apache-2.0
- Sentence-level embeddings — designed for short text like "collect wood"
- HuggingFace model id: `sentence-transformers/all-MiniLM-L6-v2`
- Only 3 unique instructions → embeddings precomputed and cached at startup

**Why not CLIP dual encoder (Combination A)?**

- 4× larger frozen footprint (200M vs 51M)
- Requires resize to 224×224 (blurs pixel art via bicubic interpolation)
- Shared embedding space provides no practical benefit with only 3 distinct instructions
- The MLP action head trivially learns cross-modal interaction at this scale

---

## 1. Data Changes

**File:** `src/vla_agent/data.py`

### 1.1 TrajectoryDataset instruction support

The dataset now returns the instruction string with each sample. The instruction comes from each manifest's `"instruction"` field (already present in all MVP-0b manifests: `"collect wood"`, `"place table"`, `"collect stone"`).

**Modified `__getitem__` returns:**

| Key | Type | Description |
|-----|------|-------------|
| `observation` | `torch.FloatTensor` shape `(3, 64, 64)` | Unchanged from MVP-1 |
| `action` | `torch.LongTensor` scalar | Unchanged from MVP-1 |
| `instruction` | `str` | Task instruction (e.g., `"collect wood"`) |

**Loading change:** during `_load_directories`, read the `"instruction"` field from each manifest and store it per episode slice. Each sample's instruction is looked up from its episode's source directory.

If a manifest lacks an `"instruction"` field, use `""` (empty string) as fallback for backward compatibility. This should not happen with MVP-0b data.

### 1.2 Task-instruction mapping

The instruction-to-task mapping is fixed for the project and used by the evaluation script:

| Instruction | Task key | Success condition |
|-------------|----------|-------------------|
| `"collect wood"` | `collect_wood` | `info["inventory"]["wood"] >= 1` |
| `"place table"` | `place_table` | `info["achievements"]["place_table"] >= 1` |
| `"collect stone"` | `collect_stone` | `info["inventory"]["stone"] >= 1` |

Define this mapping as a module-level constant `INSTRUCTION_TASK_MAP` in `evaluate_policy.py` (or wherever it is most natural).

---

## 2. VLA Model

**File:** `src/vla_agent/models.py`

### 2.1 InstructionEncoder

A utility class that encodes instruction strings into fixed-size embeddings using frozen `all-MiniLM-L6-v2`. This is **not** an `nn.Module` — it is a stateful encoder used at training/evaluation startup.

```python
class InstructionEncoder:
    def __init__(self, device: str | torch.device = "cpu"):
        """Load frozen all-MiniLM-L6-v2 and prepare embedding cache."""

    def encode(self, text: str) -> torch.Tensor:
        """Encode one instruction → (384,) float32 tensor on self.device."""

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode multiple instructions → (B, 384) float32 tensor."""

    @property
    def embed_dim(self) -> int:
        """Return embedding dimension (384)."""
```

**Implementation details:**

- Model loaded via `transformers.AutoTokenizer` + `transformers.AutoModel` from `"sentence-transformers/all-MiniLM-L6-v2"`.
- Pooling: mean pooling over token embeddings (standard sentence-transformers approach). Compute attention mask–weighted mean of the last hidden state.
- All encoder parameters set to `requires_grad=False`.
- Forward passes run under `torch.no_grad()`.
- Internal cache: `dict[str, torch.Tensor]` keyed by input string. Each unique string is encoded once; subsequent calls return the cached tensor.
- The encoder model and cached tensors live on the specified `device`.

### 2.2 CrafterVLA

The VLA model: frozen ConvNeXt-Tiny vision encoder + precomputed text embeddings + trainable MLP action head.

```python
class CrafterVLA(nn.Module):
    def __init__(self, text_embed_dim: int = 384, num_actions: int = 8, pretrained: bool = True):
        """
        Args:
            text_embed_dim: dimension of precomputed text embeddings (384 for MiniLM)
            num_actions: number of output action classes (8)
            pretrained: if True, load ImageNet-1K V1 weights for ConvNeXt-Tiny;
                        if False, use random initialization (for unit tests)
        """

    def forward(self, image: torch.Tensor, text_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, 64, 64) float32, values in [0, 1]
            text_embed: (B, text_embed_dim) float32, precomputed instruction embeddings
        Returns:
            logits: (B, num_actions) float32
        """
```

**Architecture breakdown:**

| Component | Details |
|-----------|---------|
| Vision encoder | ConvNeXt-Tiny backbone, classification head removed, outputs 768-d pooled feature vector |
| ImageNet normalization | Applied inside `forward()`: `(image - mean) / std` using ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] stored as registered buffers |
| Text input | Precomputed 384-d embedding tensor (from `InstructionEncoder`, not re-encoded here) |
| Fusion | `torch.cat([vision_features, text_embed], dim=1)` → 1152-d vector |
| Action head | `Linear(1152, 256) → ReLU → Linear(256, num_actions)` |

**Vision encoder construction:**

Use `torchvision.models.convnext_tiny(weights=...)` and extract the 768-d feature representation by removing the final classification linear layer. The exact extraction method is left to the implementer, but the contract is: input `(B, 3, 64, 64)` → output `(B, 768)`. Verify with a shape assertion.

**Freezing contract:**

- All vision encoder parameters: `requires_grad = False`
- Vision encoder forward runs under `torch.no_grad()` to avoid storing activation tensors for backward pass
- Only `action_head` parameters are trainable
- ImageNet normalization buffers are not parameters (registered buffers)

**Parameter counts (approximate):**

| Component | Params | Trainable |
|-----------|--------|-----------|
| ConvNeXt-Tiny (frozen) | ~28.6M | No |
| Action head | ~297K | Yes |
| Total | ~28.9M | ~297K |

### 2.3 CrafterCNN unchanged

The existing `CrafterCNN` class is not modified. MVP-1 code continues to work identically.

---

## 3. Training Script Changes

**File:** `scripts/train_imitation.py`

### 3.1 New CLI argument

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-type` | `str` choices `[cnn, vla]` | `cnn` | Model architecture to train |

All existing arguments remain unchanged with the same defaults. When `--model-type cnn`, behavior is byte-for-byte identical to pre-MVP-2 code.

Default `--output-dir` remains `artifacts/models/mvp1`. When running MVP-2, pass `--output-dir artifacts/models/mvp2` explicitly. Default `--experiment-name` remains `mvp1`; pass `--experiment-name mvp2` for MVP-2.

### 3.2 VLA training flow (when `--model-type vla`)

1. Load `TrajectoryDataset` from `--data-dirs` (unchanged — instruction strings now included in samples).
2. Split into train/val via `train_val_split` (unchanged — episode-level split).
3. Initialize `InstructionEncoder` on the training device.
4. Collect all unique instruction strings from the dataset and pre-encode them into a `dict[str, torch.Tensor]` cache.
5. Initialize `CrafterVLA(pretrained=True)` and move to device.
6. Create optimizer: `Adam(model.action_head.parameters(), lr=lr)` — **only action head params**, not the full model.
7. For each epoch:
   a. **Train:** for each batch, extract instruction strings from the batch dict, look up cached text embeddings, stack into a tensor, call `model(obs, text_embeds)`, compute cross-entropy loss, backprop (gradients flow only through action head), step optimizer.
   b. **Validate:** same forward pass logic under `torch.no_grad()`, compute loss and accuracy.
   c. Log metrics, save best checkpoint.
8. Save final checkpoint and `train_log.json`.
9. Log to MLflow if enabled.

### 3.3 Console output

Same format as MVP-1:

```
Epoch 05/20 | train_loss=0.842 | val_loss=1.023 | val_acc=0.671 | best=true
```

### 3.4 Saved artifacts

Same structure as MVP-1:

| File | Description |
|------|-------------|
| `best_model.pt` | Full model state dict of best model (includes frozen ConvNeXt-Tiny weights) |
| `final_model.pt` | Full model state dict after last epoch |
| `train_log.json` | Per-epoch metrics with config including `"model_type": "vla"` |

`train_log.json` config section includes `"model_type": "vla"` (or `"cnn"` for MVP-1 runs).

---

## 4. Evaluation Script Changes

**File:** `scripts/evaluate_policy.py`

### 4.1 New CLI support

| Argument | Change |
|----------|--------|
| `--policy-type` | Add `"vla"` to choices (existing: `"cnn"`) |

When `--policy-type vla`, the script loads `CrafterVLA` instead of `CrafterCNN` and runs per-instruction evaluation.

### 4.2 VLA evaluation flow

When `--policy-type vla`:

1. Load `CrafterVLA` from `--model` checkpoint.
2. Initialize `InstructionEncoder` on the evaluation device.
3. For each instruction in `["collect wood", "place table", "collect stone"]`:
   a. Encode instruction → `text_embed` tensor.
   b. Run `num_episodes` episodes:
      - Create `CrafterEnv(seed=base_seed + i)`.
      - At each step: convert obs to tensor, call `model(obs_tensor, text_embed)`, take `argmax`.
      - After episode: check if the **instructed** task succeeded (using the task-instruction mapping).
   c. Record per-episode results.
4. Compute per-instruction success rates.
5. Save results and print summary.

**Seed strategy:** all 3 instructions use the **same** seed range (`base_seed` through `base_seed + num_episodes - 1`). This means the same 50 environments are used for each instruction, making per-instruction comparison fair and enabling direct comparison with MVP-1 (which also used `--base-seed 1000`).

### 4.3 Console output

```
Evaluating instruction: "collect wood"
  Episode 01/50 | seed=1000 | steps=87 | reward=1.0 | success=true
  ...
Evaluating instruction: "place table"
  Episode 01/50 | seed=1000 | steps=145 | reward=2.0 | success=false
  ...
Evaluating instruction: "collect stone"
  ...

Done. 150 episodes evaluated (50 per instruction).
  "collect wood":   34/50 (68.0%)
  "place table":    25/50 (50.0%)
  "collect stone":  12/50 (24.0%)
Saved: artifacts/eval/mvp2/eval_results.json
```

### 4.4 Saved artifacts

`eval_results.json`:

```json
{
  "model": "artifacts/models/mvp2/best_model.pt",
  "model_type": "vla",
  "num_episodes_per_instruction": 50,
  "base_seed": 1000,
  "max_steps": 300,
  "success_rates": {
    "collect_wood": 0.68,
    "place_table": 0.50,
    "collect_stone": 0.24
  },
  "instructions": {
    "collect wood": {
      "task": "collect_wood",
      "success_rate": 0.68,
      "successes": 34,
      "episodes": [
        {"seed": 1000, "num_steps": 87, "total_reward": 1.0, "success": true}
      ]
    },
    "place table": {
      "task": "place_table",
      "success_rate": 0.50,
      "successes": 25,
      "episodes": [...]
    },
    "collect stone": {
      "task": "collect_stone",
      "success_rate": 0.24,
      "successes": 12,
      "episodes": [...]
    }
  }
}
```

The top-level `success_rates` dict mirrors MVP-1's format for easy automated comparison.

---

## 5. Package Layout

Modified files:

```
src/vla_agent/
  data.py              # MODIFIED: __getitem__ returns instruction string
  models.py            # MODIFIED: add InstructionEncoder, CrafterVLA
scripts/
  train_imitation.py   # MODIFIED: add --model-type vla support
  evaluate_policy.py   # MODIFIED: add --policy-type vla support
```

New dependencies: `torchvision>=0.15`, `transformers>=4.30`. Add via `uv add torchvision transformers`.

---

## 6. Acceptance Criteria

### AC-1: TrajectoryDataset returns instruction strings

- Each sample dict contains an `"instruction"` key with a non-empty string value.
- The instruction matches the manifest's `"instruction"` field for that sample's source directory.
- Existing `"observation"` and `"action"` fields are unchanged in type, shape, and values.
- All existing MVP-1 tests pass without modification (backward compatible).

### AC-2: InstructionEncoder produces correct embeddings

- `encode("collect wood")` returns a tensor of shape `(384,)` dtype `float32`.
- `encode_batch(["collect wood", "place table"])` returns shape `(2, 384)`.
- Same input string → identical output tensor (deterministic, cached).
- `embed_dim` property returns `384`.
- Different instruction strings produce different embeddings (cosine similarity < 1.0).

### AC-3: CrafterVLA forward pass

- Input `(B, 3, 64, 64)` image + `(B, 384)` text_embed → output `(B, 8)` float32 logits.
- Vision encoder is ConvNeXt-Tiny (when `pretrained=True`, loaded with ImageNet-1K V1 weights).
- `argmax` of output gives a valid action index in `[0, 7]`.

### AC-4: Only action head is trainable

- Trainable parameter count (`requires_grad=True`) is < 500K.
- Total parameter count is > 28M (includes frozen ConvNeXt-Tiny).
- After `loss.backward()`, vision encoder parameters have `grad is None` (no gradients propagated).

### AC-5: VLA training runs end-to-end

- `scripts/train_imitation.py --model-type vla` completes without error on a small subset (2 epochs, ≤ 3 episodes per policy).
- `best_model.pt`, `final_model.pt`, and `train_log.json` are created in the output directory.
- `train_log.json` config contains `"model_type": "vla"` and per-epoch metrics with `train_loss`, `val_loss`, `val_acc`.
- Val accuracy increases from epoch 1 to the best epoch (model learns).

### AC-6: VLA training produces a useful model

- After training on full 500-episode datasets for 20 epochs, val action-prediction accuracy is **above 60%**.
- Soft target — comparable to or better than MVP-1's 71.6%.

### AC-7: VLA evaluation runs end-to-end

- `scripts/evaluate_policy.py --policy-type vla` completes without error for ≥ 5 episodes per instruction.
- `eval_results.json` is created with per-instruction success rates.
- Results include per-episode details for each instruction.

### AC-8: Instruction conditioning improves task-specific behavior

- When evaluated with per-task instructions (50 episodes per instruction, same seeds as MVP-1), at least 2 of 3 instructed-task success rates exceed the corresponding MVP-1 baselines:
  - MVP-1 `collect_wood`: 8%
  - MVP-1 `place_table`: 84%
  - MVP-1 `collect_stone`: 10%
- This is the core project result: language grounding enables task-specific behavior that the vision-only model cannot achieve.

### AC-9: Backward compatibility

- `--model-type cnn` (default) produces identical training behavior to pre-MVP-2 code.
- `--policy-type cnn` (default) produces identical evaluation behavior to pre-MVP-2 code.
- All existing MVP-1 unit and integration tests pass without modification.

### AC-10: Model save/load roundtrip

- `CrafterVLA` state dict saved via `torch.save()` and loaded into a fresh `CrafterVLA()` via `load_state_dict()` produces identical outputs for the same inputs.

### AC-11: MLflow tracking

- After VLA training, an MLflow experiment exists with all hyperparameters (including `model_type`), per-epoch metrics, and final best metrics logged.
- `--no-mlflow` disables MLflow logging without affecting other behavior.

### AC-12: Reproducibility

- Training with the same seed, data, and device produces identical `train_log.json` metrics.
- Evaluation with the same seed and model produces identical `eval_results.json`.

---

## 7. Testing Notes

### Unit tests (no Crafter, no GPU, no pretrained weight downloads)

Tests must not download pretrained weights from the internet. Use these strategies:

**TrajectoryDataset instruction support:**
- Create synthetic `.npz` files with test manifests that include `"instruction"` fields.
- Verify `__getitem__` returns `"instruction"` key with the correct string.
- Verify `"observation"` and `"action"` are unchanged.

**InstructionEncoder:**
- Mock the HuggingFace model loading (`AutoTokenizer.from_pretrained`, `AutoModel.from_pretrained`) to avoid network access.
- With a mock encoder that returns fixed-shape tensors: verify `encode()` returns shape `(384,)`, `encode_batch()` returns `(B, 384)`, caching works (same string → same tensor object), `embed_dim` returns 384.

**CrafterVLA:**
- Use `CrafterVLA(pretrained=False)` for unit tests — this initializes ConvNeXt-Tiny with random weights, no download needed.
- Verify forward pass: `(B, 3, 64, 64)` + `(B, 384)` → `(B, 8)`.
- Verify only action head parameters have `requires_grad=True`.
- Verify trainable param count < 500K, total param count > 28M.
- Verify vision encoder gradients are `None` after backward.

**Model save/load:**
- Save and load `CrafterVLA(pretrained=False)` state dict, verify identical output.

### Integration tests (`@pytest.mark.integration`)

These tests may download pretrained weights on first run:

- `TrajectoryDataset` loads real trajectory files from `artifacts/trajectories/` and returns correct instruction strings.
- VLA training end-to-end: 2 epochs on a small subset with `--model-type vla`, verify output files.
- VLA evaluation end-to-end: 3 episodes per instruction with `--policy-type vla`, verify `eval_results.json`.

---

## 8. Expected Results & Interpretation

**Offline accuracy (val set):** 60–80%. The VLA model has the same frame information as MVP-1 plus instruction conditioning. Accuracy should be comparable or slightly better because the text embedding provides a disambiguating signal.

**Online success rates (rollout with per-task instructions):**

| Instruction | MVP-1 rate | MVP-2 expected | Why |
|-------------|-----------|----------------|-----|
| collect wood | 8% | 30–60% | Instruction directs model toward wood-chopping behavior instead of defaulting to place_table |
| place table | 84% | 50–80% | May decrease — MVP-1's high rate came from always pursuing this task; MVP-2 only pursues it when instructed |
| collect stone | 10% | 15–40% | Most challenging (longest dependency chain); instruction helps but single-frame input limits multi-step planning |

**Key result:** MVP-2 success rates should be **more balanced** across tasks. MVP-1 collapses to one dominant behavior; MVP-2 follows instructions. The gap quantifies the value of language grounding for goal-directed behavior.

**If MVP-2 does not improve over MVP-1:** this would suggest either (a) frozen ConvNeXt features don't transfer well to Crafter's pixel art, or (b) the action head needs more capacity. Ablation candidates in that case: resize images to 224×224, try CLIP dual encoder (research Combination A), or unfreeze the last ConvNeXt stage.

---

## 9. Non-Functional Requirements

- **UV only** — `uv sync`, `uv run pytest`, `uv run python`. Never pip.
- **New dependencies:** `torchvision>=0.15`, `transformers>=4.30`. Add via `uv add torchvision transformers`.
- **No global state.** All randomness flows from explicit seeds.
- **No print in library code** (`src/`). Only scripts print.
- **Seed reproducibility.** Same seed + same data + same device = identical results.
- **Imports:** absolute from `vla_agent` (e.g., `from vla_agent.models import CrafterVLA`).
- **Artifacts directory:** `artifacts/models/mvp2/` for checkpoints, `artifacts/eval/mvp2/` for evaluation results.
- **Pretrained weight caching:** ConvNeXt-Tiny weights are cached by TorchVision; MiniLM weights are cached by HuggingFace Transformers. Both download automatically on first use and are cached for subsequent runs.
- **VRAM budget:** Training must fit in 12 GB VRAM (RTX 4070) with batch size ≥ 16. Frozen encoders use `torch.no_grad()` to avoid storing activation tensors.

---

## 10. How to Run MVP-2

```bash
# Unit tests only (fast, no downloads)
uv run python -m pytest

# Integration tests (may download pretrained weights on first run)
uv run python -m pytest -m integration

# Full training
uv run python scripts/train_imitation.py \
    --model-type vla \
    --data-dirs artifacts/trajectories/collect_wood artifacts/trajectories/place_table artifacts/trajectories/collect_stone \
    --output-dir artifacts/models/mvp2 \
    --experiment-name mvp2 \
    --epochs 20 --batch-size 64 --lr 1e-3 --seed 42

# Evaluation (50 episodes per instruction = 150 total)
uv run python scripts/evaluate_policy.py \
    --model artifacts/models/mvp2/best_model.pt \
    --policy-type vla \
    --num-episodes 50 --base-seed 1000 \
    --output-dir artifacts/eval/mvp2
```
