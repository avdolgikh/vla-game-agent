# VLA Quality Improvements Spec

**Status:** Draft
**Created:** 2026-03-23
**Predecessor:** MVP-2 (mvp-2-spec.md)

---

## Goal

Systematically improve the VLA game agent's action prediction accuracy and evaluation success rates beyond the MVP-2 baseline. MVP-2 demonstrated that a frozen-encoder VLA architecture works (51.6% val_acc, 72% collect_wood success) but revealed critical weaknesses: catastrophic regression on multi-step tasks (place_table: 84% -> 22%, collect_stone: 10% -> 0%) and a large domain gap between ImageNet features and Crafter pixel art.

This spec defines 10 independent improvement directions, each designed as a self-contained experiment that can be run, measured, and combined with others. The objective is to identify the highest-impact, lowest-effort changes that push toward balanced task success across all three instructions.

## Current Baseline (MVP-2)

| Metric | Value |
|---|---|
| val_acc (8-class) | 51.6% |
| collect_wood success | 72% (MVP-1: 8%, +9x) |
| place_table success | 22% (MVP-1: 84%, -62pp) |
| collect_stone success | 0% (MVP-1: 10%, -10pp) |

### Current Architecture

- **Vision encoder:** Frozen ConvNeXt-Tiny (28.6M params, 768-d output), input 64x64 pixel art frames
- **Text encoder:** Frozen all-MiniLM-L6-v2 (22.7M params, 384-d output)
- **Action head:** Trainable MLP: Linear(1152, 256) -> ReLU -> Linear(256, 8), ~297K params
- **Input:** Single frame (no temporal context)
- **Training:** Behavioral cloning (cross-entropy) on 500 episodes (~33K samples)
- **Hardware constraint:** CPU only (no CUDA on dev machine)

### Key Findings from MVP-2

1. **224x224 resize** (upscaling 64x64 to ConvNeXt's preferred resolution) improved val_acc from 51.6% to 61.0% -- a free 9.4% boost.
2. **StochasticDepth bug:** ConvNeXt drops residual connections in train mode; must keep backbone in eval mode.
3. **Domain gap** is the primary accuracy bottleneck (ImageNet features applied to pixel art).
4. **Instruction conditioning** works powerfully for simple tasks but multi-step tasks need temporal reasoning.
5. **Per-instruction behavior divergence** (72%/22%/0%) proves text embeddings influence decisions.

---

## Improvement 1: Input Resolution (224x224 Resize)

### Summary

Upscale 64x64 input frames to 224x224 using bilinear interpolation before feeding into ConvNeXt. This matches ConvNeXt's native ImageNet training resolution, allowing its learned filters to activate on features at the spatial scale they expect.

### Already Proven

- val_acc: 51.6% -> 61.0% (+9.4pp)
- No architecture changes, no retraining of encoder needed

### What to Change

| Item | Details |
|---|---|
| Files | Model forward() method (add `torchvision.transforms.Resize(224)` in preprocessing) |
| Architecture | No change -- only input preprocessing |
| Training | Same pipeline, just slower per batch due to larger tensors |

### Implementation

Add a single transform in the model's forward pass:

```python
self.resize = torchvision.transforms.Resize(224, antialias=True)
# In forward():
x = self.resize(x)  # (B, 3, 64, 64) -> (B, 3, 224, 224)
```

### Trade-offs

- ~12x more pixels to process (224^2 / 64^2 = 12.25)
- Training and inference wall-clock time increases proportionally on CPU
- Memory usage increases for intermediate ConvNeXt activations

### Expected Impact: HIGH (already proven)

- Passes AC-6 acceptance criterion (>60% val_acc)
- Likely improves evaluation success rates across all tasks
- Risk: essentially zero -- already validated

### Complexity/Effort: VERY LOW

- 2-3 lines of code change
- No new dependencies, no data changes

### Dependencies

- Combines well with every other improvement (orthogonal change)
- Should be the default baseline for all subsequent experiments

### Risk Factors

- CPU training time increase (~12x slower per forward pass through ConvNeXt)
- Mitigated by: feature caching (run ConvNeXt once, cache 768-d vectors for each frame)

### Success Criteria

- val_acc >= 60% (already achieved: 61.0%)
- Evaluation success rates improve or remain stable vs. 64x64 baseline

---

## Improvement 2: Frame Stacking (Temporal Context)

### Summary

Provide the model with the last N frames (e.g., 4) instead of a single frame. This gives the action head access to recent visual history, enabling it to infer motion, progress, and state changes that are invisible from a single snapshot.

### Motivation

Multi-step tasks like place_table require knowing what happened before (e.g., "I just collected wood, now I should craft a table"). A single frame cannot distinguish "approaching a tree" from "just chopped a tree" if the visual state is similar. Frame stacking is the simplest form of temporal context.

### What to Change

| Item | Details |
|---|---|
| Files | Dataset class (return sequences of N frames), model forward() (accept stacked input), evaluation loop (maintain frame buffer) |
| Architecture | Two options below |
| Training | Mini-batches now contain sequences; dataset returns (frames[t-N+1:t+1], text, action_t) |

### Architecture Options

**Option A: Channel concatenation**
- Stack N frames along channel dim: input becomes (B, 3*N, H, W) = (B, 12, 64, 64) for N=4
- Requires modifying ConvNeXt's first conv layer (in_channels=3 -> 12) or using a small adapter conv
- Adapter approach: Conv2d(12, 3, kernel_size=1) before ConvNeXt (maps 12 channels back to 3)

**Option B: Per-frame encoding + temporal aggregation**
- Encode each frame independently through ConvNeXt -> N vectors of 768-d
- Aggregate: mean pooling, max pooling, or learned weighted average
- Preserves frozen ConvNeXt without modification
- Action head input: aggregated 768-d + 384-d text = 1152-d (unchanged)

**Recommendation:** Option B is simpler and preserves the frozen encoder contract. Option A is more expressive but requires an adapter layer.

### Implementation Notes

- Dataset must handle episode boundaries (pad with zeros or repeat first frame for early timesteps)
- Evaluation loop must maintain a rolling buffer of the last N frames
- For Option B, batch processing N frames per step increases compute by factor N

### Expected Impact: HIGH

- Enables the model to perceive temporal patterns critical for multi-step tasks
- place_table requires: collect_wood -> approach table location -> place. Without temporal context, the model cannot know it already has wood.
- collect_stone requires: find stone -> approach -> mine. Motion direction is invisible from one frame.

### Complexity/Effort: MEDIUM

- Dataset changes are straightforward but must handle edge cases (episode start, padding)
- Evaluation loop changes are moderate (frame buffer management)
- Option B requires no architecture changes beyond the aggregation layer

### Dependencies

- Combines extremely well with Improvement 1 (224x224 resolution)
- Combines well with Improvement 4 (wider action head to process richer input)
- Natural stepping stone to Improvement 3 (recurrent model)

### Risk Factors

- CPU compute scales linearly with N for Option B (4x slower if N=4)
- Diminishing returns beyond N=8-16 frames (Crafter runs at ~20 fps equivalent)
- Feature caching mitigates the compute cost (pre-extract ConvNeXt features for all frames)

### Success Criteria

- val_acc > 65% (up from 61.0% with 224x224)
- place_table success > 40% (up from 22%)
- collect_stone success > 5% (up from 0%)

---

## Improvement 3: Recurrent Temporal Model (LSTM/GRU)

### Summary

Instead of frame stacking (fixed window), process frames sequentially through an LSTM or GRU. The recurrent hidden state acts as a compressed memory of the entire episode history, enabling the model to learn long-range temporal dependencies.

### Motivation

Frame stacking provides a fixed window of N frames. For tasks requiring longer context (e.g., remembering inventory state from 50 steps ago), a recurrent model can maintain relevant information across the full episode. The hidden state learns to encode task-relevant history: "I collected wood 30 steps ago, I've been walking toward the crafting area, now I should place."

### What to Change

| Item | Details |
|---|---|
| Files | Model class (add LSTM/GRU layer), dataset class (return full episode sequences), training loop (sequence-level batching with truncated BPTT), evaluation loop (hidden state management across steps) |
| Architecture | Per-frame: ConvNeXt -> 768-d, concat text -> 1152-d -> LSTM -> hidden -> action head |
| Training | Sequence-level batching: episodes as sequences, truncated backpropagation through time (TBPTT) |

### Architecture

```
Frame_t -> ConvNeXt (frozen) -> 768-d
                                        -> concat -> 1152-d -> LSTM(1152, 512) -> h_t
Text    -> MiniLM (frozen)   -> 384-d                                               -> MLP(512, 8) -> action_t
```

- LSTM hidden size: 512 (adds ~3.3M params for single-layer LSTM)
- GRU alternative: ~2.5M params, simpler, often performs comparably
- Action head: Linear(512, 256) -> ReLU -> Linear(256, 8)

### Training Changes

- **Sequence batching:** Each training sample is a full episode (or truncated segment of length T)
- **Truncated BPTT:** Backpropagate through T=32 or T=64 steps at a time
- **Hidden state detaching:** Detach hidden state every T steps to limit memory usage
- **Padding/packing:** Variable-length episodes require PyTorch packed sequences or padding

### Evaluation Changes

- Maintain LSTM hidden state across the entire episode (do not reset between steps)
- Reset hidden state at the start of each new episode
- Hidden state must be passed through the evaluation loop explicitly

### Expected Impact: HIGH

- Learns temporal patterns like "just collected wood -> now place table"
- Can encode inventory state, navigation progress, and task phase implicitly
- Handles arbitrarily long dependencies (unlike fixed-window frame stacking)

### Complexity/Effort: HIGH

- Significant training infrastructure changes (sequence batching, TBPTT)
- Dataset must return ordered episode sequences (not shuffled individual frames)
- Evaluation loop requires hidden state management
- Debugging recurrent models is harder (vanishing gradients, hidden state drift)

### Dependencies

- Improvement 1 (224x224) is orthogonal and should be applied
- Supersedes Improvement 2 (frame stacking) -- no need for both, though frame stacking can be tried first as a simpler experiment
- Combines well with Improvement 4 (wider head) and Improvement 5 (unfreezing)

### Risk Factors

- Training instability with recurrent models (gradient clipping recommended)
- CPU training is slow for sequence models (LSTM over 200-step episodes)
- Overfitting on 33K samples with 3.3M new parameters -- regularization needed (dropout on LSTM, weight decay)
- Sequence batching is memory-intensive

### Success Criteria

- val_acc > 70%
- place_table success > 50%
- collect_stone success > 10%
- All three tasks show non-trivial success rates simultaneously

---

## Improvement 4: Wider/Deeper Action Head

### Summary

Increase the capacity of the trainable MLP action head from the current 2-layer (297K params) to 3-layer or 4-layer variants with more hidden units. This provides the head with more capacity for learning cross-modal interactions between vision and text features.

### Motivation

The current head (1152 -> 256 -> 8) compresses 1152-dimensional multimodal features through a single 256-d bottleneck before the 8-class output. This may be insufficient for learning complex decision boundaries, especially when the frozen encoders produce features not perfectly suited to the task. A wider/deeper head can learn more expressive transformations.

### What to Change

| Item | Details |
|---|---|
| Files | Model class (modify action head definition) |
| Architecture | See options below |
| Training | Same pipeline, marginal compute increase |

### Architecture Options

| Variant | Architecture | Params | Notes |
|---|---|---|---|
| Current | 1152 -> 256 -> 8 | ~297K | Baseline |
| Wide | 1152 -> 512 -> 256 -> 8 | ~657K | 2x wider first layer, one extra layer |
| Deep | 1152 -> 512 -> 256 -> 128 -> 8 | ~690K | Three hidden layers |
| Wide+Deep | 1152 -> 768 -> 384 -> 192 -> 8 | ~1.17M | Maximum capacity variant |

All variants should include:
- ReLU activations between layers
- Dropout (p=0.1-0.3) between layers for regularization
- Optional BatchNorm or LayerNorm between layers

### Expected Impact: MEDIUM

- More capacity for cross-modal feature interaction
- Unlikely to solve fundamental issues (domain gap, temporal reasoning)
- Best when combined with richer input (frame stacking, better features)
- Diminishing returns beyond ~1M params given 33K training samples

### Complexity/Effort: VERY LOW

- 5-10 lines of code change
- No data or training pipeline changes
- Easy to sweep over variants

### Dependencies

- Orthogonal to all other improvements
- Most valuable when combined with Improvement 2 (frame stacking) or Improvement 3 (recurrent model), which provide richer input to the head
- Also valuable with Improvement 5 (unfreezing), where the head needs to co-adapt with changing encoder features

### Risk Factors

- Overfitting with larger heads on 33K samples
- Mitigated by: dropout, weight decay, early stopping
- CPU training cost increase is negligible (<5%)

### Success Criteria

- val_acc improvement of 1-3pp over baseline with same encoder setup
- No degradation on any single task

---

## Improvement 5: Unfreeze Last ConvNeXt Stage (Progressive Fine-tuning)

### Summary

Unfreeze the last 1-2 stages of ConvNeXt-Tiny to allow the vision encoder to adapt its features from ImageNet to the Crafter pixel art domain. Use progressive unfreezing (head first, then later stages) and differential learning rates to prevent catastrophic forgetting.

### Motivation

The domain gap between ImageNet photographs and Crafter 64x64 pixel art is the single largest accuracy bottleneck identified in MVP-2. ConvNeXt's features were learned on 224x224 natural images -- they encode textures, edges, and object parts that may not align with Crafter's blocky, low-resolution aesthetic. Fine-tuning the last stages adapts high-level feature representations to the target domain while preserving low-level features (edges, colors) that transfer well.

### What to Change

| Item | Details |
|---|---|
| Files | Model class (selective parameter unfreezing), training script (differential LR, progressive schedule) |
| Architecture | Same ConvNeXt-Tiny, but last stage(s) become trainable |
| Training | Multi-phase schedule, differential LR groups, stronger regularization |

### ConvNeXt-Tiny Stage Structure

```
ConvNeXt-Tiny stages:
  Stage 0: 96-d,  3 blocks  (~0.4M params)  -- low-level features (edges, textures)
  Stage 1: 192-d, 3 blocks  (~1.5M params)  -- mid-level features
  Stage 2: 384-d, 9 blocks  (~8.5M params)  -- high-level features
  Stage 3: 768-d, 3 blocks  (~7.1M params)  -- task-level features <-- unfreeze first
```

### Progressive Unfreezing Schedule

| Phase | Epochs | Trainable Components | LR |
|---|---|---|---|
| 1 | 1-5 | Action head only | 1e-3 |
| 2 | 6-10 | Action head + Stage 3 | Head: 1e-3, Stage 3: 1e-4 |
| 3 | 11-15 | Action head + Stage 3 + Stage 2 | Head: 1e-3, Stage 3: 1e-4, Stage 2: 1e-5 |

### Implementation Notes

- Use separate optimizer parameter groups with different learning rates
- Keep backbone in train mode ONLY for unfrozen stages; frozen stages stay in eval mode (StochasticDepth fix)
- Apply weight decay (0.01-0.05) to unfrozen encoder parameters
- Consider gradient accumulation if batch size must decrease due to memory

### Expected Impact: HIGH

- Directly addresses the #1 bottleneck (domain gap)
- Fine-tuned features will be specific to Crafter's visual vocabulary
- Even partial fine-tuning (Stage 3 only) adds 7.1M trainable params -- significant capacity

### Complexity/Effort: MEDIUM

- Selective unfreezing requires careful parameter group management
- Progressive schedule adds training loop complexity
- Differential LR requires optimizer group configuration
- Must handle the StochasticDepth eval/train mode split per stage

### Dependencies

- Improvement 1 (224x224) should be applied first -- fine-tuning at native resolution is more effective
- Combines well with Improvement 4 (wider head) to co-adapt head and features
- Somewhat competes with Improvement 6 (replacing ConvNeXt entirely)
- Improvement 8 (more data) strongly recommended to prevent overfitting 7.1M+ params on 33K samples

### Risk Factors

- **Overfitting:** 7.1M new trainable params vs. 33K samples is a dangerous ratio
  - Mitigations: differential LR (10x lower), weight decay, early stopping, data augmentation
- **Catastrophic forgetting:** Aggressive fine-tuning can destroy useful pretrained features
  - Mitigations: progressive unfreezing, low LR, short fine-tuning phases
- **CPU training time:** Backpropagating through ConvNeXt stages is much slower than head-only training
  - Mitigations: feature caching for Phase 1, gradient checkpointing for Phase 2+
- **StochasticDepth interaction:** Must carefully manage eval/train mode per stage

### Success Criteria

- val_acc > 70% (up from 61.0% with 224x224)
- Improvement across all three tasks (no catastrophic regression on any single task)
- No signs of severe overfitting (val_acc within 10pp of train_acc)

---

## Improvement 6: Replace Frozen ConvNeXt with Trainable Lightweight CNN

### Summary

Replace the frozen ConvNeXt-Tiny (28.6M params) with a small, fully trainable CNN designed for the Crafter domain. This eliminates the domain gap entirely by learning visual features from scratch on task-specific data.

### Motivation

MVP-1's CrafterCNN (~350K params, trained from scratch) achieved 71.6% val_acc -- significantly better than MVP-2's frozen ConvNeXt (51.6%). This strongly suggests that task-specific features outweigh the benefit of pretrained features when the domain gap is large. A hybrid approach (trainable CNN for vision + frozen MiniLM for text) could combine the best of both worlds.

### What to Change

| Item | Details |
|---|---|
| Files | Model class (replace ConvNeXt with lightweight CNN), config (update encoder parameters) |
| Architecture | Trainable CNN (e.g., MVP-1's CrafterCNN or similar) + frozen MiniLM + MLP head |
| Training | Standard training, full backprop through CNN |

### Architecture

```
Frame  -> CrafterCNN (trainable, ~350K params) -> 256-d or 512-d
                                                                    -> concat -> (640 or 896)-d -> MLP -> action
Text   -> MiniLM (frozen, 22.7M params)        -> 384-d
```

**CrafterCNN design (example):**
```
Conv2d(3, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)     # 64->32 or 224->112
Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)    # 32->16 or 112->56
Conv2d(64, 128, 3, padding=1) -> ReLU -> MaxPool2d(2)   # 16->8  or 56->28
Conv2d(128, 256, 3, padding=1) -> ReLU -> AdaptiveAvgPool2d(1) -> Flatten -> 256-d
```

### Expected Impact: MEDIUM-HIGH

- Eliminates domain gap entirely -- features are learned for Crafter
- MVP-1's 71.6% was achieved WITHOUT text conditioning; adding text should push higher
- Smaller model trains faster on CPU
- But: loses pretrained knowledge (object recognition, spatial reasoning from ImageNet)

### Complexity/Effort: LOW-MEDIUM

- Straightforward architecture swap
- CNN design already proven in MVP-1
- Training is simpler (no frozen/unfrozen stage management)
- Evaluation loop unchanged

### Dependencies

- Mutually exclusive with Improvement 5 (unfreezing ConvNeXt)
- Combines well with Improvement 2 (frame stacking) and Improvement 3 (recurrent model)
- Improvement 4 (wider head) is orthogonal
- Improvement 8 (more data) helps -- trainable CNN benefits from more data

### Risk Factors

- Loses pretrained visual knowledge (may matter for generalization)
- Small CNN may not capture complex spatial relationships
- If combined with 224x224 input, CNN is processing 12x more pixels with far fewer params
- Recommendation: use 64x64 native resolution with this approach (no upscaling needed)

### Success Criteria

- val_acc > 65% (proving text conditioning adds value over MVP-1's 71.6% with simpler metric)
- Balanced success across tasks (no single task below 10%)
- Training time per epoch < 50% of ConvNeXt-based model

---

## Improvement 7: Better Data Collection (DAgger)

### Summary

Implement DAgger (Dataset Aggregation) to address the fundamental distribution mismatch between expert demonstrations and the learned policy's actual state distribution.

### Motivation

The current expert demonstrations are collected using scripted bots with access to the full game state (world map, entity coordinates, inventory). The model only observes a single 64x64 frame. This creates a massive observation mismatch: the expert's decisions are optimal given full information, but may appear suboptimal or inconsistent from the frame-only perspective. DAgger addresses this by collecting data from states the learned policy actually visits, labeled with expert actions.

### What to Change

| Item | Details |
|---|---|
| Files | New data collection script (policy rollout + expert labeling), training pipeline (iterative collection/training) |
| Architecture | No changes |
| Training | Iterative: train -> rollout -> label -> aggregate -> retrain |

### DAgger Algorithm

```
1. Train initial policy pi_0 on expert demonstrations D_0
2. For round i = 1, 2, ..., N:
   a. Roll out pi_{i-1} in environment, collecting states S_i
   b. Query expert oracle for correct action at each state in S_i
   c. Aggregate: D_i = D_{i-1} + (S_i, expert_actions)
   d. Train pi_i on D_i
3. Return pi_N
```

### Expert Oracle Design

The Crafter scripted bot must be modified to:
- Accept an arbitrary game state (not just start from the beginning)
- Given a frame + game state, return the optimal action
- This may require hooking into Crafter's internal state representation

### Expected Impact: MEDIUM-HIGH

- Directly addresses the train/test distribution mismatch
- The model sees states it actually encounters (including recovery from mistakes)
- Proven technique in imitation learning literature

### Complexity/Effort: HIGH

- Requires modifying the expert bot to act as an oracle for arbitrary states
- Requires policy rollout infrastructure (run model in environment)
- Iterative training loop adds pipeline complexity
- Multiple rounds of data collection and retraining

### Dependencies

- Requires a trained model from any other experiment as the initial policy
- Combines well with everything, especially Improvement 8 (more data -- DAgger naturally generates more)
- Most valuable after core architecture improvements (temporal context, better features)

### Risk Factors

- Expert oracle may give noisy labels (scripted bot's decisions from partial observation may be suboptimal)
- Multiple training rounds are time-consuming on CPU
- If the oracle uses full game state but the model only sees frames, the mismatch is reduced but not eliminated
- Need to balance DAgger data with original expert data (mixing ratio)

### Success Criteria

- val_acc improvement of 5+pp over same architecture without DAgger
- Evaluation success rates improve on all three tasks
- Particularly: reduction in "confused" behavior where the model takes obviously wrong actions

---

## Improvement 8: More Training Data (Scale Up)

### Summary

Increase the training dataset from 500 episodes (~33K samples) to 5000+ episodes (~330K samples) by running the scripted expert bots for longer.

### Motivation

33K samples is small for training even a 297K-parameter head, let alone for approaches that unfreeze encoder parameters (Improvement 5) or use recurrent models (Improvement 3). More data generally helps, especially for behavioral cloning where diverse state coverage matters.

### What to Change

| Item | Details |
|---|---|
| Files | Data collection script (increase episode count), possibly data loading (streaming for larger datasets) |
| Architecture | No changes |
| Training | Longer training time, possibly more epochs needed |

### Scaling Plan

| Dataset Size | Episodes | Samples (est.) | Collection Time (est.) | Notes |
|---|---|---|---|---|
| Current | 500 | ~33K | Already done | Baseline |
| 2x | 1000 | ~66K | ~1 hour | Quick sanity check |
| 5x | 2500 | ~165K | ~2.5 hours | Sweet spot for CPU training |
| 10x | 5000 | ~330K | ~5 hours | Maximum practical for CPU |

### Expected Impact: MEDIUM

- More data always helps, especially for under-represented actions and states
- Diminishing returns without addressing structural limitations (no temporal context, domain gap)
- Most impactful when combined with model capacity increases (unfreezing, recurrent model)

### Complexity/Effort: LOW

- Just run the existing collection script for longer
- No code changes needed (assuming dataset loader handles arbitrary sizes)
- Time cost is the main bottleneck

### Dependencies

- Essential for Improvement 5 (unfreezing) -- 7.1M params need more than 33K samples
- Helpful for Improvement 3 (recurrent model) -- sequence training benefits from more episodes
- Orthogonal to all other improvements

### Risk Factors

- Collection time on CPU (scripted bots should be fast, but 5K episodes take hours)
- Disk space for larger datasets (trajectories with frames can be large)
- Training time increases linearly with dataset size
- Diminishing returns: 10x data with same architecture may yield only 3-5pp improvement

### Success Criteria

- val_acc improvement of 2-5pp from data scaling alone (same architecture)
- More balanced per-class accuracy (under-represented actions benefit most)
- No performance regression on any task

---

## Improvement 9: MLP to Transformer Action Head (Cross-Attention)

### Summary

Replace the concatenation + MLP fusion mechanism with a cross-attention Transformer that enables vision and text features to interact more expressively. Instead of concatenating pooled vectors, use unpooled ConvNeXt feature maps as vision tokens and MiniLM outputs as text tokens, with cross-attention between them.

### Motivation

The current approach concatenates a single 768-d vision vector with a single 384-d text vector. This discards spatial information from the vision encoder and limits the interaction between modalities to what the MLP can learn from the concatenated vector. Cross-attention allows each text token to attend to relevant spatial locations in the image, and vice versa -- a much richer interaction.

### What to Change

| Item | Details |
|---|---|
| Files | Model class (new Transformer-based action head) |
| Architecture | Vision tokens (from ConvNeXt feature map) cross-attend to text tokens (from MiniLM) |
| Training | Same pipeline, moderate compute increase |

### Architecture

```
Frame -> ConvNeXt (frozen) -> feature map (768, 2, 2) -> flatten -> 4 vision tokens of 768-d
                                                                                                -> project to d_model
Text  -> MiniLM (frozen)   -> token embeddings -> K text tokens of 384-d
                                                                                                -> project to d_model

Cross-Attention Block (x1-2):
  - MultiHeadAttention(query=vision_tokens, key=text_tokens, value=text_tokens)
  - FFN
  - LayerNorm

Output: mean-pool attention output -> MLP(d_model, 8) -> action logits
```

- d_model: 256 or 512
- Heads: 4 or 8
- Layers: 1-2 (keep small for CPU training)
- Estimated params: ~500K-2M depending on configuration

### Expected Impact: LOW for current setup, HIGH if combined with temporal model

- With only 3 instructions, the text-vision interaction is simple enough for concatenation + MLP
- Cross-attention becomes valuable with more instructions, longer text, or spatial reasoning requirements
- When combined with temporal model (Improvement 3), cross-attention enables the model to focus on instruction-relevant parts of each frame

### Complexity/Effort: MEDIUM-HIGH

- Implementing cross-attention correctly requires careful dimension management
- Need to extract unpooled features from ConvNeXt (feature map, not pooled vector)
- Positional encodings for vision tokens (spatial) and text tokens (sequential)
- More hyperparameters to tune (d_model, heads, layers, dropout)

### Dependencies

- Improvement 1 (224x224) provides larger feature maps (7x7 = 49 tokens vs 2x2 = 4 tokens at 64x64)
- Combines well with Improvement 3 (recurrent model) for temporal + cross-modal reasoning
- Less useful with Improvement 6 (lightweight CNN) which has smaller feature maps

### Risk Factors

- Over-engineering for the current 3-instruction setup
- Slower training on CPU (attention is O(n*m) in sequence lengths)
- More hyperparameters = more tuning time
- May not outperform simple concatenation given the limited number of instructions

### Success Criteria

- val_acc improvement of 2-5pp over concatenation + MLP baseline (same encoder setup)
- Qualitative: attention maps show task-relevant spatial focus per instruction
- Practical: training time increase < 2x over MLP head

---

## Improvement 10: Curriculum Learning

### Summary

Train the model progressively, starting with the easiest task (collect_wood) and gradually introducing harder tasks (place_table, collect_stone). This prevents the model from being overwhelmed by difficult, rare-success tasks early in training.

### Motivation

The current training mixes all three tasks from the start. collect_wood is the simplest (single-step: find tree, approach, chop), while place_table and collect_stone are multi-step and rarer in the demonstration data. Training on all tasks simultaneously may cause the model to focus on the most common/easiest patterns, neglecting the harder tasks. Curriculum learning explicitly controls this progression.

### What to Change

| Item | Details |
|---|---|
| Files | Training script (curriculum schedule), dataset class (task filtering/weighting) |
| Architecture | No changes |
| Training | Multi-phase training with controlled task exposure |

### Curriculum Schedule

| Phase | Epochs | Tasks Included | Sampling Weight |
|---|---|---|---|
| 1 | 1-3 | collect_wood only | 100% wood |
| 2 | 4-7 | collect_wood + place_table | 50% / 50% |
| 3 | 8-12 | All three tasks | 33% / 33% / 33% |
| 4 | 13-15 | All three tasks, hard-weighted | 20% / 40% / 40% (emphasis on hard tasks) |

### Alternative: Loss Weighting Curriculum

Instead of controlling data exposure, weight the cross-entropy loss per task:
```python
# Phase 1: equal weights
# Phase 2: upweight underperforming tasks
weights = {task: 1.0 / (success_rate + 0.1) for task, success_rate in eval_results.items()}
```

### Expected Impact: LOW-MEDIUM

- Helps with class imbalance and task difficulty imbalance
- Most effective when the model has the capacity to learn all tasks (so other improvements should come first)
- Limited impact if the fundamental issue is missing temporal context (Improvements 2/3)

### Complexity/Effort: LOW

- Dataset filtering/weighting is straightforward
- Training loop changes are minimal (phase transitions based on epoch)
- Easy to combine with other improvements

### Dependencies

- Orthogonal to all other improvements
- Most valuable after temporal context (Improvement 2/3) is added -- the model has the capacity to learn multi-step tasks, and curriculum helps it learn them efficiently
- Pairs well with Improvement 8 (more data) for balanced task coverage

### Risk Factors

- Early phases may overfit to easy tasks, then catastrophically forget when hard tasks are introduced
- The curriculum schedule is sensitive to hyperparameters (when to transition, how to weight)
- If all three tasks share the same underlying actions (which they do -- 8 actions), task-specific training may be less beneficial than expected

### Success Criteria

- More balanced success rates across tasks (narrower gap between best and worst)
- No single task below 5% success
- val_acc comparable to or better than uniform training

---

## Recommended Experiment Priority

### Impact / Effort Matrix

| Improvement | Impact | Effort | Impact/Effort | Priority |
|---|---|---|---|---|
| 1. Input Resolution (224x224) | HIGH (proven) | VERY LOW | **VERY HIGH** | **P0 -- Do First** |
| 4. Wider/Deeper Action Head | MEDIUM | VERY LOW | **HIGH** | **P1** |
| 2. Frame Stacking | HIGH | MEDIUM | **HIGH** | **P1** |
| 6. Lightweight CNN | MEDIUM-HIGH | LOW-MEDIUM | **MEDIUM-HIGH** | **P2** |
| 5. Unfreeze ConvNeXt | HIGH | MEDIUM | **MEDIUM-HIGH** | **P2** |
| 10. Curriculum Learning | LOW-MEDIUM | LOW | **MEDIUM** | **P3** |
| 8. More Training Data | MEDIUM | LOW | **MEDIUM** | **P3** |
| 3. Recurrent Model | HIGH | HIGH | **MEDIUM** | **P3** |
| 7. DAgger | MEDIUM-HIGH | HIGH | **MEDIUM** | **P4** |
| 9. Transformer Head | LOW-HIGH* | MEDIUM-HIGH | **LOW** | **P5** |

*Transformer head impact is context-dependent: LOW alone, HIGH with temporal model.

### Recommended Experiment Sequence

**Phase 1: Quick Wins (1-2 days)**

Apply these independently to establish a strong baseline:

1. **Improvement 1 (224x224)** -- already proven, apply immediately. New baseline: ~61% val_acc.
2. **Improvement 4 (wider head)** -- sweep 3 variants on top of 224x224. Takes minutes to try.
3. Measure: val_acc, per-task success rates.

**Phase 2: Temporal Context (3-5 days)**

The single highest-impact area. Two competing approaches to evaluate:

4. **Improvement 2 (frame stacking, Option B)** -- simpler, faster to implement. Try N=4 and N=8.
5. If frame stacking shows clear improvement, consider **Improvement 3 (recurrent model)** as a follow-up.
6. Measure: emphasis on place_table and collect_stone success (temporal tasks).

**Phase 3: Domain Adaptation (3-5 days)**

Address the domain gap. Two competing approaches:

7. **Improvement 6 (lightweight CNN)** -- quick to try if MVP-1's CNN code exists. Compare against ConvNeXt.
8. **Improvement 5 (unfreeze ConvNeXt)** -- try if 224x224 + wider head + frame stacking still underperforms.
9. These are mutually exclusive -- pick the winner.
10. Measure: val_acc on held-out data, watch for overfitting.

**Phase 4: Data and Training (ongoing)**

11. **Improvement 8 (more data)** -- start collection early (runs in background), use when ready.
12. **Improvement 10 (curriculum)** -- try after Phase 2, especially if task imbalance persists.

**Phase 5: Advanced (if needed)**

13. **Improvement 7 (DAgger)** -- only if evaluation success rates plateau despite good val_acc.
14. **Improvement 9 (Transformer head)** -- only if scaling to more instructions/tasks.

### Recommended Combinations

The following combination packages are expected to yield the best results:

**Package A: Conservative (easiest, fastest)**
- Improvements 1 + 4: 224x224 + wider head
- Expected: ~63-65% val_acc
- Effort: 1 day

**Package B: Temporal Focus (highest expected impact)**
- Improvements 1 + 4 + 2: 224x224 + wider head + frame stacking
- Expected: ~68-72% val_acc, balanced task success
- Effort: 3-4 days

**Package C: Domain Adaptation**
- Improvements 6 + 4 + 2: Lightweight CNN + wider head + frame stacking
- Expected: ~70-75% val_acc (task-specific features + temporal context)
- Effort: 4-5 days

**Package D: Maximum Effort**
- Improvements 1 + 5 + 3 + 4 + 8: 224x224 + unfrozen ConvNeXt + LSTM + wider head + more data
- Expected: ~75-80% val_acc, balanced success
- Effort: 7-10 days
- Risk: complex, many interacting changes

### Target Metrics

| Milestone | val_acc | collect_wood | place_table | collect_stone |
|---|---|---|---|---|
| Current (MVP-2) | 51.6% | 72% | 22% | 0% |
| Phase 1 target | 63% | 75% | 30% | 5% |
| Phase 2 target | 70% | 80% | 50% | 15% |
| Phase 3 target | 75% | 85% | 60% | 25% |
| Stretch goal | 80% | 90% | 70% | 40% |

---

## Appendix: Key Constraints

- **CPU only:** No CUDA available on the dev machine. All training and evaluation runs on CPU. This makes compute-heavy approaches (large models, long sequences, many epochs) painful. Feature caching is essential for iterating quickly.
- **33K samples:** Current dataset is small. Overfitting is a constant risk, especially for approaches that add trainable parameters.
- **3 instructions:** The task set is small. Some improvements (Transformer head, curriculum) are designed for larger task sets and may show limited benefit here.
- **Pixel art domain:** 64x64 resolution, blocky graphics, limited color palette. Pretrained ImageNet features are mismatched. Task-specific or fine-tuned features are likely necessary for high accuracy.
