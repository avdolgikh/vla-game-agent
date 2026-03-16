# VLA Components Research Specification

## Goal

Identify a concrete set of pretrained, open-source model components that can be assembled into a Vision-Language-Action (VLA) architecture for fine-tuning on a small game-playing dataset. The deliverable is a ranked shortlist of component combinations with specific model names, parameter counts, licenses, and integration notes — not code.

## Context

### The Project

We are building an instruction-conditioned game agent for Crafter (a 2D procedurally-generated survival game). The agent receives:

- **A game frame** (64×64 RGB image) — the visual observation
- **A text instruction** (short natural language, e.g., "collect wood", "place table", "collect stone") — the task specification

And must output:

- **A discrete action** (one of 8 possible actions: noop, move_left, move_right, move_up, move_down, do, place_table, make_wood_pickaxe)

This is a classification problem: given (image, text) → predict one of 8 action classes.

### What We Have Already

- **Training data**: ~500 episodes across 3 tasks (collect_wood, place_table, collect_stone), collected from scripted expert policies. Each episode is a sequence of (64×64 RGB observation, action) pairs. Total: ~33,000 (observation, action) samples.
- **Baseline (MVP-1)**: A small CNN (Nature DQN architecture, ~350K params) trained from scratch on all 3 tasks mixed, with no text input. Achieves 71.6% frame-level validation accuracy but collapses to one dominant behavior at evaluation time (84% success on place_table, 8% on collect_wood, 10% on collect_stone). This is the expected failure mode — without instruction conditioning, the model cannot distinguish tasks.
- **Framework**: PyTorch. Training and evaluation scripts exist. The new model must be a PyTorch `nn.Module` that can replace `CrafterCNN` in the existing pipeline.

### Why Pretrained Components

Training vision and language representations from scratch on 500 episodes is insufficient. Pretrained encoders bring:

1. **Visual understanding** — recognizing objects, terrain, spatial layout from pixels
2. **Language understanding** — encoding instruction semantics
3. The only thing that needs to be learned from our data is the **action mapping**: (visual features + language features) → action logits

This is the standard VLA paradigm (as in RT-2, Octo, OpenVLA for robotics). We are applying it to a game environment.

## Requirements

### Hardware Constraints

- **GPU**: NVIDIA GeForce RTX 4070, 12 GB VRAM
- **Platform**: Windows 11, Python 3.11+, PyTorch
- Fine-tuning (forward + backward pass through trainable parameters, plus frozen encoder inference) must fit within 12 GB VRAM with batch size ≥ 16
- Inference must be fast enough for real-time evaluation (the agent steps in a game loop)

### Model Architecture Requirements

The VLA model has three logical components:

1. **Vision encoder** — maps a 64×64 RGB image to a feature vector
   - Must be pretrained on a large vision dataset (ImageNet, LAION, etc.)
   - Should be frozen or partially frozen during fine-tuning (to preserve representations and reduce VRAM)
   - Must handle 64×64 input (natively or via resizing/interpolation)
   - Preferred: produces a single feature vector (not a sequence), or has a standard pooling strategy

2. **Text encoder** — maps a short text instruction to a feature vector
   - Must be pretrained on a large text corpus
   - Should be frozen during fine-tuning
   - Must handle very short inputs (2-3 words like "collect wood")
   - Preferred: produces a single sentence-level embedding vector

3. **Action head** — maps concatenated (vision_features, text_features) to 8 action logits
   - This is the only component trained from scratch on our data
   - Small MLP (1-2 hidden layers)
   - This is where all the learning happens

### Component Selection Criteria

For each component, evaluate:

| Criterion | Details |
|-----------|---------|
| **Parameter count** | Must fit in 12 GB VRAM alongside other components during training. Smaller is better — we are not chasing SOTA, we need something that works within our constraints. |
| **Pretrained weights availability** | Must have publicly available weights on HuggingFace, torch hub, or similar. No gated/restricted access. |
| **License** | Must be permissive for research/personal use (Apache 2.0, MIT, BSD, CC-BY preferred). |
| **PyTorch compatibility** | Must load into a PyTorch `nn.Module` without framework conversion. HuggingFace Transformers, timm, or torch hub preferred. |
| **Input compatibility** | Vision: must handle 64×64 RGB (or accept resize to model's native resolution). Text: must handle 2-3 word phrases. |
| **Feature extraction simplicity** | Should produce a fixed-size feature vector with minimal code. No complex multi-scale feature pyramid or decoder needed. |
| **Community adoption** | Prefer well-established models with active maintenance, documentation, and usage examples. |
| **Fine-tuning footprint** | With the encoder frozen, the trainable parameters (action head only) should be < 1M params. If partial fine-tuning of the encoder is considered, LoRA or similar parameter-efficient methods should be feasible. |

### What Is NOT In Scope

- We are NOT building a generative model (no image generation, no text generation)
- We are NOT using a full VLM (no autoregressive token prediction for actions)
- We are NOT doing reinforcement learning
- We are NOT training encoders from scratch
- We do NOT need video understanding (single-frame input, no temporal context)
- We do NOT need models larger than ~1B total parameters (frozen + trainable combined)

## Deliverable

A structured report containing:

1. **Recommended vision encoder(s)** — 2-3 candidates ranked by fit, with:
   - Model name and variant (e.g., "ViT-B/16 from CLIP via openai/clip-vit-base-patch16")
   - Parameter count
   - Output feature dimension
   - Native input resolution and behavior at 64×64
   - Where to download (HuggingFace model ID, torch hub call, etc.)
   - License
   - Pros/cons for our use case

2. **Recommended text encoder(s)** — 2-3 candidates ranked by fit, with:
   - Model name and variant
   - Parameter count
   - Output embedding dimension
   - Behavior on very short inputs (2-3 words)
   - Where to download
   - License
   - Pros/cons for our use case

3. **Recommended combination(s)** — 1-2 specific (vision_encoder, text_encoder) pairings with:
   - Total parameter count (frozen + trainable)
   - Estimated VRAM usage during training (batch size 16-64)
   - Whether the vision and text encoders share a backbone (e.g., CLIP has both)
   - Integration sketch: what each component outputs, how features are combined
   - Any known issues or caveats

4. **Alternatives considered but rejected** — brief notes on models that were evaluated but don't fit (too large, wrong license, no PyTorch support, etc.)

## Notes

- CLIP-family models are natural candidates because they provide both vision and text encoders in a shared embedding space. But also consider whether separate specialized encoders (e.g., DINOv2 for vision + a sentence-transformer for text) might be a better fit.
- Our images are 64×64 — much smaller than typical ViT input (224×224). Consider whether upscaling introduces artifacts vs. using models that handle smaller inputs natively.
- Our text inputs are trivially short ("collect wood"). A large language model is overkill. A small sentence encoder may suffice.
- Consider whether a shared embedding space (where image and text features are aligned, as in CLIP) provides any advantage for our task, or whether independent encoders feeding into a learned action head work equally well.
- The game's visual style is procedurally-generated 2D pixel art — quite different from natural images. Note whether pretrained vision models are likely to transfer well to this domain.
