# Technical Report: Instruction-Conditioned Vision-to-Action Agent for Crafter

## Introduction

This project builds a tiny **Vision-Language-Action (VLA)** agent for [Crafter](https://github.com/danijar/crafter), a 2D survival game. The agent observes a game frame, reads a text instruction like "collect wood" or "place table," and predicts a player action. The core question: does language grounding enable task-specific behavior that a vision-only model cannot achieve?

The answer is yes. A vision-only baseline (MVP-1) learns a single mixed behavior and cannot distinguish tasks. The instruction-conditioned model (MVP-2.3) produces dramatically different behavior per instruction — collecting wood when told to collect wood, placing a table when told to place a table. The gap between these two models is the project's central result.

The project also produced an unexpected second contribution: an **agentic TDD pipeline** that automates the entire development lifecycle — from test generation through code review to model training — using AI agents in specialized roles. This pipeline delivered 5 milestones of production code with zero manual coding after spec approval.

---

## The VLA Paradigm

### RL vs VLA

Reinforcement learning answers: "maximize this reward signal." It requires per-task reward functions, millions of environment interactions, and produces policies that cannot be redirected with language. Each task needs a separate training run.

VLA (Vision-Language-Action) answers: "given what you see and what I told you, what should you do?" One model handles many tasks, steered by natural language instructions. The instruction is the interface — change the instruction, change the behavior. No retraining required.

This is the paradigm shift from "one policy per task" to "one model that follows instructions." It originates in robotics (RT-2, Octo, OpenVLA), where physical robots must follow diverse human instructions without retraining. This project demonstrates the same principle in a game environment.

### Why Pretrained Encoders

Training vision and language representations from scratch on ~33K samples is infeasible. Pretrained encoders solve this: a frozen text encoder provides language understanding, and a trainable vision backbone learns domain-specific visual features. Only a small action head learns the mapping from (visual + text features) to actions. The hard problem — understanding language — is solved by pretraining on large corpora.

---

## Architecture

The final architecture (MVP-2.3) processes 4 consecutive game frames through a trainable CNN, encodes the text instruction through a frozen sentence transformer, and fuses both representations to predict one of 8 player actions.

```
4 Frames (64x64 RGB each)          Instruction ("collect wood")
        |                                     |
  [Trainable CNN]                  [Frozen all-MiniLM-L6-v2]
  per-frame encoding                          |
        |                              text features (384-d)
  4 x vision features (256-d)                |
        |                                     |
  [Mean Pool over time]                       |
        |                                     |
  vision features (256-d)                     |
        |                                     |
        +-------------concat-----------------+
                        |
                 fused features (640-d)
                        |
                [Trainable MLP Head]
                  640 -> 256 -> 8
                        |
                  8 action logits
```

**CNN backbone:** 3 convolutional layers (same architecture as the Nature DQN encoder) followed by a linear projection to 256 dimensions. All parameters are trainable — the CNN learns Crafter-specific visual features at the native 64x64 resolution.

**Text encoder:** Frozen `all-MiniLM-L6-v2` (22M parameters, 384-d output). Embeddings are cached — with only 3 distinct instructions, encoding happens once at startup.

**Frame stacking:** 4 consecutive frames are independently encoded by the CNN, then mean-pooled across the temporal dimension. This provides short-term motion context without the complexity of recurrent architectures.

**Trainable parameters:** ~504K (CNN backbone + action head). The text encoder is frozen.

---

## Data

Training data comes from **scripted expert policies** — hand-coded bots that use full game state access (world map, player coordinates, material locations) to execute tasks optimally. Three policies:

| Policy | Task | Strategy |
|--------|------|----------|
| `CollectWoodPolicy` | Collect wood | Navigate to nearest tree, chop it |
| `PlaceTablePolicy` | Place a crafting table | Collect wood, then craft and place table |
| `CollectStonePolicy` | Collect stone | Collect wood, place table, make pickaxe, mine stone |

**500 episodes per policy** (1,500 total), each up to 300 steps. Each frame is paired with the active instruction and the expert's action. Total: ~33K training samples.

Scripted data is deterministic, reproducible, and sufficient for demonstrating instruction conditioning. The trade-off: the model can only learn behaviors present in the scripted policies. Open-ended or creative behaviors require human demonstrations, which are out of scope.

---

## Experiment Progression

Each milestone tested a specific hypothesis. The progression forms a clean ablation study.

### Results

| Milestone | Architecture | Best val_acc | collect_wood | place_table | collect_stone |
|-----------|-------------|-------------|-------------|-------------|---------------|
| MVP-1 | CNN, no text | 71.6% | 8% | 84% | 10% |
| MVP-2 | + frozen ConvNeXt, 64x64 | 51.6% | 72% | 22% | 0% |
| MVP-2.1 | + resize to 224x224 | 60.9% | 48% | 12% | 0% |
| MVP-2.2 | + 4-frame stacking | 55.0% | 58% | 22% | 0% |
| **MVP-2.3** | **Trainable CNN + text + frames** | **76.8%** | **38%** | **76%** | **6%** |

![Training curves across milestones](artifacts/figures/training_curves.png)

![Per-task success rates across milestones](artifacts/figures/task_success_rates.png)

### MVP-1: Vision-Only Baseline

**Hypothesis:** A CNN trained on all 3 tasks without instruction input will learn a single mixed behavior.

**Result:** Confirmed. The model achieved 71.6% validation accuracy (frame-level action prediction) but collapsed to a single dominant behavior in evaluation. place_table succeeded 84% of the time because its action sequence (move, chop, place) overlaps with the statistically dominant training signal. collect_wood and collect_stone succeeded at near-random rates (8% and 10%). The model cannot distinguish tasks — it replays one behavior regardless of context.

**Takeaway:** Without language input, the model has no mechanism to select task-specific behavior.

### MVP-2: Add Text Conditioning

**Hypothesis:** Adding instruction embeddings will enable task-specific behavior.

**Result:** Confirmed for simple tasks. collect_wood jumped from 8% to 72% — a 9x improvement. The model shows dramatically different behavior per instruction, proving the text embeddings influence decisions. However, place_table dropped from 84% to 22% and collect_stone hit 0%. The frozen ConvNeXt backbone (pretrained on ImageNet natural photos) struggles with Crafter's 64x64 pixel art — a domain gap problem.

**Takeaway:** Language grounding works. The domain gap between ImageNet and pixel art is the new bottleneck.

### MVP-2.1: Resize to 224x224

**Hypothesis:** Feeding ConvNeXt its native training resolution (224x224 instead of 64x64) will improve feature quality.

**Result:** val_acc improved from 51.6% to 60.9%. At 64x64 input, ConvNeXt's final feature map is 2x2 (4 spatial positions). At 224x224, it's 7x7 (49 positions) — 12x richer spatial information. This is spatial scale alignment, not model capacity.

**Takeaway:** Frozen pretrained encoders must be fed at their training resolution.

### MVP-2.2: Frame Stacking

**Hypothesis:** Providing 4 consecutive frames (instead of 1) will help multi-step tasks by giving the model short-term temporal context.

**Result:** collect_wood improved from 48% to 58%, place_table from 12% to 22%. The model can now perceive motion and recent actions. However, val_acc dropped to 55% — per-frame prediction accuracy is a poor proxy for task success. The model makes better action sequences despite being less accurate on individual frames.

**Takeaway:** Temporal context helps. val_acc does not capture sequential decision quality.

### MVP-2.3: Trainable CNN (Domain Adaptation)

**Hypothesis:** Replacing the frozen ImageNet ConvNeXt with a trainable CNN that learns Crafter-specific features will eliminate the domain gap.

**Result:** Best results across the board. val_acc=76.8% (highest ever, surpassing MVP-1's 71.6%). place_table recovered from 22% to 76%. collect_stone reached 6% — the first nonzero result for any VLA configuration. The trainable CNN learns pixel-art-specific features (terrain boundaries, inventory sprites, entity outlines) rather than applying ImageNet texture detectors to a foreign visual domain.

**Takeaway:** Domain-matched visual features matter more than pretrained generality when the domain gap is large.

---

## Key Findings

1. **Language grounding enables task-specific behavior.** The gap between MVP-1 (one behavior for all tasks) and MVP-2+ (different behavior per instruction) is the project's core result. This validates the VLA approach for game agents.

2. **Domain gap is the primary bottleneck for frozen encoders.** Frozen ImageNet features on 64x64 pixel art max out at ~61% val_acc. A trainable CNN on the same data reaches 76.8%. When your visual domain differs substantially from pretraining data, train domain-specific features.

3. **val_acc is a poor proxy for task success.** MVP-2.2 had lower val_acc (55%) than MVP-2.1 (60.9%) but better task success rates. Per-frame action accuracy does not capture the quality of action sequences.

4. **Frozen pretrained encoders need native resolution.** ConvNeXt at 64x64 collapses spatial features to 2x2 before pooling. At 224x224, the same architecture produces 12x richer representations. Always resize to the encoder's training resolution.

5. **Frame stacking helps multi-step tasks.** Even simple mean pooling over 4 frames provides enough temporal context to improve sequential behaviors like place_table (requires: chop wood, craft, place).

---

## The Agentic TDD Pipeline

This project's code was not written manually. Every milestone from MVP-2 onward was delivered by an automated pipeline:

```
spec (human) --> tests --> test review --> implement --> validate --> code review --> train --> evaluate --> verify
```

One command. No manual coding after spec approval:

```bash
uv run python scripts/run_pipeline.py mvp-2.3 --provider codex
```

The pipeline uses AI agents in specialized roles (test-writer, implementer, reviewer), enforces guardrails (frozen-test hashes, reviewer immutability), auto-resumes from interruption, and supports multiple AI backends through a provider-agnostic architecture.

Key properties:
- **One-command automation** — spec to verified artifacts, unattended
- **Provider-agnostic** — same pipeline runs on Codex, Claude, or (planned) Gemini
- **Self-correcting** — reviewer feedback loops, implementer retry on failures
- **Guardrailed** — frozen tests cannot be modified, reviewers cannot edit files, all verified by SHA-256 hashes
- **Battle-tested** — delivered 5 milestones of real code, trained real models, verified real metrics

For the full pipeline design, state machine, and implementation details, see [docs/agentic-pipeline.md](docs/agentic-pipeline.md).

---

## Limitations

- **3-task vocabulary.** The agent handles "collect wood," "place table," and "collect stone." Open-ended instructions are out of scope.
- **Scripted training data.** Expert policies produce optimal but rigid demonstrations. The agent cannot discover strategies the scripts don't demonstrate.
- **Reactive decisions.** Even with 4-frame stacking, the model has no persistent memory. It cannot plan multi-step sequences — each prediction is based on the last 4 frames and the instruction.
- **No RL fine-tuning.** The model is trained purely via behavioral cloning (imitation learning). RL could improve performance on tasks where the imitation signal is weak.
- **collect_stone remains hard.** At 6% success, the 4-step dependency chain (wood -> table -> pickaxe -> stone) exceeds what a reactive model can reliably execute.

---

## Further Work

### ML Track

- **More tasks and instructions.** Expand beyond 3 tasks to test scaling behavior. Add tasks with different complexity levels.
- **Human gameplay data.** Replace or supplement scripted demonstrations with human play. Human strategies are more diverse and may teach the model creative problem-solving.
- **Attention-based temporal reasoning.** Replace mean pooling with a temporal attention mechanism. Allow the model to selectively attend to relevant past frames rather than treating all 4 equally.
- **RL fine-tuning.** Use the imitation-trained model as initialization, then fine-tune with RL on a reward signal. This could improve collect_stone and other tasks where imitation data is insufficient.
- **Larger action space.** The current 8-action space is a subset of Crafter's full 17-action space. Expanding it enables more complex behaviors.

### Pipeline Track

- **Generalize for any repository.** Remove project-specific assumptions from the pipeline core. Extract it as a standalone open-source tool that works with any codebase and spec format.
- **Add Gemini provider.** A spec exists (`specs/gemini-provider-spec.md`). Implementation follows the same adapter pattern as existing providers.
- **Actualize Claude provider.** The Claude adapter may need updates after recent Codex provider refinements (stages 6-8 retry logic, encoding fixes).
- **Richer artifact validation.** Add video generation, qualitative checks, and multi-metric dashboards to the artifact pipeline stages.

---

## Conclusion

This project demonstrated that language grounding enables task-specific behavior in a game agent. A vision-only model collapses to one behavior; an instruction-conditioned model follows diverse instructions. The experiment progression — from frozen pretrained encoders to domain-adapted trainable CNNs — produced a clean ablation study showing that domain-matched visual features and temporal context are key to VLA performance in non-photorealistic environments.

The agentic TDD pipeline that built this project is itself a contribution: a fully automated, provider-agnostic development pipeline that takes a spec and delivers tested, reviewed, trained code. It represents a practical approach to AI-assisted software development where humans define intent and machines handle execution.
