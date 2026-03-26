# vla-game-agent

A tiny instruction-conditioned vision-to-action agent for [Crafter](https://github.com/danijar/crafter).

Takes a game frame + a text instruction, predicts a player action. Demonstrates the [VLA (Vision-Language-Action)](https://arxiv.org/abs/2307.15818) paradigm applied to games: one model, many tasks, steered by language.

---

## Architecture

```
4 Frames (64x64 RGB)            Instruction ("collect wood")
        |                                  |
  [Trainable CNN]               [Frozen all-MiniLM-L6-v2]
  3-conv, 256-d per frame            384-d text embedding
        |                                  |
  [Mean Pool x4 frames]                   |
        |                                  |
        +-----------concat-----------------+
                      |
               [MLP Action Head]
                 640 -> 256 -> 8
                      |
                8 action logits
```

~504K trainable parameters (CNN + action head). Text encoder is frozen. Trains on a single GPU in minutes.

---

## Key Results

| Model | val_acc | collect_wood | place_table | collect_stone |
|-------|---------|-------------|-------------|---------------|
| MVP-1: CNN only (no text) | 71.6% | 8% | 84% | 10% |
| **MVP-2.3: CNN + text + 4-frame** | **76.8%** | **38%** | **76%** | **6%** |

**The headline:** MVP-1 has no instruction input — it collapses to a single behavior and cannot distinguish tasks (84% place_table is accidental). MVP-2.3 follows instructions: it collects wood when told "collect wood" and places a table when told "place table."

The full 5-milestone experiment progression is in the [technical report](report.md).

![Training curves](artifacts/figures/training_curves.png)
![Task success rates](artifacts/figures/task_success_rates.png)

---

## Demo

Sample videos of the MVP-2.3 agent following different instructions are in [`artifacts/demo/mvp2.3/`](artifacts/demo/mvp2.3/).

Generate your own:

```bash
uv run python scripts/demo_policy.py \
  --policy-type vla-cnn \
  --model artifacts/models/mvp2.3/best_model.pt \
  --num-frames 4 --output-dir artifacts/demo/mvp2.3
```

---

## Quick Start

```bash
# Setup
uv sync

# Run tests
uv run python -m pytest

# Train a VLA model
uv run python scripts/train_imitation.py \
  --model-type vla-cnn \
  --data-dirs artifacts/trajectories/collect_wood \
              artifacts/trajectories/place_table \
              artifacts/trajectories/collect_stone \
  --output-dir artifacts/models/mvp2.3 \
  --epochs 20 --batch-size 64 --lr 1e-3 --seed 42 \
  --device cuda --num-frames 4

# Evaluate
uv run python scripts/evaluate_policy.py \
  --model artifacts/models/mvp2.3/best_model.pt \
  --policy-type vla-cnn \
  --num-episodes 50 --device cuda --num-frames 4
```

---

## Project Structure

```
src/vla_agent/
    envs/           Gymnasium-style Crafter wrapper (8-action space)
    models.py       CrafterCNN, CrafterVLA, InstructionEncoder
    data.py         TrajectoryDataset with frame stacking
    policies.py     Scripted expert policies (data collection)
    pipeline/       Agentic TDD pipeline (core + providers)

scripts/
    train_imitation.py      Train CNN, VLA, or VLA-CNN models
    evaluate_policy.py      Evaluate via environment rollouts
    collect_trajectories.py Generate training data from scripted experts
    demo_policy.py          Record demo videos
    plot_results.py         Generate comparison plots
    run_pipeline.py         Launch the agentic TDD pipeline

specs/              Approved specifications for each milestone
tests/              143 unit tests (98 integration tests, skipped by default)
artifacts/          Trained models, evaluation results, figures, demo videos
```

---

## Agentic TDD Pipeline

This project's code was built by an automated pipeline — not written manually. One command takes a human-approved spec and delivers tested, reviewed, trained code:

```bash
uv run python scripts/run_pipeline.py <task-id> --provider codex
```

```
spec (human) -> tests -> review -> implement -> validate -> review -> train -> evaluate -> verify
```

8 stages, fully automated. AI agents in specialized roles (test-writer, implementer, reviewer). Frozen-test guardrails. Auto-resume from interruption. Provider-agnostic (Codex, Claude, Gemini planned).

Delivered 5 milestones of production code, trained real models, verified real metrics. See [docs/agentic-pipeline.md](docs/agentic-pipeline.md) for the full design.

---

## Limitations and Further Work

**Current limitations:** 3-task vocabulary, scripted training data, reactive model (no persistent memory), no RL fine-tuning. The agent is a proof of concept, not a deployable system.

**ML directions:** More tasks, human gameplay data, attention-based temporal reasoning, RL fine-tuning.

**Pipeline directions:** Extract as standalone open-source tool, add Gemini provider, generalize beyond this project.

See the [technical report](report.md) for detailed analysis and the full experiment progression.

---

## License

MIT
