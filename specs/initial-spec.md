# Pre-Spec: Tiny VLA-Style Game Agent

## Status

Draft v0.1

## Purpose

This repository is a small, credible portfolio project at the intersection of modern ML and game development. It is **not** meant to be a full robotics-grade VLA or a general game-playing foundation model. Instead, it is a short prototype that demonstrates a practical, game-oriented analogue of the Visual-Language-Action paradigm.

The core idea is to build a **tiny instruction-conditioned vision-to-action agent** that takes:

* a game frame or short frame stack,
* a short natural-language instruction,
* and predicts a player-like action.

This project is intentionally scoped as a **POC / prototype** rather than a research program.

---

## High-Level Recommendation

### Primary option: Crafter

Build the first version in **Crafter**.

Why:

* lightweight and fast to iterate on,
* Python-friendly,
* visually simple enough for quick ML experimentation,
* still rich enough to feel game-like rather than toy-like,
* aligned with recent embodied-agent / instruction-following research.

### Alternative option: ViZDoom

Use **ViZDoom** as the second option if the goal shifts toward:

* more visually impressive demos,
* FPS combat behavior,
* short-horizon reactive policies,
* stronger “gameplay bot” aesthetics.

---

## Project Positioning

### Honest positioning

This project should be presented as one of the following:

* **Instruction-conditioned visual game agent**
* **Vision-to-action imitation model for games**
* **Tiny VLA-style game bot**

### Not the claim

Do **not** position this as:

* a full VLA foundation model,
* a general game-playing agent,
* a robotics transfer project,
* a production-ready game AI system.

---

## Core Goals

1. Touch the VLA paradigm in a practical, lightweight way.
2. Apply modern neural-network-based ML directly to games.
3. Produce a small but concrete portfolio artifact.
4. Keep scope tight enough to finish quickly.
5. Leave room for one or two meaningful extensions later.

---

## Non-Goals

* No large-scale pretraining.
* No robotics datasets.
* No huge human gameplay dataset collection effort.
* No reinforcement-learning-heavy research program in v1.
* No open-ended planning across hundreds of tasks.
* No simultaneous PCG + game-agent project in v1.

PCG is interesting, but should be treated as a separate follow-up project unless a very small extension naturally fits later.

---

# Option A: Crafter-Based Tiny VLA-Style Agent

## Working Title

**CrafterVLA-Tiny**

## One-Sentence Description

A small multimodal agent for Crafter that maps game frames and short language goals to discrete player actions using imitation learning.

## Why Crafter

Crafter is the best first target because it offers:

* a survival / crafting loop,
* meaningful action semantics,
* simple visuals,
* low engineering overhead,
* quick iteration cycles,
* enough complexity to make instruction-following nontrivial.

It is a good compromise between “toy environment” and “massive research platform.”

## v1 Problem Formulation

At each timestep, the model receives:

* current RGB observation,
* optional short history of previous frames,
* a text instruction,
* optional previous action,

and predicts:

* one discrete action from the Crafter action space.

## Example Instructions

Start with a small fixed instruction set such as:

* `collect wood`
* `collect stone`
* `place table`
* `craft wood pickaxe`
* `eat cow`
* `avoid zombie`
* `sleep when tired`

The initial version should use a **closed, short instruction vocabulary** rather than open-ended free-form text.

## Data Strategy

### Preferred v1 approach

Use **scripted / heuristic policies** to generate trajectories.

Why:

* fast,
* controllable,
* reproducible,
* avoids collecting human gameplay.

Each trajectory should include:

* observation frames,
* action labels,
* active instruction,
* episode metadata,
* optional success flags.

### Optional v1.1

Record a small amount of manual play data for qualitative comparison or fine-tuning.

### Avoid initially

* large human demonstration datasets,
* web video extraction,
* complex inverse action recovery.

## Model Design

Keep the first model very small and legible.

### Input encoders

#### Vision encoder

Options:

* small CNN,
* lightweight ResNet,
* tiny ViT only if it does not slow iteration too much.

Recommendation: start with a **small CNN or lightweight ResNet**.

#### Text encoder

Options:

* learned embedding over fixed instruction vocabulary,
* tiny text transformer,
* frozen sentence encoder.

Recommendation: start with a **learned instruction embedding** or very small text encoder.

### Fusion

Concatenate or cross-attend:

* visual features,
* text embedding,
* optional previous-action embedding.

Recommendation: start with **simple concatenation + MLP**.

### Output head

* categorical distribution over discrete actions.

### Optional temporal modeling

For v1:

* stack 4 frames, or
* add a tiny GRU.

Recommendation: start with **frame stack first**, GRU later if needed.

## Training Objective

### Primary objective

Behavior cloning / supervised imitation learning:

* cross-entropy loss over action labels.

### Optional auxiliary losses

Only add later if needed:

* instruction-task success prediction,
* contrastive alignment between instruction and observation,
* action-validity mask loss.

## Evaluation

### Quantitative

Track:

* action prediction accuracy,
* instruction-conditioned episode success rate,
* average reward,
* success by task,
* success by episode length,
* confusion matrix for actions.

### Qualitative

Produce:

* rollout videos,
* side-by-side target instruction + gameplay,
* failure case gallery,
* examples where language changes behavior.

## v1 Success Criteria

The project is successful if it can show:

1. the same visual state leads to different actions under different instructions,
2. the agent completes a few short Crafter tasks better than a trivial baseline,
3. the demo is understandable in under 30 seconds.

## Baselines

Implement at least these baselines:

* random policy,
* heuristic scripted policy,
* vision-only imitation model,
* text-conditioned model.

The key comparison is:
**vision-only vs instruction-conditioned**.

## Risks

### Main risk

The agent may just memorize local correlations and ignore language.

### Mitigations

* design tasks where instruction genuinely changes behavior,
* balance datasets across goals,
* measure instruction sensitivity,
* include same-state / different-instruction evaluation slices.

### Secondary risk

Crafter action semantics may still be harder than expected.

Mitigation:

* shrink task set,
* reduce action subset,
* create curriculum tasks.

## Suggested v1 Scope

### Narrow v1

Choose 3 tasks only:

* `collect wood`
* `collect stone`
* `place table`

This is enough for a real POC.

### Good v1.1

Expand to 5–7 tasks with simple crafting / survival behaviors.

---

# Option B: ViZDoom Combat Bot

## Working Title

**DoomVLA-Tiny**

## One-Sentence Description

A small instruction-conditioned visual combat agent in ViZDoom that maps first-person frames and tactical prompts to player-like actions.

## Why ViZDoom

ViZDoom is attractive because it offers:

* strong visual/gameplay appeal,
* mature RL / research tooling,
* FPS control dynamics,
* easier “bot” intuition for viewers,
* short reactive horizons for combat-focused tasks.

This option is likely more visually impressive, but somewhat less naturally language-grounded than Crafter.

## v1 Problem Formulation

Input:

* first-person RGB frame or frame stack,
* short instruction / tactical prompt,
* optional health/ammo features,
* optional previous action.

Output:

* discrete movement / combat action.

## Example Instructions

Start with simple tactical prompts such as:

* `attack nearest enemy`
* `strafe left and fire`
* `retreat when low health`
* `collect ammo`
* `seek medkit`
* `hold position`

## Why This Works

The language here acts less like open-ended planning and more like **policy selection / tactical conditioning**. That is still useful and honest.

## Data Strategy

### Preferred v1 approach

Generate data using:

* scripted combat heuristics,
* simple autopilot policies,
* hand-authored scenarios,
* optional short manual demos.

### Best scenario design

Use highly controlled arenas with one clear tactical objective per scenario.

Example:

* single enemy, small room, low health, medkit visible,
* multiple enemies, enough ammo, attack instruction,
* obstacle-rich room, strafe instruction.

## Model Design

Very similar to Crafter option:

* vision encoder,
* small text encoder,
* fusion block,
* action head.

Because FPS dynamics are more reactive, temporal modeling may matter more here.

Recommendation:

* frame stack first,
* optionally add GRU if needed.

## Evaluation

Track:

* survival time,
* kill count,
* damage dealt,
* instruction-conditioned success,
* resource pickup success,
* action prediction accuracy if imitation labels exist.

## v1 Success Criteria

The project is successful if the model behaves differently and correctly under tactical prompts like:

* attack,
* retreat,
* collect ammo,
* seek health.

## Risks

### Main risk

The demo looks cool, but language may be shallow.

Mitigation:

* build scenario pairs where the same visual state should produce different actions based on text.

### Secondary risk

FPS control can be noisy and sensitive.

Mitigation:

* reduce action space,
* use simple scenarios,
* start with short-horizon behavior only.

## Suggested v1 Scope

### Narrow v1

Choose 3 instructions only:

* `attack`
* `retreat`
* `collect ammo`

### Good v1.1

Add `seek medkit` and `hold position`.

---

# Recommendation Between the Two

## If the goal is strongest research / embodied-agent coherence

Choose **Crafter**.

## If the goal is strongest visual demo / game-bot feel

Choose **ViZDoom**.

## If the goal is fastest credible portfolio artifact

Still choose **Crafter first**.

Why:

* cleaner instruction-following story,
* better fit for survival / crafting goals,
* less likely to collapse into pure reflex combat,
* closer to recent embodied-agent foundations work.

---

# Technical Approach

## Initial Stack

* Python
* PyTorch
* Gymnasium-compatible environment wrappers where applicable
* Weights & Biases or MLflow for runs
* ffmpeg / moviepy for rollout videos
* simple YAML config system

## Keep the system intentionally simple

Avoid in v1:

* distributed training,
* heavy JAX infra,
* giant pretrained backbones,
* multi-node data pipelines,
* complex agent orchestration stacks.

This repo should feel compact and readable.

---

# Proposed Repository Structure

```text
project-root/
  README.md
  pyproject.toml
  requirements.txt
  configs/
    crafter_bc_small.yaml
    vizdoom_bc_small.yaml
  src/
    envs/
      crafter_env.py
      vizdoom_env.py
      wrappers.py
    data/
      schema.py
      collectors.py
      scripted_crafter.py
      scripted_vizdoom.py
      datasets.py
    models/
      vision.py
      text.py
      fusion.py
      policy.py
      temporal.py
    train/
      train_bc.py
      eval_bc.py
      metrics.py
    rollout/
      rollout.py
      render.py
    utils/
      config.py
      seed.py
      logging.py
  notebooks/
    exploratory_training.ipynb
  artifacts/
    videos/
    figures/
  docs/
    pre_spec.md
    experiment_log.md
```

---

# MVP Plan

## MVP-0: Environment + Data

* run environment locally,
* define reduced action space,
* implement scripted trajectory generator,
* save dataset in a clean schema.

## MVP-1: Pure imitation baseline

* train vision-only action model,
* confirm pipeline works end-to-end,
* generate rollout videos.

## MVP-2: Instruction-conditioned policy

* add text input,
* compare against vision-only baseline,
* test same-state / different-instruction behavior.

## MVP-3: Public portfolio polish

* short README,
* architecture diagram,
* sample videos,
* key findings,
* honest limitations.

---

# Deliverables

## Minimum deliverables

* working training pipeline,
* at least one trained checkpoint,
* rollout videos,
* comparison plots,
* short write-up.

## Strong portfolio deliverables

* ablation: vision-only vs vision+text,
* clean task definitions,
* deterministic scripted data generation,
* short blog-style explanation of why this is a VLA-style game prototype.

---

# Public README Narrative

## Suggested framing

This project explores whether a lightweight analogue of the Visual-Language-Action paradigm can be applied to games. Instead of robotics, the agent acts in a game environment. It observes pixels, reads a short goal, and predicts player-like actions. The focus is not scale, but fast prototyping, instruction-conditioning, and portfolio-quality evidence that multimodal policy learning can be meaningfully applied to game environments.

---

# Explicit Scope Cuts

To keep this project small, the following are deliberately excluded from v1:

* open-ended natural language,
* long-horizon planning,
* RL fine-tuning,
* world models,
* multi-agent setups,
* procedural content generation,
* transfer across games,
* LLM-based reasoning loops.

These can be listed later under “future work.”

---

# Future Extensions

## Extension A: Tiny DAgger loop

Use the scripted teacher to relabel failure states and reduce compounding error.

## Extension B: Contrastive grounding

Learn better instruction-observation alignment using a CLIP-like auxiliary loss.

## Extension C: Temporal policy

Swap frame stacking for GRU / transformer-lite temporal modeling.

## Extension D: Small PCG follow-up

Use a separate tiny model to generate task layouts, arenas, or resource placements for evaluation.

## Extension E: CrafterDojo-inspired direction

Reuse ideas from foundation-model tooling around Crafter for stronger instruction-following or behavior priors.

---

# Final Decision Guidance

If starting today, the recommended sequence is:

1. build **Crafter MVP**,
2. finish a small, honest, complete prototype,
3. only then decide whether to add a **ViZDoom combat variant**.

The main rule is:
**finish one narrow, legible project before expanding scope.**

---

# Author Notes

This project should optimize for:

* clarity,
* finishability,
* visible ML depth,
* visible game relevance,
* honest scoping.

The best version of this repository is not the most ambitious one. It is the one that clearly demonstrates a modern ML idea applied well to a game environment, with a scope small enough to actually complete.
