# Spec: Hello Crafter (MVP-0a)

## Status

Approved

## Goal

Get Crafter running with a thin environment wrapper and prove the pipeline works by executing a random-policy rollout that saves frames to disk. This is the foundation every later milestone builds on — environment, action space, observation shape, and rendering are all pinned here.

---

## Scope

### In scope

- Project scaffolding (`pyproject.toml`, package layout)
- Gymnasium-compatible Crafter wrapper with a reduced action space
- Random-policy rollout script that saves frames and/or video
- Seed-controlled reproducibility

### Out of scope

- Models, training, data collection
- Scripted/heuristic policies (that's MVP-0b)
- Dataset schema or storage
- Config system (hardcode everything for now)
- Notebooks, docs, README

---

## 1. Project Scaffolding

### `pyproject.toml`

```toml
[project]
name = "vla-game-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "crafter>=1.8",
    "torch>=2.0",
    "gymnasium>=0.29",
    "numpy>=1.24",
    "imageio[ffmpeg]>=2.31",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Dependency notes:
- `crafter` is the environment.
- `torch` is listed now so the venv is ready for MVP-1; nothing in this milestone imports it.
- `gymnasium` for the wrapper interface.
- `imageio[ffmpeg]` for saving rollout videos.
- Exact version pins may be adjusted during implementation as long as the minimum versions above are respected.

### Package layout

```
src/
  vla_agent/
    __init__.py
    envs/
      __init__.py
      crafter_env.py
scripts/
  random_rollout.py
```

- The package is `vla_agent`.
- Only the files listed above are created in this milestone. Do not create placeholder files for future modules.

---

## 2. Crafter Environment Wrapper

**File:** `src/vla_agent/envs/crafter_env.py`

### 2.1 Full Crafter Action Space (reference)

Crafter exposes 17 discrete actions (indices 0–16):

| Index | Action |
|-------|--------|
| 0 | noop |
| 1 | move_left |
| 2 | move_right |
| 3 | move_up |
| 4 | move_down |
| 5 | do (interact/attack) |
| 6 | sleep |
| 7 | place_stone |
| 8 | place_table |
| 9 | place_furnace |
| 10 | place_plant |
| 11 | make_wood_pickaxe |
| 12 | make_stone_pickaxe |
| 13 | make_iron_pickaxe |
| 14 | make_wood_sword |
| 15 | make_stone_sword |
| 16 | make_iron_sword |

### 2.2 Reduced Action Space

For the narrow v1 tasks (`collect wood`, `collect stone`, `place table`), only the following actions are needed:

| Reduced Index | Full Index | Action | Why kept |
|---------------|-----------|--------|----------|
| 0 | 0 | noop | baseline / idle |
| 1 | 1 | move_left | navigation |
| 2 | 2 | move_right | navigation |
| 3 | 3 | move_up | navigation |
| 4 | 4 | move_down | navigation |
| 5 | 5 | do | interact with world objects (mine trees, rocks) |
| 6 | 8 | place_table | crafting task |

Total: **7 actions** (indices 0–6 in the reduced space).

### 2.3 Wrapper Class

```python
class CrafterEnv:
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `0` | RNG seed for the Crafter environment |
| `image_size` | `tuple[int, int]` | `(64, 64)` | Observation resize target (H, W) |

**Gymnasium-style interface — required methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `reset()` | `tuple[np.ndarray, dict]` | Returns `(obs, info)`. `obs` is RGB uint8 `(H, W, 3)`. `info` contains at minimum `{"inventory": dict, "achievements": dict}`. |
| `step(action: int)` | `tuple[np.ndarray, float, bool, bool, dict]` | Takes a **reduced** action index (0–6), maps it to the full Crafter action, steps the env. Returns `(obs, reward, terminated, truncated, info)`. |
| `close()` | `None` | Cleans up the environment. |

**Required properties / attributes:**

| Name | Type | Description |
|------|------|-------------|
| `action_names` | `list[str]` | `["noop", "move_left", "move_right", "move_up", "move_down", "do", "place_table"]` |
| `num_actions` | `int` | `7` |

**Action mapping behavior:**

- `step()` must accept only reduced-space indices (0–6).
- If an out-of-range action is passed, raise `ValueError`.
- Internally, map the reduced index to the corresponding full Crafter action index before calling the underlying environment.

**Observation handling:**

- Crafter natively renders at 64×64. If `image_size` matches native resolution, no resize is needed.
- If `image_size` differs from native, resize using area interpolation (numpy or PIL).
- Observations are always returned as `np.ndarray` with dtype `uint8` and shape `(H, W, 3)`.

### 2.4 Implementation Notes

- Crafter's own API is `crafter.Env(seed=seed)`. It is **not** a Gymnasium env — it has `env.reset()` → `obs` and `env.step(action)` → `(obs, reward, done, info)`. The wrapper must adapt this to the 5-tuple Gymnasium convention.
- Crafter does not distinguish `terminated` vs `truncated`. Set `terminated = done` and `truncated = False`.
- The Crafter `info` dict includes `inventory` and `achievements` — pass these through.

---

## 3. Random Rollout Script

**File:** `scripts/random_rollout.py`

### 3.1 Behavior

Run one episode of Crafter with a random policy (uniform over the reduced action space). Save the results to an output directory.

### 3.2 CLI Interface

The script must be runnable via:

```bash
uv run python scripts/random_rollout.py --seed 42 --output-dir artifacts/rollouts/test_run
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | `int` | `42` | Seed for both env and action sampling |
| `--output-dir` | `str` | `artifacts/rollouts/run` | Directory for output files |
| `--max-steps` | `int` | `200` | Max steps per episode (Crafter episodes can be long) |
| `--save-video` | flag | off | If set, also save an mp4 video |

### 3.3 Outputs

The script must create `output-dir/` containing:

1. **`frames/`** — directory of PNG images named `frame_000.png`, `frame_001.png`, etc. Each is the raw RGB observation from the env.
2. **`episode.json`** — episode metadata:
   ```json
   {
     "seed": 42,
     "num_steps": 153,
     "total_reward": 2.0,
     "actions_taken": [3, 1, 5, 0, ...],
     "action_names_taken": ["move_up", "move_left", "do", "noop", ...],
     "final_inventory": {"wood": 3, "stone": 1},
     "achievements": {"collect_wood": 1}
   }
   ```
3. **`rollout.mp4`** (only if `--save-video`) — video of the episode at 10 fps.

### 3.4 Console Output

While running, print one line per step:

```
Step 003 | action: do (5) | reward: 1.0 | total: 1.0
```

At the end, print a summary:

```
Episode done: 153 steps, total reward 2.0
Saved 153 frames to artifacts/rollouts/test_run/frames/
Saved episode metadata to artifacts/rollouts/test_run/episode.json
```

---

## 4. Acceptance Criteria

These are the concrete requirements that tests and review should verify.

### AC-1: Package installs cleanly

- `uv sync --all-extras` succeeds (UV is the only package manager — never use pip).
- `from vla_agent.envs.crafter_env import CrafterEnv` works.

### AC-2: CrafterEnv has correct interface

- `env = CrafterEnv(seed=0)` creates an environment.
- `env.num_actions == 7`.
- `env.action_names` returns the 7-element list in the order specified.
- `reset()` returns `(obs, info)` where `obs.shape == (64, 64, 3)` and `obs.dtype == np.uint8`.
- `step(action)` returns a 5-tuple `(obs, reward, terminated, truncated, info)` with correct types.
- `step(7)` raises `ValueError`.
- `step(-1)` raises `ValueError`.

### AC-3: Action mapping is correct

- Stepping with reduced action `6` (place_table) maps to Crafter action index `8`.
- Stepping with reduced action `5` (do) maps to Crafter action index `5`.
- Stepping with reduced action `0` (noop) maps to Crafter action index `0`.

### AC-4: Episode lifecycle works

- An episode can run to completion: call `reset()`, then `step()` repeatedly until `terminated == True` or a step limit is reached.
- Calling `reset()` after an episode ends starts a new episode.
- `close()` does not raise.

### AC-5: Seed reproducibility

- Two `CrafterEnv(seed=42)` instances, both reset and stepped with the same actions, produce identical observations (byte-equal numpy arrays).

### AC-6: Random rollout script produces correct outputs

- Running the script creates the output directory.
- `frames/` contains at least 1 PNG file.
- Each PNG is a valid image loadable by `imageio` with shape `(64, 64, 3)`.
- `episode.json` is valid JSON containing all required keys.
- `episode.json["num_steps"]` equals the number of frames saved.
- `episode.json["actions_taken"]` has length equal to `num_steps`.
- `episode.json["action_names_taken"]` has length equal to `num_steps` and every entry is in the valid action names list.

### AC-7: Video output (when flag is set)

- With `--save-video`, `rollout.mp4` is created in the output directory.
- The file is a valid video readable by `imageio`.

---

## 5. Testing Notes

### What to unit-test (no Crafter dependency)

- Action name list and count.
- Action index mapping logic (reduced → full).
- Out-of-range action validation.
- Episode metadata JSON structure.

### What to integration-test (requires Crafter)

- Full `reset()` / `step()` / `close()` lifecycle.
- Observation shape and dtype.
- Seed reproducibility.
- Rollout script end-to-end.

Mark integration tests with `@pytest.mark.integration` so they can be skipped in fast CI if needed, but they must pass locally.

---

## 6. Non-Functional Requirements

- **UV only** — use `uv sync`, `uv run pytest`, `uv run python`. Never pip.
- No global state. All randomness flows from explicit seeds.
- No print statements in library code (`src/`). Only the script (`scripts/`) prints.
- Imports must work on Python 3.11+.
- No dependencies beyond what is listed in `pyproject.toml`.
