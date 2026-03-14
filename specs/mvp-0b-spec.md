# Spec: Scripted Policies & Trajectory Data (MVP-0b)

## Status

Approved

## Goal

Build three scripted expert policies for basic Crafter tasks and a data-collection script that runs them and saves trajectory data to disk. This produces the training dataset that MVP-1 (vision-only imitation baseline) will learn from.

The three tasks — `collect wood`, `place table`, `collect stone` — form a natural difficulty progression that mirrors Crafter's tech tree. Each policy demonstrates goal-directed behavior the VLA agent must later reproduce from pixels and text instructions alone.

---

## Scope

### In scope

- Expand CrafterEnv action space from 7 to 8 (add `make_wood_pickaxe`)
- Expand CrafterEnv info dict (add `player_pos`, `player_facing` to `reset()`)
- Three scripted policies: `collect_wood`, `place_table`, `collect_stone`
- Shared greedy navigation utility used by all three policies
- Trajectory storage format (`.npz` + `manifest.json`)
- Data collection script (`scripts/collect_trajectories.py`)

### Out of scope

- Models, training, loss functions (MVP-1)
- Complex pathfinding (A*, BFS) — greedy is sufficient
- Config system — hardcode everything
- Policies beyond the three listed
- Full 17-action Crafter space — only add the one action needed
- Notebooks, docs, README

---

## Crafter Mechanics Reference

This section documents the Crafter engine internals the policies depend on. All values come from `crafter/data.yaml` and the crafter source (v1.8+).

### Material indices (world map)

The world is a 64x64 grid. Each cell holds a material index:

| Index | Material | Walkable | Notes |
|-------|----------|----------|-------|
| 0 | water | no | `do` on water restores thirst |
| 1 | grass | yes | `do` on grass has 10% chance to yield 1 sapling |
| 2 | stone | no | `do` yields 1 stone (requires `wood_pickaxe`) |
| 3 | path | yes | left behind after mining stone/coal/iron/diamond |
| 4 | sand | yes | |
| 5 | tree | no | `do` yields 1 wood, cell becomes grass |
| 6 | lava | no | |
| 7 | coal | no | `do` yields 1 coal (requires `wood_pickaxe`) |
| 8 | iron | no | `do` yields 1 iron (requires `stone_pickaxe`) |
| 9 | diamond | no | `do` yields 1 diamond (requires `iron_pickaxe`) |
| 10 | table | no | crafting station, placed by player |
| 11 | furnace | no | smelting station, placed by player |

Walkable cells for table placement: grass (1), sand (4), path (3).

### Facing and interaction model

- Every move action updates `player.facing` to that direction, **regardless of whether the player actually moves**.
- The player moves only if the target cell is walkable and unoccupied by an object.
- Moving toward a non-walkable cell (tree, stone, water) updates facing without moving.
- `do` (action 5) operates on the cell at `player.pos + player.facing`.
- `place_table` (action 8) places a table at `player.pos + player.facing`. Requires 2 wood. Target cell must be grass/sand/path.
- `make_wood_pickaxe` (action 11) crafts a pickaxe. Requires 1 wood and a table within the 3x3 neighborhood centered on the player.

### Coordinate convention

- `player.pos` is `(x, y)` where x = column (horizontal), y = row (vertical).
- `move_left`: facing = `(-1, 0)`, `move_right`: facing = `(+1, 0)`.
- `move_up`: facing = `(0, -1)`, `move_down`: facing = `(0, +1)`.
- World material array: verify indexing convention against crafter source during implementation (`world[x, y]` or `world[y, x]`).

### Reward

- `+1.0` for each **first-time** achievement unlock per episode.
- `(current_health - previous_health) / 10` per step for health changes.

---

## 1. CrafterEnv Changes

### 1.1 Action Space (7 → 8)

Add `make_wood_pickaxe` as reduced action index 7.

| Reduced Index | Full Index | Action | Status |
|---------------|-----------|--------|--------|
| 0 | 0 | noop | existing |
| 1 | 1 | move_left | existing |
| 2 | 2 | move_right | existing |
| 3 | 3 | move_up | existing |
| 4 | 4 | move_down | existing |
| 5 | 5 | do | existing |
| 6 | 8 | place_table | existing |
| 7 | 11 | make_wood_pickaxe | **new** |

Updated class attributes:

```python
num_actions = 8
action_names = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "place_table", "make_wood_pickaxe",
]
```

`step()` must accept indices 0–7. `step(8)` and above must raise `ValueError`.

**Why only this one action?** `collect_stone` requires a wood pickaxe, which requires the `make_wood_pickaxe` action. The other two policies (`collect_wood`, `place_table`) work within the existing 7 actions. No other new actions are needed for MVP-0b.

### 1.2 Info Dict Expansion

Both `reset()` and `step()` must return info dicts containing:

| Key | Type | Description | Status |
|-----|------|-------------|--------|
| `inventory` | `dict[str, int]` | Player inventory (item name → count) | existing |
| `achievements` | `dict[str, int]` | Achievement name → unlock count | existing |
| `player_pos` | `tuple[int, int]` | `(x, y)` position in world coordinates | **new in reset()** (already present in step() from native crafter) |
| `player_facing` | `tuple[int, int]` | `(dx, dy)` facing direction | **new** |

`step()` already passes through the native crafter info which includes `player_pos`. The change is:
1. Add `player_pos` to `reset()` info (extract from `self._env._player.pos`).
2. Add `player_facing` to both `reset()` and `step()` info (extract from `self._env._player.facing`).

### 1.3 Breaking Changes from MVP-0a

The action space expansion (7 → 8) requires updating these MVP-0a tests:

- `test_num_actions` — expects 7, now 8
- `test_action_names_type_and_length` — expects length 7, now 8
- `test_action_names_exact_order` — expects 7-element list, now 8
- `test_step_out_of_range_*` — boundary shifts from 7 to 8
- `test_step_all_valid_reduced_actions` — iterates 0–6, now 0–7

These tests must be updated as part of MVP-0b to reflect the new 8-action space.

---

## 2. Scripted Policy Interface

**File:** `src/vla_agent/policies.py`

### 2.1 Base Protocol

All scripted policies implement this interface:

```python
class ScriptedPolicy(Protocol):
    @property
    def instruction(self) -> str:
        """Text instruction this policy follows (e.g. 'collect wood')."""
        ...

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        ...

    def act(self, obs: np.ndarray, info: dict) -> int:
        """Return a reduced action index given the current observation and info."""
        ...

    def succeeded(self, info: dict) -> bool:
        """Return True if the policy has achieved its goal."""
        ...
```

### 2.2 Contract

- `act()` returns a **reduced** action index (0–7), not a full Crafter action.
- `act()` is called once per timestep. The policy may maintain internal state (phase, target, stuck counter) across calls.
- `succeeded()` checks the `info` dict (inventory/achievements) to determine if the goal is met.
- `reset()` must be called before the first `act()` of each episode.
- Policies are **deterministic** given the same env seed — no internal randomness, tie-breaking is by coordinate order.

### 2.3 World Access

Scripted policies have **privileged access** to the full world state for navigation. They receive a reference to the `CrafterEnv` at construction time and may access `env._env._world` and `env._env._player` directly. This is explicitly allowed for expert data generation — the VLA model trained on this data will only see pixel observations.

### 2.4 Navigation Utility

All three policies share a common greedy navigation function. Given a target cell position, it returns the best movement action:

1. Compute the delta from player position to target: `(dx, dy)`.
2. Choose the axis with larger absolute delta as primary.
3. Return the movement action along the primary axis (e.g., `move_right` if `dx > 0`).
4. **Stuck detection:** if the player's position has not changed after a move (blocked by non-walkable terrain or object), switch to the secondary axis on the next call.
5. If stuck on both axes for a configurable number of steps (e.g., 5), pick an alternative target.

This greedy approach is sufficient because trees are abundant and stone deposits are common in Crafter worlds.

---

## 3. Policy: CollectWood

**Instruction:** `"collect wood"`

### 3.1 Success Condition

`info["inventory"]["wood"] >= 1` (player has at least 1 wood that was not present at episode start).

### 3.2 Actions Used

`move_left`, `move_right`, `move_up`, `move_down`, `do`.

### 3.3 Algorithm

**Phase 1 — Find target:** Scan the world material array for cells with material index 5 (tree). Select the nearest tree by Manhattan distance from the player. If multiple trees are equidistant, prefer the one with the smallest x, then smallest y.

**Phase 2 — Navigate:** Use the greedy navigation utility to move toward an adjacent cell of the target tree. "Adjacent" means one of the 4 cardinal neighbors.

**Phase 3 — Interact:** Once adjacent to the tree, issue a move action toward the tree. Because trees are non-walkable, this updates the player's facing without moving. Then issue `do`. The tree is chopped, yielding 1 wood and converting the cell to grass.

In practice, phases 2 and 3 merge naturally: the final navigation step is a move toward the non-walkable tree, which sets facing. The next action is `do`.

---

## 4. Policy: PlaceTable

**Instruction:** `"place table"`

### 4.1 Success Condition

`info["achievements"].get("place_table", 0) >= 1`.

### 4.2 Actions Used

`move_left`, `move_right`, `move_up`, `move_down`, `do`, `place_table`.

### 4.3 Prerequisites

2 wood in inventory. The policy must collect wood before placing the table.

### 4.4 Algorithm

**Phase 1 — Collect wood:** If `inventory["wood"] < 2`, execute the collect-wood behavior (same logic as CollectWoodPolicy) until 2 wood is held.

**Phase 2 — Position for placement:** The player needs to face a valid placement cell (grass, sand, or path — material indices 1, 4, 3). Since the player is standing on a walkable cell and surrounded by walkable terrain in most Crafter spawn areas, the last movement direction from Phase 1 typically already faces a valid cell. The policy checks:

1. Look at the cell at `player.pos + player.facing`.
2. If that cell is grass/sand/path and unoccupied by an object: proceed to Phase 3.
3. Otherwise: try each of the 4 cardinal directions by issuing a move. After each move, check the new facing cell. Since grass is the dominant terrain, this resolves within a few steps.

**Phase 3 — Place:** Issue `place_table` (reduced action 6). The table appears at the faced cell, consuming 2 wood.

---

## 5. Policy: CollectStone

**Instruction:** `"collect stone"`

### 5.1 Success Condition

`info["inventory"]["stone"] >= 1`.

### 5.2 Actions Used

`move_left`, `move_right`, `move_up`, `move_down`, `do`, `place_table`, `make_wood_pickaxe`.

### 5.3 Prerequisites

Mining stone requires a `wood_pickaxe`. Crafting a wood pickaxe requires a nearby table and 1 wood. Placing a table requires 2 wood. Total: 3 wood before any crafting.

### 5.4 Algorithm

**Phase 1 — Collect 3 wood:** Use collect-wood behavior repeatedly until `inventory["wood"] >= 3`.

**Phase 2 — Place table:** Use place-table behavior. Consumes 2 wood (1 remaining).

**Phase 3 — Craft pickaxe:** The player is adjacent to the just-placed table (it was placed at the faced cell). Issue `make_wood_pickaxe` (reduced action 7). Requires 1 wood + nearby table (3x3 check). Consumes 1 wood, grants 1 wood_pickaxe.

**Phase 4 — Find stone:** Scan the world material array for cells with material index 2 (stone). Select the nearest stone by Manhattan distance.

**Phase 5 — Navigate and mine:** Navigate to be adjacent to the stone target. Face the stone (move toward it — non-walkable, so facing updates). Issue `do`. Stone is mined, yielding 1 stone.

---

## 6. Trajectory Storage Format

### 6.1 Episode File: `episode_NNN.npz`

A compressed NumPy archive containing:

| Array Key | Shape | Dtype | Description |
|-----------|-------|-------|-------------|
| `observations` | `(T+1, 64, 64, 3)` | `uint8` | `observations[0]` = from `reset()`. `observations[t+1]` = from `step(actions[t])`. |
| `actions` | `(T,)` | `int32` | Reduced action index at each timestep. |
| `rewards` | `(T,)` | `float32` | Reward received at each timestep. |

Where `T` = number of steps taken.

**Convention:** for imitation learning, the training pair at timestep `t` is `(observations[t], actions[t])` — the observation the agent saw before choosing the action. `observations[T]` is the terminal observation (no corresponding action).

### 6.2 Manifest File: `manifest.json`

One manifest per policy collection run:

```json
{
  "policy": "collect_wood",
  "instruction": "collect wood",
  "action_space_size": 8,
  "observation_shape": [64, 64, 3],
  "num_episodes": 10,
  "base_seed": 42,
  "success_count": 10,
  "episodes": [
    {
      "file": "episode_000.npz",
      "seed": 42,
      "success": true,
      "num_steps": 47,
      "total_reward": 1.0
    }
  ]
}
```

Required fields per episode: `file`, `seed`, `success`, `num_steps`, `total_reward`.

---

## 7. Data Collection Script

**File:** `scripts/collect_trajectories.py`

### 7.1 CLI

```bash
uv run python scripts/collect_trajectories.py \
    --policy collect_wood \
    --num-episodes 10 \
    --base-seed 42 \
    --max-steps 300 \
    --output-dir artifacts/trajectories/collect_wood
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--policy` | `str` | required | Policy name: `collect_wood`, `place_table`, or `collect_stone` |
| `--num-episodes` | `int` | `10` | Number of episodes to run |
| `--base-seed` | `int` | `42` | Base seed. Episode `i` uses seed `base_seed + i` |
| `--max-steps` | `int` | `300` | Max steps per episode (early-stop on success or termination) |
| `--output-dir` | `str` | `artifacts/trajectories/<policy>` | Output directory |

### 7.2 Behavior

For each episode:

1. Create `CrafterEnv(seed=base_seed + i)`.
2. Create the specified policy with a reference to the env.
3. `obs, info = env.reset()` and `policy.reset()`.
4. Store `obs` as `observations[0]`.
5. Loop up to `max_steps`:
   - `action = policy.act(obs, info)`
   - `obs, reward, terminated, truncated, info = env.step(action)`
   - Append `obs`, `action`, `reward`.
   - Break if `policy.succeeded(info)` or `terminated` or `truncated`.
6. Save `episode_NNN.npz`.
7. Record episode metadata for the manifest.

After all episodes: write `manifest.json` and print summary.

### 7.3 Console Output

Per-episode summary line:

```
Episode 001/010 | seed=42 | steps=47 | reward=1.0 | success=true
```

Final summary:

```
Done. Saved 10 episodes to artifacts/trajectories/collect_wood/
Success rate: 10/10 (100.0%)
```

### 7.4 Output Directory Structure

```
artifacts/trajectories/collect_wood/
  manifest.json
  episode_000.npz
  episode_001.npz
  ...
  episode_009.npz
```

---

## 8. Package Layout

New and modified files:

```
src/vla_agent/
  envs/
    crafter_env.py         # MODIFIED: 8 actions, expanded info dict
  policies.py              # NEW: ScriptedPolicy protocol, 3 policies, nav utility
scripts/
  random_rollout.py        # UNCHANGED
  collect_trajectories.py  # NEW: data collection script
```

No new dependencies. Everything uses `numpy`, `crafter`, and standard library.

---

## 9. Acceptance Criteria

### AC-1: CrafterEnv has 8 actions

- `env.num_actions == 8`.
- `env.action_names` is the 8-element list in the order specified in Section 1.1.
- `step(7)` (make_wood_pickaxe) does not raise.
- `step(8)` raises `ValueError`.

### AC-2: Action mapping is correct

- Reduced action 7 maps to full Crafter action 11 (make_wood_pickaxe).
- All prior mappings (0→0, 1→1, ..., 6→8) remain unchanged.

### AC-3: Info dict includes player state

- `reset()` info contains `player_pos` (tuple of 2 ints) and `player_facing` (tuple of 2 ints).
- `step()` info contains `player_pos` and `player_facing`.

### AC-4: CollectWoodPolicy succeeds

- Running the policy for a single episode (seed=42, max_steps=300) results in `inventory["wood"] >= 1`.
- The policy completes within 300 steps.

### AC-5: PlaceTablePolicy succeeds

- Running the policy for a single episode (seed=42, max_steps=300) results in `achievements["place_table"] >= 1`.
- The policy completes within 300 steps.

### AC-6: CollectStonePolicy succeeds

- Running the policy for a single episode (seed=42, max_steps=300) results in `inventory["stone"] >= 1`.
- The policy completes within 300 steps.

### AC-7: Policy determinism

- Two runs of the same policy with the same env seed produce identical action sequences.

### AC-8: Trajectory files are well-formed

- Each `.npz` file contains `observations`, `actions`, `rewards` with correct shapes and dtypes per Section 6.1.
- `observations.shape[0] == actions.shape[0] + 1`.
- `actions.shape[0] == rewards.shape[0]`.
- All action values are in `[0, 7]`.

### AC-9: Manifest is well-formed

- `manifest.json` is valid JSON with all required fields per Section 6.2.
- `len(episodes) == num_episodes`.
- Each episode's `num_steps` matches the corresponding `.npz` file's `actions.shape[0]`.
- Each episode's `file` field points to an existing `.npz` file.

### AC-10: Data collection script end-to-end

- `uv run python scripts/collect_trajectories.py --policy collect_wood --num-episodes 3 --base-seed 42` runs without error.
- Output directory contains `manifest.json` and 3 `.npz` files.
- At least 2 out of 3 episodes succeed (collect_wood should have near-100% success rate).

### AC-11: All three policies produce valid trajectories

- Run the collection script for each of the 3 policies (3 episodes each).
- All manifest files are valid.
- Success rates: `collect_wood` >= 90%, `place_table` >= 80%, `collect_stone` >= 70%.

---

## 10. Testing Notes

### Unit tests (no Crafter dependency)

- Action space: 8 actions, correct names, correct mapping (including new action 7→11).
- Out-of-range action validation: `step(8)` raises, `step(-1)` raises.
- Policy protocol: each policy class has `instruction`, `reset`, `act`, `succeeded`.
- Trajectory manifest JSON schema validation.

### Integration tests (require Crafter, mark with `@pytest.mark.integration`)

- CrafterEnv `reset()` info contains `player_pos` and `player_facing`.
- CrafterEnv `step()` info contains `player_pos` and `player_facing`.
- `make_wood_pickaxe` action integration: place table, craft pickaxe, verify `inventory["wood_pickaxe"] >= 1`.
- Each policy succeeds within 300 steps (seed=42).
- Policy determinism: same seed produces same action sequence.
- Collection script end-to-end: runs, produces expected output files.
- Trajectory file shapes and dtypes.

### Tests to update from MVP-0a

See Section 1.3 — all tests that hardcode `num_actions == 7` or the 7-element action list must be updated to 8.

---

## 11. Non-Functional Requirements

- **UV only** — `uv sync`, `uv run pytest`, `uv run python`. Never pip.
- **No global state.** All randomness flows from explicit env seeds. Policies are deterministic.
- **No print in library code** (`src/`). Only scripts (`scripts/`) print.
- **No new dependencies.** Everything uses numpy, crafter, and the standard library.
- **Seed reproducibility.** Same `base_seed` + same policy = identical trajectories.
- **Imports:** absolute from `vla_agent` (e.g., `from vla_agent.policies import CollectWoodPolicy`).
