"""Scripted expert policies for Crafter instruction-conditioned data collection."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from vla_agent.envs.crafter_env import CrafterEnv

Position = tuple[int, int]
Direction = tuple[int, int]

NOOP_ACTION = 0
MOVE_ACTIONS: dict[Direction, int] = {
    (-1, 0): 1,  # move_left
    (1, 0): 2,  # move_right
    (0, -1): 3,  # move_up
    (0, 1): 4,  # move_down
}
DO_ACTION = 5
PLACE_TABLE_ACTION = 6
MAKE_WOOD_PICKAXE_ACTION = 7
CARDINAL_DIRECTIONS: list[Direction] = [(0, 1), (0, -1), (-1, 0), (1, 0)]
WALKABLE_TABLE_MATERIALS = {"grass", "path", "sand"}


class ScriptedPolicy(Protocol):
    """Protocol all scripted policies must follow."""

    @property
    def instruction(self) -> str: ...

    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, info: dict) -> int: ...

    def succeeded(self, info: dict) -> bool: ...


class GreedyNavigator:
    """Greedy navigator with simple stuck detection and axis switching."""

    def __init__(self, stuck_limit: int = 5) -> None:
        self._stuck_limit = stuck_limit
        self.reset()

    def reset(self) -> None:
        self._last_player_pos: Position | None = None
        self._last_move_direction: Direction | None = None
        self._stuck_steps = 0

    def next_action(self, player_pos: Position, target_pos: Position) -> int:
        if self._last_move_direction is not None and self._last_player_pos == player_pos:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        dx = target_pos[0] - player_pos[0]
        dy = target_pos[1] - player_pos[1]
        if dx == 0 and dy == 0:
            self._last_move_direction = None
            self._last_player_pos = player_pos
            return NOOP_ACTION

        primary_axis = "x" if abs(dx) >= abs(dy) else "y"
        axis = primary_axis
        if self._last_move_direction is not None and self._last_player_pos == player_pos:
            last_axis = "x" if self._last_move_direction in {(-1, 0), (1, 0)} else "y"
            if last_axis == primary_axis:
                axis = "y" if primary_axis == "x" else "x"

        direction = self._direction_for_axis(axis, dx, dy)
        if direction is None:
            axis = "y" if axis == "x" else "x"
            direction = self._direction_for_axis(axis, dx, dy)

        if direction is None:
            action = NOOP_ACTION
            self._last_move_direction = None
        else:
            action = MOVE_ACTIONS[direction]
            self._last_move_direction = direction

        self._last_player_pos = player_pos
        return action

    def should_retarget(self) -> bool:
        return self._stuck_steps >= self._stuck_limit

    @staticmethod
    def _direction_for_axis(axis: str, dx: int, dy: int) -> Direction | None:
        if axis == "x":
            if dx > 0:
                return (1, 0)
            if dx < 0:
                return (-1, 0)
        else:
            if dy > 0:
                return (0, 1)
            if dy < 0:
                return (0, -1)
        return None


class _WorldHelper:
    """Convenience wrapper for frequently accessed Crafter world data."""

    def __init__(self, env: CrafterEnv) -> None:
        self._env = env
        self._material_ids: dict[str, int] = {}

    @property
    def world(self):
        return self._env._env._world

    def in_bounds(self, pos: Position) -> bool:
        area = self.world.area
        return 0 <= pos[0] < int(area[0]) and 0 <= pos[1] < int(area[1])

    def material_id(self, name: str) -> int:
        if name not in self._material_ids:
            for idx, mat_name in self.world._mat_names.items():
                if mat_name == name:
                    self._material_ids[name] = idx
                    break
            else:
                raise ValueError(f"Material '{name}' not found in Crafter world.")
        return self._material_ids[name]

    def nearest_material(
        self, name: str, player_pos: Position, exclude: set[Position] | None = None
    ) -> Position | None:
        mat_id = self.material_id(name)
        coords = np.argwhere(self.world._mat_map == mat_id)
        best: Position | None = None
        best_score: tuple[int, int, int] | None = None
        for x, y in coords:
            pos = (int(x), int(y))
            if exclude and pos in exclude:
                continue
            score = (_manhattan(player_pos, pos), pos[0], pos[1])
            if best_score is None or score < best_score:
                best_score = score
                best = pos
        return best

    def material_matches(self, pos: Position, name: str) -> bool:
        if not self.in_bounds(pos):
            return False
        mat_id = self.material_id(name)
        return int(self.world._mat_map[pos[0], pos[1]]) == mat_id

    def cell_is_walkable_for_table(self, pos: Position) -> bool:
        if not self.in_bounds(pos):
            return False
        material, obj = self.world[(pos[0], pos[1])]
        return material in WALKABLE_TABLE_MATERIALS and obj is None


class _WoodCollectionBehavior:
    """Reusable wood-collecting routine shared across policies."""

    def __init__(
        self,
        env: CrafterEnv,
        target_wood: int,
        world_helper: _WorldHelper | None = None,
        stuck_limit: int = 5,
    ) -> None:
        self._env = env
        self._required = target_wood
        self._world = world_helper or _WorldHelper(env)
        self._navigator = GreedyNavigator(stuck_limit=stuck_limit)
        self._target_tree: Position | None = None
        self._excluded: set[Position] = set()

    def reset(self) -> None:
        self._navigator.reset()
        self._target_tree = None
        self._excluded.clear()

    def needs_more(self, info: dict) -> bool:
        return info.get("inventory", {}).get("wood", 0) < self._required

    def act(self, info: dict) -> int:
        player_pos = _player_pos_from_info(info, self._env)
        if self._target_tree is None or not self._world.material_matches(self._target_tree, "tree"):
            self._target_tree = self._world.nearest_material(
                "tree", player_pos, exclude=self._excluded
            )
            self._navigator.reset()
        if self._target_tree is None:
            return NOOP_ACTION
        if self._navigator.should_retarget():
            self._excluded.add(self._target_tree)
            self._target_tree = self._world.nearest_material(
                "tree", player_pos, exclude=self._excluded
            )
            self._navigator.reset()
            if self._target_tree is None:
                return NOOP_ACTION
        if not _is_adjacent(player_pos, self._target_tree):
            return self._navigator.next_action(player_pos, self._target_tree)
        direction = _direction_toward(player_pos, self._target_tree)
        facing = _player_facing_from_info(info, self._env)
        if facing == direction:
            return DO_ACTION
        return MOVE_ACTIONS[direction]


class _TablePlacementBehavior:
    """Faces a valid placement cell, then issues the place_table action."""

    def __init__(self, env: CrafterEnv, world_helper: _WorldHelper | None = None) -> None:
        self._env = env
        self._world = world_helper or _WorldHelper(env)
        self._direction_queue: list[Direction] = []

    def reset(self) -> None:
        self._direction_queue.clear()

    def act(self, info: dict) -> int:
        player_pos = _player_pos_from_info(info, self._env)
        facing = _player_facing_from_info(info, self._env)
        if self._cell_is_placeable(player_pos, facing):
            return PLACE_TABLE_ACTION
        if not self._direction_queue:
            self._direction_queue = [d for d in CARDINAL_DIRECTIONS if d != facing] or list(
                CARDINAL_DIRECTIONS
            )
        direction = self._direction_queue.pop(0)
        return MOVE_ACTIONS[direction]

    def _cell_is_placeable(self, player_pos: Position, facing: Direction) -> bool:
        target = (player_pos[0] + facing[0], player_pos[1] + facing[1])
        return self._world.cell_is_walkable_for_table(target)


class CollectWoodPolicy:
    """Scripted policy that chops the nearest tree to collect wood."""

    def __init__(self, env: CrafterEnv) -> None:
        self._env = env
        world = _WorldHelper(env)
        self._collector = _WoodCollectionBehavior(env, target_wood=1, world_helper=world)

    @property
    def instruction(self) -> str:
        return "collect wood"

    def reset(self) -> None:
        self._collector.reset()

    def act(self, obs: np.ndarray, info: dict) -> int:
        if self.succeeded(info):
            return NOOP_ACTION
        return self._collector.act(info)

    def succeeded(self, info: dict) -> bool:
        return info.get("inventory", {}).get("wood", 0) >= 1


class PlaceTablePolicy:
    """Collects 2 wood, then places a crafting table on walkable ground."""

    def __init__(self, env: CrafterEnv) -> None:
        self._env = env
        world = _WorldHelper(env)
        self._collector = _WoodCollectionBehavior(env, target_wood=2, world_helper=world)
        self._placer = _TablePlacementBehavior(env, world_helper=world)
        self._phase = "collect"

    @property
    def instruction(self) -> str:
        return "place table"

    def reset(self) -> None:
        self._phase = "collect"
        self._collector.reset()
        self._placer.reset()

    def act(self, obs: np.ndarray, info: dict) -> int:
        if self.succeeded(info):
            return NOOP_ACTION
        if self._phase == "collect":
            if self._collector.needs_more(info):
                return self._collector.act(info)
            self._phase = "place"
            self._placer.reset()
        if info.get("inventory", {}).get("wood", 0) < 2:
            self._phase = "collect"
            self._collector.reset()
            self._placer.reset()
            return self._collector.act(info)
        return self._placer.act(info)

    def succeeded(self, info: dict) -> bool:
        return info.get("achievements", {}).get("place_table", 0) >= 1


class CollectStonePolicy:
    """Places a table, crafts a pickaxe, then mines stone."""

    def __init__(self, env: CrafterEnv) -> None:
        self._env = env
        world = _WorldHelper(env)
        self._world = world
        self._collector = _WoodCollectionBehavior(env, target_wood=3, world_helper=world)
        self._placer = _TablePlacementBehavior(env, world_helper=world)
        self._stone_navigator = GreedyNavigator()
        self._phase = "collect_wood"
        self._target_stone: Position | None = None
        self._stone_exclusions: set[Position] = set()

    @property
    def instruction(self) -> str:
        return "collect stone"

    def reset(self) -> None:
        self._phase = "collect_wood"
        self._collector.reset()
        self._placer.reset()
        self._stone_navigator.reset()
        self._target_stone = None
        self._stone_exclusions.clear()

    def act(self, obs: np.ndarray, info: dict) -> int:
        if self.succeeded(info):
            return NOOP_ACTION
        self._advance_phase(info)
        if self._phase == "collect_wood":
            return self._collector.act(info)
        if self._phase == "place_table":
            if info.get("inventory", {}).get("wood", 0) < 2:
                self._phase = "collect_wood"
                self._collector.reset()
                self._placer.reset()
                return self._collector.act(info)
            return self._placer.act(info)
        if self._phase == "craft_pickaxe":
            if info.get("inventory", {}).get("wood", 0) <= 0:
                self._phase = "collect_wood"
                self._collector.reset()
                return self._collector.act(info)
            return MAKE_WOOD_PICKAXE_ACTION
        return self._mine_stone_action(info)

    def succeeded(self, info: dict) -> bool:
        return info.get("inventory", {}).get("stone", 0) >= 1

    def _advance_phase(self, info: dict) -> None:
        if self._phase == "collect_wood" and not self._collector.needs_more(info):
            self._phase = "place_table"
            self._placer.reset()
        if self._phase == "place_table" and info.get("achievements", {}).get("place_table", 0) >= 1:
            self._phase = "craft_pickaxe"
        if self._phase == "craft_pickaxe" and info.get("inventory", {}).get("wood_pickaxe", 0) >= 1:
            self._phase = "mine_stone"
            self._stone_navigator.reset()
            self._target_stone = None
            self._stone_exclusions.clear()

    def _mine_stone_action(self, info: dict) -> int:
        if info.get("inventory", {}).get("wood_pickaxe", 0) < 1:
            self._phase = "craft_pickaxe"
            return MAKE_WOOD_PICKAXE_ACTION
        player_pos = _player_pos_from_info(info, self._env)
        if self._stone_navigator.should_retarget() and self._target_stone is not None:
            self._stone_exclusions.add(self._target_stone)
            self._target_stone = None
        if self._target_stone is None or not self._world.material_matches(
            self._target_stone, "stone"
        ):
            self._target_stone = self._world.nearest_material(
                "stone", player_pos, exclude=self._stone_exclusions
            )
            self._stone_navigator.reset()
        if self._target_stone is None:
            return NOOP_ACTION
        if not _is_adjacent(player_pos, self._target_stone):
            return self._stone_navigator.next_action(player_pos, self._target_stone)
        direction = _direction_toward(player_pos, self._target_stone)
        facing = _player_facing_from_info(info, self._env)
        if facing == direction:
            return DO_ACTION
        return MOVE_ACTIONS[direction]


def _player_pos_from_info(info: dict, env: CrafterEnv) -> Position:
    pos = info.get("player_pos")
    if pos is not None:
        return _as_int_tuple(pos)
    return _as_int_tuple(env._env._player.pos)


def _player_facing_from_info(info: dict, env: CrafterEnv) -> Direction:
    facing = info.get("player_facing")
    if facing is not None:
        return _as_int_tuple(facing)
    return _as_int_tuple(env._env._player.facing)


def _as_int_tuple(values: Sequence[int] | np.ndarray | tuple[int, int]) -> Position:
    return int(values[0]), int(values[1])


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _is_adjacent(a: Position, b: Position) -> bool:
    return _manhattan(a, b) == 1


def _direction_toward(src: Position, dst: Position) -> Direction:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx > 0:
        return (1, 0)
    if dx < 0:
        return (-1, 0)
    if dy > 0:
        return (0, 1)
    if dy < 0:
        return (0, -1)
    return (0, 1)
