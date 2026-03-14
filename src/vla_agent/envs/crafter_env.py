"""Gymnasium-style wrapper for the Crafter environment with a reduced action space."""

import numpy as np

import crafter
from crafter import objects as crafter_objects


# Mapping from reduced action index to full Crafter action index.
_ACTION_MAP: dict[int, int] = {
    0: 0,  # noop
    1: 1,  # move_left
    2: 2,  # move_right
    3: 3,  # move_up
    4: 4,  # move_down
    5: 5,  # do
    6: 8,  # place_table
    7: 11,  # make_wood_pickaxe
}

_ACTION_NAMES: list[str] = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "place_table",
    "make_wood_pickaxe",
]

_PLACE_TABLE_INDEX = _ACTION_NAMES.index("place_table")


class CrafterEnv:
    """Crafter environment with an 8-action reduced action space.

    Wraps the native crafter.Env to provide a Gymnasium-style interface:
    reset() -> (obs, info), step(action) -> (obs, reward, terminated, truncated, info).
    """

    action_names: list[str] = _ACTION_NAMES
    num_actions: int = 8
    action_map: dict[int, int] = _ACTION_MAP

    def __init__(self, seed: int = 0, image_size: tuple[int, int] = (64, 64)) -> None:
        self._seed = seed
        self._image_size = image_size
        self._env = crafter.Env(seed=seed)
        self._register_table_semantic_class()

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset the environment and return (obs, info)."""
        obs = self._env.reset()
        obs = self._maybe_resize(obs)
        info = self._extract_info_from_env()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step with a reduced-space action index (0–7)."""
        if action < 0 or action >= self.num_actions:
            raise ValueError(
                f"Action {action} is out of range. Valid range: 0 to {self.num_actions - 1}."
            )
        table_target: tuple[int, int] | None = None
        if action == _PLACE_TABLE_INDEX:
            player = self._env._player
            player_pos = self._to_int_tuple(player.pos)
            facing = self._to_int_tuple(player.facing)
            table_target = (player_pos[0] + facing[0], player_pos[1] + facing[1])
        full_action = self.action_map[action]
        obs, reward, done, info = self._env.step(full_action)
        obs = self._maybe_resize(obs)
        self._maybe_add_table_object(table_target)
        enriched_info = self._extract_info_from_env(info)
        return obs, float(reward), bool(done), False, enriched_info

    def close(self) -> None:
        """Clean up the environment."""
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_resize(self, obs: np.ndarray) -> np.ndarray:
        """Resize observation to self._image_size if needed."""
        h, w = self._image_size
        if obs.shape[:2] == (h, w):
            return obs
        from PIL import Image

        img = Image.fromarray(obs).resize((w, h), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)

    def _register_table_semantic_class(self) -> None:
        """Ensure SemanticView knows how to render our synthetic table objects."""
        sem_view = getattr(self._env, "_sem_view", None)
        if sem_view is None:
            return
        obj_ids = getattr(sem_view, "_obj_ids", None)
        if obj_ids is None or _TableObject in obj_ids:
            return
        base_index = len(self._env._world._mat_ids)
        obj_ids[_TableObject] = base_index + len(obj_ids)

    def _maybe_add_table_object(self, target: tuple[int, int] | None) -> None:
        """Ensure placed tables also have a backing object for downstream logic."""
        if target is None or not self._is_within_world(target):
            return
        self._register_table_semantic_class()
        world = self._env._world
        material, obj = world[target]
        if material != "table" or obj is not None:
            return
        table_obj = _TableObject(world, target)
        world.add(table_obj)

    def _is_within_world(self, pos: tuple[int, int]) -> bool:
        area = self._env._world.area
        return 0 <= pos[0] < int(area[0]) and 0 <= pos[1] < int(area[1])

    def _extract_info_from_env(self, base_info: dict | None = None) -> dict:
        """Build info dict from the underlying Crafter env player state."""
        info = dict(base_info or {})
        player = self._env._player
        info["inventory"] = dict(player.inventory)
        info["achievements"] = dict(player.achievements)
        info["player_pos"] = self._to_int_tuple(player.pos)
        info["player_facing"] = self._to_int_tuple(player.facing)
        return info

    @staticmethod
    def _to_int_tuple(values: tuple[int, int] | np.ndarray) -> tuple[int, int]:
        """Convert a 2-sequence of coordinates to a tuple of ints."""
        return int(values[0]), int(values[1])


class _TableObject(crafter_objects.Object):
    """Simple inert object used to back placed tables."""

    @property
    def texture(self) -> str:
        return "table"

    def update(self) -> None:  # pragma: no cover - no runtime behavior needed.
        return
