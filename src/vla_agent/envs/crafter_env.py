"""Gymnasium-style wrapper for the Crafter environment with a reduced action space."""

import numpy as np

import crafter


# Mapping from reduced action index to full Crafter action index.
_ACTION_MAP: dict[int, int] = {
    0: 0,  # noop
    1: 1,  # move_left
    2: 2,  # move_right
    3: 3,  # move_up
    4: 4,  # move_down
    5: 5,  # do
    6: 8,  # place_table
}

_ACTION_NAMES: list[str] = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "place_table",
]


class CrafterEnv:
    """Crafter environment with a 7-action reduced action space.

    Wraps the native crafter.Env to provide a Gymnasium-style interface:
    reset() -> (obs, info), step(action) -> (obs, reward, terminated, truncated, info).
    """

    action_names: list[str] = _ACTION_NAMES
    num_actions: int = 7
    action_map: dict[int, int] = _ACTION_MAP

    def __init__(self, seed: int = 0, image_size: tuple[int, int] = (64, 64)) -> None:
        self._seed = seed
        self._image_size = image_size
        self._env = crafter.Env(seed=seed)

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset the environment and return (obs, info)."""
        obs = self._env.reset()
        obs = self._maybe_resize(obs)
        info = self._extract_info_from_env()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step with a reduced-space action index (0–6).

        Args:
            action: Reduced action index in [0, 6].

        Returns:
            (obs, reward, terminated, truncated, info)

        Raises:
            ValueError: If action is outside the valid range [0, 6].
        """
        if action < 0 or action >= self.num_actions:
            raise ValueError(
                f"Action {action} is out of range. Valid range: 0 to {self.num_actions - 1}."
            )
        full_action = self.action_map[action]
        obs, reward, done, info = self._env.step(full_action)
        obs = self._maybe_resize(obs)
        return obs, float(reward), bool(done), False, info

    def close(self) -> None:
        """Clean up the environment."""
        pass

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

    def _extract_info_from_env(self) -> dict:
        """Build info dict from the underlying Crafter env player state."""
        player = self._env._player
        return {
            "inventory": dict(player.inventory),
            "achievements": dict(player.achievements),
        }
