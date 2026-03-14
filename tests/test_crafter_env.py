"""Tests for vla_agent.envs.crafter_env — covering all acceptance criteria from mvp-0b-spec.md."""

import numpy as np
import pytest


def _assert_player_state(info: dict) -> None:
    """Helper to assert that info exposes player_pos/player_facing tuples of ints."""
    for key in ("player_pos", "player_facing"):
        assert key in info, f"info from CrafterEnv must include '{key}'"
        value = info[key]
        assert isinstance(value, tuple), f"'{key}' must be a tuple, got {type(value)}"
        assert len(value) == 2, f"'{key}' must contain two coordinates"
        assert all(isinstance(coord, int) for coord in value), (
            f"Coordinates in '{key}' must be ints, got {value}"
        )


def _find_valid_place_table_direction(player, world) -> tuple[int, int]:
    """Return a cardinal direction that faces a valid table placement cell."""
    pos = np.array(player.pos, dtype=int)
    area = tuple(int(dim) for dim in world.area)
    for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        target = pos + np.array([dx, dy], dtype=int)
        if not (0 <= target[0] < area[0] and 0 <= target[1] < area[1]):
            continue
        target_tuple = (int(target[0]), int(target[1]))
        mat_id = int(world._mat_map[target_tuple])
        material = world._mat_names[mat_id]
        if material not in ("grass", "path", "sand"):
            continue
        if world._obj_map[target_tuple] != 0:
            continue
        return dx, dy
    pytest.fail("Unable to find an adjacent cell suitable for table placement.")


# ---------------------------------------------------------------------------
# AC-1: Package import works
# ---------------------------------------------------------------------------


def test_import_crafter_env():
    """AC-1: 'from vla_agent.envs.crafter_env import CrafterEnv' must work."""
    from vla_agent.envs.crafter_env import CrafterEnv  # noqa: F401


# ---------------------------------------------------------------------------
# AC-2: CrafterEnv has correct interface (unit-testable parts — no Crafter)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCrafterEnvInterface:
    """AC-2: Static interface properties — num_actions and action_names."""

    def test_num_actions(self):
        """env.num_actions must equal 8."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            assert env.num_actions == 8
        finally:
            env.close()

    def test_action_names_type_and_length(self):
        """env.action_names must be a list of 8 strings."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            assert isinstance(env.action_names, list)
            assert len(env.action_names) == 8
            assert all(isinstance(n, str) for n in env.action_names)
        finally:
            env.close()

    def test_action_names_exact_order(self):
        """env.action_names must match the exact 8-element list from the spec."""
        from vla_agent.envs.crafter_env import CrafterEnv

        expected = [
            "noop",
            "move_left",
            "move_right",
            "move_up",
            "move_down",
            "do",
            "place_table",
            "make_wood_pickaxe",
        ]
        env = CrafterEnv(seed=0)
        try:
            assert env.action_names == expected
        finally:
            env.close()

    def test_step_out_of_range_high_raises_value_error(self):
        """step(8) must raise ValueError (one above valid range 0-7)."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            with pytest.raises(ValueError):
                env.step(8)
        finally:
            env.close()

    def test_step_out_of_range_negative_raises_value_error(self):
        """step(-1) must raise ValueError."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            with pytest.raises(ValueError):
                env.step(-1)
        finally:
            env.close()

    def test_step_out_of_range_large_raises_value_error(self):
        """step(100) must raise ValueError."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            with pytest.raises(ValueError):
                env.step(100)
        finally:
            env.close()


# ---------------------------------------------------------------------------
# AC-3: Action mapping (unit-testable without running Crafter steps)
# ---------------------------------------------------------------------------

FULL_ACTION_MAP = {
    0: 0,  # noop  → 0
    1: 1,  # move_left → 1
    2: 2,  # move_right → 2
    3: 3,  # move_up → 3
    4: 4,  # move_down → 4
    5: 5,  # do → 5
    6: 8,  # place_table → 8
    7: 11,  # make_wood_pickaxe → 11
}


@pytest.mark.integration
class TestActionMapping:
    """AC-3: Reduced-action-to-full-action index mapping."""

    def test_action_map_attribute_exists(self):
        """CrafterEnv must expose the action mapping (as an attribute or via inspection)."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            # The mapping should be accessible. The spec doesn't name the attribute,
            # but the class must have a way to inspect the full_action_index for each
            # reduced index. We test this via a public attribute 'action_map' or
            # '_action_map'. If neither exists, the integration tests will catch it.
            has_map = hasattr(env, "action_map") or hasattr(env, "_action_map")
            assert has_map, (
                "CrafterEnv must have an 'action_map' or '_action_map' attribute "
                "mapping reduced indices to full Crafter indices"
            )
        finally:
            env.close()

    def test_action_map_place_table_maps_to_8(self):
        """Reduced action 6 (place_table) must map to full Crafter action 8."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            mapping = getattr(env, "action_map", None) or getattr(env, "_action_map")
            assert mapping[6] == 8, (
                f"place_table (reduced=6) must map to full index 8, got {mapping[6]}"
            )
        finally:
            env.close()

    def test_action_map_do_maps_to_5(self):
        """Reduced action 5 (do) must map to full Crafter action 5."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            mapping = getattr(env, "action_map", None) or getattr(env, "_action_map")
            assert mapping[5] == 5, f"do (reduced=5) must map to full index 5, got {mapping[5]}"
        finally:
            env.close()

    def test_action_map_noop_maps_to_0(self):
        """Reduced action 0 (noop) must map to full Crafter action 0."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            mapping = getattr(env, "action_map", None) or getattr(env, "_action_map")
            assert mapping[0] == 0, f"noop (reduced=0) must map to full index 0, got {mapping[0]}"
        finally:
            env.close()

    def test_action_map_make_wood_pickaxe_maps_to_11(self):
        """Reduced action 7 must map to full Crafter action 11 (make_wood_pickaxe)."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            mapping = getattr(env, "action_map", None) or getattr(env, "_action_map")
            assert mapping[7] == 11, (
                f"make_wood_pickaxe (reduced=7) must map to full index 11, got {mapping[7]}"
            )
        finally:
            env.close()

    def test_action_map_all_entries(self):
        """All 8 reduced actions must map to the correct full Crafter indices."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            mapping = getattr(env, "action_map", None) or getattr(env, "_action_map")
            for reduced, full in FULL_ACTION_MAP.items():
                assert mapping[reduced] == full, (
                    f"Reduced action {reduced} should map to {full}, got {mapping[reduced]}"
                )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# AC-2 + AC-4: Integration — full lifecycle, obs shape, dtype, info keys
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCrafterEnvIntegration:
    """Integration tests that require a working Crafter installation."""

    def test_reset_returns_obs_and_info(self):
        """reset() must return a (obs, info) 2-tuple."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            result = env.reset()
            assert isinstance(result, tuple) and len(result) == 2, (
                "reset() must return a 2-tuple (obs, info)"
            )
        finally:
            env.close()

    def test_reset_obs_shape(self):
        """reset() observation must have shape (64, 64, 3) by default."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            obs, _ = env.reset()
            assert obs.shape == (64, 64, 3), f"Expected obs shape (64, 64, 3), got {obs.shape}"
        finally:
            env.close()

    def test_reset_obs_dtype(self):
        """reset() observation must be np.uint8."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            obs, _ = env.reset()
            assert obs.dtype == np.uint8, f"Expected obs dtype uint8, got {obs.dtype}"
        finally:
            env.close()

    def test_reset_info_has_inventory(self):
        """reset() info dict must contain 'inventory' key."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            _, info = env.reset()
            assert "inventory" in info, "info from reset() must contain 'inventory'"
        finally:
            env.close()

    def test_reset_info_has_achievements(self):
        """reset() info dict must contain 'achievements' key."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            _, info = env.reset()
            assert "achievements" in info, "info from reset() must contain 'achievements'"
        finally:
            env.close()

    def test_reset_info_has_player_state(self):
        """reset() info dict must expose player position and facing."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            _, info = env.reset()
            _assert_player_state(info)
        finally:
            env.close()

    def test_step_returns_5_tuple(self):
        """step(action) must return a 5-tuple (obs, reward, terminated, truncated, info)."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            result = env.step(0)
            assert isinstance(result, tuple) and len(result) == 5, "step() must return a 5-tuple"
        finally:
            env.close()

    def test_step_obs_shape(self):
        """step() observation must have shape (64, 64, 3)."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            obs, _, _, _, _ = env.step(0)
            assert obs.shape == (64, 64, 3), f"Expected step obs shape (64, 64, 3), got {obs.shape}"
        finally:
            env.close()

    def test_step_obs_dtype(self):
        """step() observation must be np.uint8."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            obs, _, _, _, _ = env.step(0)
            assert obs.dtype == np.uint8, f"Expected step obs dtype uint8, got {obs.dtype}"
        finally:
            env.close()

    def test_step_reward_is_float(self):
        """step() reward must be a float."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            _, reward, _, _, _ = env.step(0)
            assert isinstance(reward, float), f"Expected reward to be float, got {type(reward)}"
        finally:
            env.close()

    def test_step_terminated_is_bool(self):
        """step() terminated must be a bool."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            _, _, terminated, _, _ = env.step(0)
            assert isinstance(terminated, bool), (
                f"Expected terminated to be bool, got {type(terminated)}"
            )
        finally:
            env.close()

    def test_step_truncated_is_bool(self):
        """step() truncated must be a bool."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            _, _, _, truncated, _ = env.step(0)
            assert isinstance(truncated, bool), (
                f"Expected truncated to be bool, got {type(truncated)}"
            )
        finally:
            env.close()

    def test_step_truncated_always_false(self):
        """Crafter does not distinguish terminated vs truncated; truncated must always be False."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            for _ in range(5):
                _, _, terminated, truncated, _ = env.step(0)
                assert truncated is False, "truncated must always be False"
                if terminated:
                    break
        finally:
            env.close()

    def test_step_info_has_inventory(self):
        """step() info dict must contain 'inventory' key."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            _, _, _, _, info = env.step(0)
            assert "inventory" in info, "info from step() must contain 'inventory'"
        finally:
            env.close()

    def test_step_info_has_achievements(self):
        """step() info dict must contain 'achievements' key."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            _, _, _, _, info = env.step(0)
            assert "achievements" in info, "info from step() must contain 'achievements'"
        finally:
            env.close()

    def test_step_info_has_player_state(self):
        """step() info dict must expose player position and facing."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            _, _, _, _, info = env.step(0)
            _assert_player_state(info)
        finally:
            env.close()

    def test_make_wood_pickaxe_requires_table_and_wood(self):
        """
        make_wood_pickaxe must succeed once a table is placed nearby and wood is available.
        """
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            player = env._env._player
            world = env._env._world
            dx, dy = _find_valid_place_table_direction(player, world)
            player.inventory["wood"] = 2
            player.facing = (dx, dy)
            _, _, _, _, info = env.step(6)
            assert info["achievements"].get("place_table", 0) >= 1
            player_pos = tuple(int(coord) for coord in player.pos)
            target_pos = (player_pos[0] + dx, player_pos[1] + dy)
            material, obj = world[target_pos]
            assert material == "table"
            assert obj is not None
            player.inventory["wood"] = 1
            player.facing = (dx, dy)
            _, _, _, _, info = env.step(7)
            assert info["inventory"].get("wood_pickaxe", 0) >= 1
        finally:
            env.close()

    def test_step_all_valid_reduced_actions(self):
        """All 8 reduced actions (0-7) must be accepted without error."""
        from vla_agent.envs.crafter_env import CrafterEnv

        for action in range(8):
            env = CrafterEnv(seed=0)
            try:
                env.reset()
                env.step(action)  # must not raise
            finally:
                env.close()

    def test_close_does_not_raise(self):
        """close() must not raise an exception."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        env.reset()
        env.close()  # must not raise

    def test_close_without_reset_does_not_raise(self):
        """close() called before reset() must not raise."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        env.close()  # must not raise


# ---------------------------------------------------------------------------
# AC-4: Episode lifecycle — run to completion, reset after done
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEpisodeLifecycle:
    """AC-4: Episode lifecycle — run to completion or step limit, then reset."""

    def test_episode_runs_until_terminated_or_limit(self):
        """An episode runs until terminated==True or the step limit (200) is reached."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0)
        try:
            env.reset()
            max_steps = 200
            terminated = False
            steps = 0
            while not terminated and steps < max_steps:
                _, _, terminated, _, _ = env.step(0)  # noop policy
                steps += 1
            assert steps > 0, "At least one step must have occurred"
        finally:
            env.close()

    def test_reset_after_terminated_starts_new_episode(self):
        """reset() after a terminated episode returns a fresh observation."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=1)
        try:
            obs1, _ = env.reset()

            # Run until terminated or limit
            terminated = False
            steps = 0
            last_obs = obs1
            while not terminated and steps < 300:
                last_obs, _, terminated, _, _ = env.step(0)
                steps += 1

            # Now reset and get a fresh observation
            obs2, info2 = env.reset()
            assert obs2.shape == (64, 64, 3)
            assert obs2.dtype == np.uint8
            assert "inventory" in info2
            assert "achievements" in info2
        finally:
            env.close()


# ---------------------------------------------------------------------------
# AC-5: Seed reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSeedReproducibility:
    """AC-5: Two envs with the same seed and same actions must produce identical observations."""

    def test_same_seed_same_reset_obs(self):
        """Two envs with the same seed produce byte-equal observations after reset()."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env1 = CrafterEnv(seed=42)
        env2 = CrafterEnv(seed=42)
        try:
            obs1, _ = env1.reset()
            obs2, _ = env2.reset()
            np.testing.assert_array_equal(
                obs1, obs2, err_msg=("reset() with same seed must produce byte-equal observations")
            )
        finally:
            env1.close()
            env2.close()

    def test_same_seed_same_step_sequence(self):
        """Two envs with the same seed and same actions produce byte-equal observations for 10 steps."""
        from vla_agent.envs.crafter_env import CrafterEnv

        actions = [3, 2, 5, 0, 1, 4, 6, 0, 3, 2]  # 10 deterministic actions
        env1 = CrafterEnv(seed=42)
        env2 = CrafterEnv(seed=42)
        try:
            env1.reset()
            env2.reset()
            for i, action in enumerate(actions):
                obs1, _, done1, _, _ = env1.step(action)
                obs2, _, done2, _, _ = env2.step(action)
                np.testing.assert_array_equal(
                    obs1,
                    obs2,
                    err_msg=(f"Step {i}: same seed + same action must yield byte-equal obs"),
                )
                if done1 or done2:
                    break
        finally:
            env1.close()
            env2.close()

    def test_different_seeds_produce_different_obs(self):
        """Two envs with different seeds produce different observations after reset()."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env1 = CrafterEnv(seed=0)
        env2 = CrafterEnv(seed=1)
        try:
            obs1, _ = env1.reset()
            obs2, _ = env2.reset()
            # With overwhelming probability two different seeds produce different frames
            assert not np.array_equal(obs1, obs2), (
                "Different seeds should produce different initial observations"
            )
        finally:
            env1.close()
            env2.close()


# ---------------------------------------------------------------------------
# AC-2: Custom image_size — resizing
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCustomImageSize:
    """AC-2 (extended): image_size parameter resizes the observation correctly."""

    def test_custom_image_size_reset(self):
        """reset() returns obs with the custom image_size shape."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0, image_size=(32, 32))
        try:
            obs, _ = env.reset()
            assert obs.shape == (32, 32, 3), f"Expected (32, 32, 3), got {obs.shape}"
            assert obs.dtype == np.uint8
        finally:
            env.close()

    def test_custom_image_size_step(self):
        """step() returns obs with the custom image_size shape."""
        from vla_agent.envs.crafter_env import CrafterEnv

        env = CrafterEnv(seed=0, image_size=(32, 32))
        try:
            env.reset()
            obs, _, _, _, _ = env.step(0)
            assert obs.shape == (32, 32, 3), f"Expected (32, 32, 3), got {obs.shape}"
            assert obs.dtype == np.uint8
        finally:
            env.close()
