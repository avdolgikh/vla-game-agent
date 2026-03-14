"""Tests for the scripted policies defined in vla_agent.policies."""

from __future__ import annotations

from typing import Any, Callable, Type

import pytest

from vla_agent.envs.crafter_env import CrafterEnv
from vla_agent.policies import CollectStonePolicy, CollectWoodPolicy, PlaceTablePolicy

PolicySuccessCheck = Callable[[dict], bool]
PolicySpec = tuple[Type[Any], str, PolicySuccessCheck]

MAX_STEPS = 300

_POLICY_SPECS: list[PolicySpec] = [
    (
        CollectWoodPolicy,
        "collect wood",
        lambda info: info.get("inventory", {}).get("wood", 0) >= 1,
    ),
    (
        PlaceTablePolicy,
        "place table",
        lambda info: info.get("achievements", {}).get("place_table", 0) >= 1,
    ),
    (
        CollectStonePolicy,
        "collect stone",
        lambda info: info.get("inventory", {}).get("stone", 0) >= 1,
    ),
]


def _run_policy_once(
    policy_cls: Type[Any], seed: int, max_steps: int = MAX_STEPS
) -> tuple[list[int], dict, str, bool]:
    """Run a scripted policy until it succeeds or the step budget is exhausted."""
    env = CrafterEnv(seed=seed)
    try:
        policy = policy_cls(env)
        obs, info = env.reset()
        policy.reset()
        if policy.succeeded(info):
            return [], info, policy.instruction, True
        actions: list[int] = []
        for _ in range(max_steps):
            action = int(policy.act(obs, info))
            actions.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            if policy.succeeded(info) or terminated or truncated:
                break
        else:
            raise AssertionError(f"{policy_cls.__name__} did not finish within {max_steps} steps")
        policy_success = policy.succeeded(info)
        return actions, info, policy.instruction, policy_success
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("policy_cls,instruction,success_check", _POLICY_SPECS)
def test_scripted_policy_succeeds(
    policy_cls: Type[Any], instruction: str, success_check: PolicySuccessCheck
) -> None:
    actions, info, policy_instruction, success_flag = _run_policy_once(policy_cls, seed=42)
    assert policy_instruction == instruction, "Policy instruction must match the spec"
    assert success_flag, f"{policy_cls.__name__} should report success via succeeded()"
    assert success_check(info), "Final info must satisfy the policy success condition"
    assert actions, "The policy must emit at least one action"
    assert len(actions) <= MAX_STEPS
    assert all(0 <= action < CrafterEnv.num_actions for action in actions)


@pytest.mark.integration
@pytest.mark.parametrize("policy_cls", [CollectWoodPolicy, PlaceTablePolicy, CollectStonePolicy])
def test_scripted_policy_determinism(policy_cls: Type[Any]) -> None:
    actions_a, _, _, success_a = _run_policy_once(policy_cls, seed=123)
    actions_b, _, _, success_b = _run_policy_once(policy_cls, seed=123)
    assert success_a and success_b, "Both runs should succeed"
    assert actions_a == actions_b, "Same seed and policy must produce identical action sequences"
