"""Registry and observation contract tests for G1 box placement."""

from __future__ import annotations

import numpy as np

from unilab.base.registry import ensure_registries, make
from unilab.envs.manipulation.g1.box_placement import GOAL_DIM, STATE_DIM
from unilab.envs.manipulation.g1.success import BoxPlacementSuccessCriteria


def test_g1_box_placement_registry_contract():
    ensure_registries()
    env = make("G1BoxPlacement", num_envs=2, sim_backend="mujoco")
    try:
        assert env.obs_groups_spec["obs"] == STATE_DIM + GOAL_DIM
        assert env.obs_groups_spec["critic"] == GOAL_DIM + 1
        state = env.init_state()
        assert set(state.obs) == {"obs", "critic"}
        assert state.obs["obs"].shape == (2, STATE_DIM + GOAL_DIM)
        step_state = env.step(np.zeros((2, env.action_space.shape[0]), dtype=np.float32))
        assert step_state.obs["obs"].shape == (2, STATE_DIM + GOAL_DIM)
        assert "success_rate" in step_state.info["log"]
    finally:
        env.close()


def test_box_placement_success_criteria_stable_counter():
    criteria = BoxPlacementSuccessCriteria(stable_steps=3)
    counter = np.zeros(4, dtype=np.int32)
    ok = np.array([True, True, False, True])
    counter, _ = criteria.update_stable_counter(counter, ok)
    assert counter.tolist() == [1, 1, 0, 1]
    counter, succeeded = criteria.update_stable_counter(counter, ok)
    assert counter.tolist() == [2, 2, 0, 2]
    counter, succeeded = criteria.update_stable_counter(counter, ok)
    assert succeeded.tolist() == [True, True, False, True]
