"""Task entry alias for G1 box placement.

Requested deliverable path: ``unilab/tasks/g1_box_placement.py``.
Implementation owner: ``unilab.envs.manipulation.g1.box_placement``.
"""

from unilab.envs.manipulation.g1.box_placement import (
    GOAL_DIM,
    GOAL_END_IDX,
    GOAL_START_IDX,
    STATE_DIM,
    G1BoxPlacementCfg,
    G1BoxPlacementEnv,
    G1BoxPlacementEnvCfg,
    G1BoxPlacementEnvRegistered,
)

__all__ = [
    "GOAL_DIM",
    "GOAL_END_IDX",
    "GOAL_START_IDX",
    "STATE_DIM",
    "G1BoxPlacementCfg",
    "G1BoxPlacementEnv",
    "G1BoxPlacementEnvCfg",
    "G1BoxPlacementEnvRegistered",
]
