"""G1 box placement task for MotrixSim / MuJoCo backends.

Corresponds to the requested ``unilab/tasks/g1_box_placement.py`` deliverable;
UniLab keeps task logic under ``envs/manipulation/`` per registry contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationCapabilities, DomainRandomizationProvider, ResetPlan
from unilab.dr.dr_utils import validate_common_reset_randomization, zero_actions
from unilab.envs.common.rotation import np_matrix_from_quat
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.envs.manipulation.g1.success import BoxPlacementSuccessCriteria

# Goal vector layout (18D, aligned with Scaling-CRL goal encoder input):
#   box pos(3) + box euler(3) + left EE pos(3) + euler(3) + right EE pos(3) + euler(3)
GOAL_DIM = 18
STATE_DIM = 93  # gyro(3)+gravity(3)+joint_diff(29)+joint_vel(29)+action(29)
GOAL_START_IDX = STATE_DIM
GOAL_END_IDX = STATE_DIM + GOAL_DIM


def _quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Convert w-first quaternions (N, 4) to roll-pitch-yaw."""
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=1).astype(quat.dtype)


def _pose6_from_pos_quat(pos_w: np.ndarray, quat_w: np.ndarray) -> np.ndarray:
    return np.concatenate([pos_w, _quat_to_euler_xyz(quat_w)], axis=1)


def _tilt_deg_from_quat(quat_w: np.ndarray) -> np.ndarray:
    rot = np_matrix_from_quat(quat_w)
    z_axis = rot[:, :, 2]
    cos_tilt = np.clip(z_axis[:, 2], -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt)).astype(quat_w.dtype)


@dataclass
class BoxPlacementRewardConfig:
    """Sparse CRL-friendly reward with light shaping."""

    success_reward: float = 100.0
    distance_penalty_scale: float = -0.1
    grasp_slip_penalty_scale: float = -0.5
    grasp_slip_threshold_m: float = 0.25
    posture_penalty_scale: float = -2.0
    max_tilt_deg: float = 45.0
    min_base_height: float = 0.45


@dataclass
class G1BoxPlacementCfg(G1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_box_placement.xml")
        )
    )
    box_body_name: str = "cargo_box"
    platform_top_geom_name: str = "platform_top"
    left_ee_body_name: str = "left_wrist_yaw_link"
    right_ee_body_name: str = "right_wrist_yaw_link"
    box_init_xy: tuple[float, float] = (0.8, 0.0)
    box_init_z: float = 0.15
    platform_center_xy: tuple[float, float] = (1.5, 0.0)
    platform_top_z: float = 0.6
    success_criteria: BoxPlacementSuccessCriteria = field(
        default_factory=BoxPlacementSuccessCriteria
    )
    reward_config: BoxPlacementRewardConfig = field(default_factory=BoxPlacementRewardConfig)
    max_episode_steps: int = 1000


@registry.envcfg("G1BoxPlacement")
@dataclass
class G1BoxPlacementEnvCfg(G1BoxPlacementCfg):
    """Registered Hydra/env config for G1 box placement."""

    pass


class BoxPlacementDRProvider(DomainRandomizationProvider):
    """Reset provider: stand keyframe + randomized box spawn on the floor."""

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        validate_common_reset_randomization(env, capabilities)

    def build_reset_plan(self, env: G1BoxPlacementEnv, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.broadcast_to(env._init_qpos, (num_reset, env._init_qpos.shape[0])).copy()
        qvel = np.zeros((num_reset, env._init_qvel.shape[0]), dtype=np.float32)
        jitter_xy = np.random.uniform(-0.05, 0.05, size=(num_reset, 2)).astype(np.float32)
        box_xy = np.asarray(env._cfg.box_init_xy, dtype=np.float32) + jitter_xy
        qpos[:, env._obj_pos_slice] = np.stack(
            [box_xy[:, 0], box_xy[:, 1], np.full(num_reset, env._cfg.box_init_z, dtype=np.float32)],
            axis=1,
        )
        qpos[:, env._obj_quat_slice] = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_reset, 1)
        )
        env._stable_counter[env_ids] = 0
        env._episode_seed[env_ids] = env._next_episode_seed
        env._next_episode_seed += float(num_reset)
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates={"current_actions": zero_actions(num_reset, env._num_action)},
        )

    def build_reset_observation(
        self, env: G1BoxPlacementEnv, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        gravity = env._backend.get_sensor_data(env._cfg.sensor.upvector)[env_ids]
        dof_pos = env.get_dof_pos()[env_ids, : env._num_action]
        dof_vel = env.get_dof_vel()[env_ids, : env._num_action]
        state_obs = env._compute_state_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)
        rollout_goal = np.broadcast_to(env._target_goal, (len(env_ids), GOAL_DIM)).copy()
        return env._compute_obs_dict(state_obs, rollout_goal, env._episode_seed[env_ids])


class G1BoxPlacementEnv(G1BaseEnv):
    """G1 pick-place task with CRL goal space and MotrixSim scene contract."""

    _cfg: G1BoxPlacementCfg
    _keyframe_name = "stand"
    _use_global_dtype = False

    def __init__(self, cfg: G1BoxPlacementCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            add_body_sensors=True,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._reward_cfg = cfg.reward_config
        self._success_cfg = cfg.success_criteria
        self._box_body_ids = self._backend.get_body_ids([self._cfg.box_body_name])
        self._left_ee_ids = self._backend.get_body_ids([self._cfg.left_ee_body_name])
        self._right_ee_ids = self._backend.get_body_ids([self._cfg.right_ee_body_name])
        self._platform_xy = np.asarray(self._cfg.platform_center_xy, dtype=np.float32)
        self._platform_top_z = float(self._cfg.platform_top_z)
        self._target_goal = self._build_target_goal_vector()
        self._stable_counter = np.zeros((num_envs,), dtype=np.int32)
        self._episode_seed = np.zeros((num_envs,), dtype=np.float32)
        self._next_episode_seed = 1.0
        nq = self._init_qpos.shape[0]
        self._obj_pos_slice = slice(nq - 7, nq - 4)
        self._obj_quat_slice = slice(nq - 4, nq)
        self._init_domain_randomization(BoxPlacementDRProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # obs = state || goal for CRL actor; critic stores goal + episode seed for relabeling.
        return {"obs": STATE_DIM + GOAL_DIM, "critic": GOAL_DIM + 1}

    @property
    def goal_start_idx(self) -> int:
        return GOAL_START_IDX

    @property
    def goal_end_idx(self) -> int:
        return GOAL_END_IDX

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def goal_dim(self) -> int:
        return GOAL_DIM

    def _build_target_goal_vector(self) -> np.ndarray:
        """Fixed placement goal: box centered on platform with upright orientation."""
        box_pos = np.array(
            [self._cfg.platform_center_xy[0], self._cfg.platform_center_xy[1], self._platform_top_z + 0.15],
            dtype=np.float32,
        )
        box_euler = np.zeros(3, dtype=np.float32)
        left_pos = box_pos + np.array([0.0, 0.12, 0.05], dtype=np.float32)
        right_pos = box_pos + np.array([0.0, -0.12, 0.05], dtype=np.float32)
        ee_euler = np.zeros(3, dtype=np.float32)
        return np.concatenate([box_pos, box_euler, left_pos, ee_euler, right_pos, ee_euler]).astype(
            np.float32
        )

    def _read_goal_features(self) -> np.ndarray:
        box_pos = self._backend.get_body_pos_w(self._box_body_ids)[:, 0, :]
        box_quat = self._backend.get_body_quat_w(self._box_body_ids)[:, 0, :]
        left_pos = self._backend.get_body_pos_w(self._left_ee_ids)[:, 0, :]
        left_quat = self._backend.get_body_quat_w(self._left_ee_ids)[:, 0, :]
        right_pos = self._backend.get_body_pos_w(self._right_ee_ids)[:, 0, :]
        right_quat = self._backend.get_body_quat_w(self._right_ee_ids)[:, 0, :]
        return np.concatenate(
            [
                _pose6_from_pos_quat(box_pos, box_quat),
                _pose6_from_pos_quat(left_pos, left_quat),
                _pose6_from_pos_quat(right_pos, right_quat),
            ],
            axis=1,
        ).astype(np.float32)

    def _compute_state_obs(
        self, info: dict[str, Any], linvel, gyro, gravity, dof_pos, dof_vel
    ) -> np.ndarray:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        last_actions = info.get("current_actions", np.zeros_like(diff))
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        return np.concatenate(
            [
                noisy_gyro * 0.25,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel * 0.05,
                last_actions,
            ],
            axis=1,
            dtype=np.float32,
        )

    def _compute_obs_dict(
        self, state_obs: np.ndarray, goal: np.ndarray, episode_seed: np.ndarray
    ) -> dict[str, np.ndarray]:
        actor_obs = np.concatenate([state_obs, goal], axis=1)
        critic = np.concatenate(
            [goal, episode_seed[:, None].astype(np.float32)], axis=1
        )
        return {"obs": actor_obs.astype(np.float32), "critic": critic.astype(np.float32)}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()[:, : self._num_action]
        dof_vel = self.get_dof_vel()[:, : self._num_action]

        state_obs = self._compute_state_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        achieved_goal = self._read_goal_features()
        # CRL rollout goal: target placement configuration (relabeling happens in learner).
        rollout_goal = np.broadcast_to(self._target_goal, (self._num_envs, GOAL_DIM)).copy()
        obs = self._compute_obs_dict(state_obs, rollout_goal, self._episode_seed)

        reward, terminated, success_flags = self._compute_reward_and_done(
            state.info, gravity, achieved_goal
        )
        state = state.replace(obs=obs, reward=reward, terminated=terminated)
        if "log" not in state.info:
            state.info["log"] = {}
        state.info["log"]["success_rate"] = float(np.mean(success_flags))
        if np.any(success_flags):
            state.info["log"]["episode_success"] = float(np.mean(success_flags))
            state.info["log"]["success_episode_length"] = float(
                np.mean(state.info["steps"][success_flags] + 1)
            )
        state.info["achieved_goal"] = achieved_goal
        state.info["goal_start_idx"] = GOAL_START_IDX
        state.info["goal_end_idx"] = GOAL_END_IDX
        state.info["state_dim"] = STATE_DIM
        return state

    def _compute_reward_and_done(
        self,
        info: dict[str, Any],
        gravity: np.ndarray,
        achieved_goal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self._reward_cfg
        dtype = np.float32

        box_pos = achieved_goal[:, 0:3]
        box_bottom_z = box_pos[:, 2] - 0.15
        horizontal_error = np.linalg.norm(box_pos[:, :2] - self._platform_xy[None, :], axis=1)
        height_error = np.abs(box_bottom_z - self._platform_top_z)

        box_quat = self._backend.get_body_quat_w(self._box_body_ids)[:, 0, :]
        tilt_deg = _tilt_deg_from_quat(box_quat)

        frame_ok = self._success_cfg.frame_satisfied(
            horizontal_error=horizontal_error,
            height_error=height_error,
            tilt_deg=tilt_deg,
        )
        self._stable_counter, succeeded = self._success_cfg.update_stable_counter(
            self._stable_counter, frame_ok
        )

        reward = np.zeros((self._num_envs,), dtype=dtype)
        reward += cfg.distance_penalty_scale * horizontal_error

        left_pos = self._backend.get_body_pos_w(self._left_ee_ids)[:, 0, :]
        right_pos = self._backend.get_body_pos_w(self._right_ee_ids)[:, 0, :]
        grasp_dist = np.minimum(
            np.linalg.norm(left_pos - box_pos, axis=1),
            np.linalg.norm(right_pos - box_pos, axis=1),
        )
        slip = np.maximum(0.0, grasp_dist - cfg.grasp_slip_threshold_m)
        reward += cfg.grasp_slip_penalty_scale * slip

        tilt = np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))
        tilt_deg_robot = np.degrees(tilt)
        posture_bad = tilt_deg_robot > cfg.max_tilt_deg
        reward += cfg.posture_penalty_scale * posture_bad.astype(dtype)
        base_z = self._backend.get_base_pos()[:, 2]
        fallen = base_z < cfg.min_base_height
        terminated = np.logical_or(posture_bad, fallen)
        reward[succeeded] += cfg.success_reward
        terminated = np.logical_or(terminated, succeeded)
        return reward.astype(dtype), terminated.astype(bool), succeeded

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        if self._cfg.max_episode_steps <= 0:
            return np.zeros((self._num_envs,), dtype=bool)
        return state.info["steps"] >= int(self._cfg.max_episode_steps)


@registry.env("G1BoxPlacement", sim_backend="mujoco")
@registry.env("G1BoxPlacement", sim_backend="motrix")
class G1BoxPlacementEnvRegistered(G1BoxPlacementEnv):
    """Registry-facing env class."""

    pass
