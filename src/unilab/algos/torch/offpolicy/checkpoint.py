"""Checkpoint helpers for off-policy double-buffer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.training.run import parse_checkpoint_iteration

OFFPOLICY_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class OffPolicyResumeState:
    learner: dict[str, Any]
    iteration: int
    replay: dict[str, Any] | None
    reward_stats_ptr: int


OFFPOLICY_WEIGHTS_FORMAT = "offpolicy_weights"
OFFPOLICY_FULL_FORMAT = "offpolicy_full"


def is_structured_offpolicy_checkpoint(checkpoint: dict[str, Any]) -> bool:
    return (
        isinstance(checkpoint, dict)
        and "learner" in checkpoint
        and "training" in checkpoint
        and checkpoint.get("format")
        in (OFFPOLICY_WEIGHTS_FORMAT, OFFPOLICY_FULL_FORMAT)
    )


def is_full_offpolicy_checkpoint(checkpoint: dict[str, Any]) -> bool:
    return (
        isinstance(checkpoint, dict)
        and checkpoint.get("format") == OFFPOLICY_FULL_FORMAT
        and "learner" in checkpoint
        and "training" in checkpoint
    )


def is_weights_offpolicy_checkpoint(checkpoint: dict[str, Any]) -> bool:
    return (
        isinstance(checkpoint, dict)
        and checkpoint.get("format") == OFFPOLICY_WEIGHTS_FORMAT
        and "learner" in checkpoint
        and "training" in checkpoint
    )


def is_legacy_learner_checkpoint(checkpoint: dict[str, Any]) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    if is_structured_offpolicy_checkpoint(checkpoint):
        return False
    return "actor" in checkpoint or "qnet" in checkpoint


def extract_learner_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if is_structured_offpolicy_checkpoint(checkpoint):
        learner = checkpoint["learner"]
        if not isinstance(learner, dict):
            raise ValueError("Structured off-policy checkpoint has invalid learner payload")
        return learner
    if is_legacy_learner_checkpoint(checkpoint):
        return checkpoint
    raise ValueError("Unrecognized off-policy checkpoint format")


def serialize_replay_buffer(replay_buffer: ReplayBuffer) -> dict[str, Any]:
    return {
        "storage": replay_buffer._storage.detach().cpu().clone(),
        "ptr": int(replay_buffer.ptr[0].item()),
        "size": int(replay_buffer.size[0].item()),
        "capacity": int(replay_buffer.capacity),
        "obs_dim": int(replay_buffer._obs_dim),
        "action_dim": int(replay_buffer._action_dim),
        "critic_dim": int(replay_buffer._critic_dim),
    }


def restore_replay_buffer(replay_buffer: ReplayBuffer, replay_state: dict[str, Any]) -> None:
    expected_capacity = int(replay_buffer.capacity)
    expected_obs_dim = int(replay_buffer._obs_dim)
    expected_action_dim = int(replay_buffer._action_dim)
    expected_critic_dim = int(replay_buffer._critic_dim)

    if int(replay_state["capacity"]) != expected_capacity:
        raise ValueError(
            "Replay buffer capacity mismatch during resume: "
            f"checkpoint={replay_state['capacity']} current={expected_capacity}"
        )
    if int(replay_state["obs_dim"]) != expected_obs_dim:
        raise ValueError(
            "Replay obs_dim mismatch during resume: "
            f"checkpoint={replay_state['obs_dim']} current={expected_obs_dim}"
        )
    if int(replay_state["action_dim"]) != expected_action_dim:
        raise ValueError(
            "Replay action_dim mismatch during resume: "
            f"checkpoint={replay_state['action_dim']} current={expected_action_dim}"
        )
    if int(replay_state["critic_dim"]) != expected_critic_dim:
        raise ValueError(
            "Replay critic_dim mismatch during resume: "
            f"checkpoint={replay_state['critic_dim']} current={expected_critic_dim}"
        )

    storage = replay_state["storage"]
    if not isinstance(storage, torch.Tensor):
        storage = torch.as_tensor(storage)
    if tuple(storage.shape) != tuple(replay_buffer._storage.shape):
        raise ValueError(
            f"Replay storage shape mismatch: checkpoint={tuple(storage.shape)} "
            f"current={tuple(replay_buffer._storage.shape)}"
        )

    replay_buffer._storage.copy_(storage.to(dtype=replay_buffer._storage.dtype))
    replay_buffer.ptr[0] = int(replay_state["ptr"])
    replay_buffer.size[0] = int(replay_state["size"])


def build_offpolicy_checkpoint(
    *,
    learner_state: dict[str, Any],
    iteration: int,
    reward_stats_ptr: int = 0,
) -> dict[str, Any]:
    return {
        "format": OFFPOLICY_WEIGHTS_FORMAT,
        "version": OFFPOLICY_CHECKPOINT_VERSION,
        "learner": learner_state,
        "training": {
            "iteration": int(iteration),
            "reward_stats_ptr": int(reward_stats_ptr),
        },
    }


def parse_offpolicy_resume_state(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: str | None = None,
) -> OffPolicyResumeState:
    if is_structured_offpolicy_checkpoint(checkpoint):
        training = checkpoint["training"]
        iteration = int(training.get("iteration", 0))
        if iteration <= 0 and checkpoint_path is not None:
            iteration = parse_checkpoint_iteration(checkpoint_path)
        replay = None
        if is_full_offpolicy_checkpoint(checkpoint):
            replay = checkpoint.get("replay")
            if replay is not None and not isinstance(replay, dict):
                raise ValueError("Full off-policy checkpoint has invalid replay payload")
        return OffPolicyResumeState(
            learner=extract_learner_state_dict(checkpoint),
            iteration=iteration,
            replay=replay,
            reward_stats_ptr=int(training.get("reward_stats_ptr", 0)),
        )

    if is_legacy_learner_checkpoint(checkpoint):
        iteration = parse_checkpoint_iteration(checkpoint_path) if checkpoint_path else 0
        return OffPolicyResumeState(
            learner=extract_learner_state_dict(checkpoint),
            iteration=iteration,
            replay=None,
            reward_stats_ptr=0,
        )

    raise ValueError("Unrecognized off-policy checkpoint format")
