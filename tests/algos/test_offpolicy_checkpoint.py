from __future__ import annotations

import torch

from unilab.algos.torch.offpolicy.checkpoint import (
    OFFPOLICY_FULL_FORMAT,
    OFFPOLICY_WEIGHTS_FORMAT,
    build_offpolicy_checkpoint,
    extract_learner_state_dict,
    parse_offpolicy_resume_state,
    restore_replay_buffer,
    serialize_replay_buffer,
)
from unilab.ipc.replay_buffer import ReplayBuffer


def test_serialize_and_restore_replay_buffer_roundtrip() -> None:
    buffer = ReplayBuffer(capacity=16, obs_dim=3, action_dim=2, device="cpu", critic_dim=4)
    buffer._storage[0] = torch.arange(buffer._storage.shape[1], dtype=torch.float32)
    buffer._storage[1] = buffer._storage[0] + 1.0
    buffer.ptr[0] = 5
    buffer.size[0] = 5

    payload = serialize_replay_buffer(buffer)
    restored = ReplayBuffer(capacity=16, obs_dim=3, action_dim=2, device="cpu", critic_dim=4)
    restore_replay_buffer(restored, payload)

    assert int(restored.ptr[0]) == 5
    assert int(restored.size[0]) == 5
    torch.testing.assert_close(restored._storage, buffer._storage)


def test_weights_checkpoint_roundtrip_preserves_learner_only() -> None:
    learner_state = {"actor": {"w": torch.tensor([1.0, 2.0])}, "update_count": 7}

    checkpoint = build_offpolicy_checkpoint(
        learner_state=learner_state,
        iteration=1000,
        reward_stats_ptr=42,
    )
    resume = parse_offpolicy_resume_state(checkpoint, checkpoint_path="model_1000.pt")

    assert checkpoint["format"] == OFFPOLICY_WEIGHTS_FORMAT
    assert "replay" not in checkpoint
    assert resume.iteration == 1000
    assert resume.reward_stats_ptr == 42
    assert extract_learner_state_dict(checkpoint)["update_count"] == 7
    assert resume.replay is None


def test_legacy_full_checkpoint_still_loads_replay() -> None:
    buffer = ReplayBuffer(capacity=8, obs_dim=2, action_dim=1, device="cpu", critic_dim=0)
    buffer.ptr[0] = 3
    buffer.size[0] = 3
    learner_state = {"actor": {"w": torch.tensor([1.0, 2.0])}, "update_count": 7}
    checkpoint = {
        "format": OFFPOLICY_FULL_FORMAT,
        "version": 1,
        "learner": learner_state,
        "replay": serialize_replay_buffer(buffer),
        "training": {"iteration": 1000, "reward_stats_ptr": 42},
    }

    resume = parse_offpolicy_resume_state(checkpoint, checkpoint_path="model_1000.pt")
    assert resume.iteration == 1000
    assert resume.replay is not None

    restored = ReplayBuffer(capacity=8, obs_dim=2, action_dim=1, device="cpu", critic_dim=0)
    restore_replay_buffer(restored, resume.replay)
    assert int(restored.size[0]) == 3


def test_legacy_checkpoint_still_loads_learner_only() -> None:
    legacy = {"actor": {"w": torch.tensor([0.5])}, "update_count": 3}
    resume = parse_offpolicy_resume_state(legacy, checkpoint_path="model_500.pt")
    assert resume.iteration == 500
    assert resume.replay is None
    assert resume.learner["update_count"] == 3
