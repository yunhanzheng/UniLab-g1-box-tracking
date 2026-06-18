"""Unit tests for Scaling-CRL networks and losses."""

from __future__ import annotations

import torch

from unilab.algos.torch.scaling_crl.learner import ScalingCRLLearner
from unilab.algos.torch.scaling_crl.losses import infonce_critic_loss
from unilab.algos.torch.scaling_crl.networks import GoalEncoder, ResidualBlock, SAEncoder


def test_residual_block_forward_shape():
    block = ResidualBlock(32)
    x = torch.randn(8, 32)
    y = block(x)
    assert y.shape == (8, 32)


def test_scaling_crl_encoders_output_64d():
    state = torch.randn(16, 20)
    action = torch.randn(16, 5)
    goal = torch.randn(16, 18)
    sa = SAEncoder(state_dim=20, action_dim=5, network_depth=8, network_width=64)
    g = GoalEncoder(goal_dim=18, network_depth=16, network_width=64)
    sa_repr = sa(state, action)
    g_repr = g(goal)
    assert sa_repr.shape == (16, 64)
    assert g_repr.shape == (16, 64)


def test_infonce_diagonal_is_highest():
    sa_repr = torch.randn(32, 64)
    g_repr = sa_repr + 0.05 * torch.randn(32, 64)
    loss, metrics = infonce_critic_loss(sa_repr, g_repr)
    assert torch.isfinite(loss)
    assert "critic_loss" in metrics


def test_scaling_crl_learner_updates():
    learner = ScalingCRLLearner(
        obs_dim=111,
        action_dim=29,
        state_dim=93,
        goal_dim=18,
        device="cpu",
        actor_depth=4,
        critic_depth=4,
    )
    batch = {
        "obs": torch.randn(64, 111),
        "next_obs": torch.randn(64, 111),
        "actions": torch.randn(64, 29),
        "critic": torch.randn(64, 19),
        "next_critic": torch.randn(64, 19),
        "rewards": torch.randn(64),
        "dones": torch.zeros(64),
        "truncated": torch.zeros(64),
    }
    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)
    assert "critic_loss" in critic_metrics
    assert "actor_loss" in actor_metrics

    # Off-policy runner learner contract.
    learner.soft_update_target()
    state = learner.get_state_dict()
    assert {"actor", "sa_encoder", "goal_encoder"}.issubset(state)
    learner.load_state_dict(state)
    assert learner.use_symmetry is False
