"""Geometric future goal relabeling for Scaling-CRL.

Port of ``scaling-crl/buffer.py`` ``TrajectoryUniformSamplingQueue.flatten_crl_fn``.
UniLab uses flat replay batches, so relabeling approximates future-goal sampling by
selecting batch elements with matching episode seeds and geometric weights.
"""

from __future__ import annotations

import torch


def relabel_goals_from_batch(
    *,
    state: torch.Tensor,
    next_state: torch.Tensor,
    achieved_goals: torch.Tensor,
    next_achieved_goals: torch.Tensor,
    episode_seed: torch.Tensor,
    goal_dim: int,
    gamma: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (actor_obs, goal) with geometric future goals when possible."""
    batch_size = state.shape[0]
    device = state.device
    goals = achieved_goals.clone()
    for i in range(batch_size):
        same_episode = episode_seed == episode_seed[i]
        future_mask = torch.arange(batch_size, device=device) >= i
        mask = same_episode & future_mask
        if not torch.any(mask):
            goals[i] = next_achieved_goals[i]
            continue
        steps = torch.arange(batch_size, device=device) - i
        weights = (gamma ** steps.float()).clamp(min=1e-8)
        weights = weights * mask.float()
        weights[i] = weights[i] + 1e-5
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, num_samples=1).item()
        goals[i] = achieved_goals[idx]
    actor_obs = torch.cat([state, goals], dim=-1)
    return actor_obs, goals
