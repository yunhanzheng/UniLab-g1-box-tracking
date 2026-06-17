"""Scaling-CRL learner for UniLab off-policy async runner."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from unilab.algos.torch.scaling_crl.goal_relabeling import relabel_goals_from_batch
from unilab.algos.torch.scaling_crl.losses import actor_critic_distance, infonce_critic_loss
from unilab.algos.torch.scaling_crl.networks import GoalEncoder, SAEncoder, ScalingCRLActor


class ScalingCRLLearner:
    """Contrastive goal-conditioned learner (Scaling-CRL port)."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        goal_dim: int,
        device: str,
        gamma: float = 0.99,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        actor_network_width: int = 256,
        critic_network_width: int = 256,
        actor_depth: int = 4,
        critic_depth: int = 4,
        embed_dim: int = 64,
        logsumexp_penalty_coeff: float = 0.1,
        entropy_param: float = 0.5,
        disable_entropy: bool = False,
        use_relu: bool = False,
        critic_obs_dim: int = 0,
    ):
        del critic_obs_dim
        self.device = torch.device(device)
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.gamma = float(gamma)
        self.logsumexp_penalty_coeff = float(logsumexp_penalty_coeff)
        self.disable_entropy = bool(disable_entropy)
        self.update_count = 0

        self.actor = ScalingCRLActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            network_width=actor_network_width,
            network_depth=actor_depth,
            use_relu=use_relu,
            device=self.device,
        ).to(self.device)
        self.sa_encoder = SAEncoder(
            state_dim=self.state_dim,
            action_dim=action_dim,
            network_width=critic_network_width,
            network_depth=critic_depth,
            embed_dim=embed_dim,
            use_relu=use_relu,
        ).to(self.device)
        self.goal_encoder = GoalEncoder(
            goal_dim=self.goal_dim,
            network_width=critic_network_width,
            network_depth=critic_depth,
            embed_dim=embed_dim,
            use_relu=use_relu,
        ).to(self.device)

        self.log_alpha = nn.Parameter(
            torch.tensor(0.0, device=self.device, dtype=torch.float32), requires_grad=True
        )
        self.target_entropy = -entropy_param * float(action_dim)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(
            list(self.sa_encoder.parameters()) + list(self.goal_encoder.parameters()),
            lr=critic_lr,
        )
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

    def _unpack_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        critic = batch.get("critic")
        next_critic = batch.get("next_critic")
        state = obs[:, : self.state_dim]
        next_state = next_obs[:, : self.state_dim]
        if critic is None:
            achieved_goals = obs[:, self.state_dim : self.state_dim + self.goal_dim]
            next_achieved_goals = next_obs[:, self.state_dim : self.state_dim + self.goal_dim]
            episode_seed = torch.zeros(state.shape[0], device=state.device)
        else:
            achieved_goals = critic[:, : self.goal_dim]
            next_achieved_goals = next_critic[:, : self.goal_dim] if next_critic is not None else achieved_goals
            episode_seed = critic[:, self.goal_dim]
        actor_obs, goals = relabel_goals_from_batch(
            state=state,
            next_state=next_state,
            achieved_goals=achieved_goals,
            next_achieved_goals=next_achieved_goals,
            episode_seed=episode_seed,
            goal_dim=self.goal_dim,
            gamma=self.gamma,
        )
        return {
            "state": state,
            "actions": batch["actions"],
            "goals": goals,
            "actor_obs": actor_obs,
        }

    def update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        data = self._unpack_batch(batch)
        sa_repr = self.sa_encoder(data["state"], data["actions"])
        g_repr = self.goal_encoder(data["goals"])
        loss, metrics = infonce_critic_loss(
            sa_repr,
            g_repr,
            logsumexp_penalty_coeff=self.logsumexp_penalty_coeff,
        )
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self.update_count += 1
        return {k: float(v.item()) for k, v in metrics.items()}

    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        data = self._unpack_batch(batch)
        actions, log_prob = self.actor.sample(data["actor_obs"])
        sa_repr = self.sa_encoder(data["state"], actions)
        g_repr = self.goal_encoder(data["goals"])
        q_pi = actor_critic_distance(sa_repr, g_repr)
        if self.disable_entropy:
            actor_loss = -q_pi.mean()
        else:
            alpha = self.log_alpha.exp()
            actor_loss = (alpha.detach() * log_prob - q_pi).mean()
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        return {
            "actor_loss": float(actor_loss.item()),
            "log_alpha": float(self.log_alpha.item()),
            "sample_entropy": float((-log_prob).mean().item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "sa_encoder": self.sa_encoder.state_dict(),
            "goal_encoder": self.goal_encoder.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.sa_encoder.load_state_dict(state["sa_encoder"])
        self.goal_encoder.load_state_dict(state["goal_encoder"])
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.update_count = int(state.get("update_count", 0))
