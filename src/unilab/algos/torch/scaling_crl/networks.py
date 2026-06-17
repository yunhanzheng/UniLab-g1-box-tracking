"""Residual encoder networks for Scaling-CRL.

Aligned with ``scaling-crl/train.py``:
  - ``residual_block`` (lines 110-125)
  - ``SA_encoder`` (lines 127-159)
  - ``G_encoder`` (lines 161-193)
  - ``Actor`` (lines 195-234)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lecun_uniform_init(tensor: nn.Parameter) -> None:
    fan_in = tensor.shape[1] if tensor.dim() > 1 else tensor.numel()
    bound = math.sqrt(1.0 / (3.0 * fan_in))
    nn.init.uniform_(tensor, -bound, bound)


class ResidualBlock(nn.Module):
    """Dense + LayerNorm + Swish x4 with residual connection (Scaling-CRL default)."""

    def __init__(self, width: int, use_relu: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(4):
            layers.extend(
                [
                    nn.Linear(width, width),
                    nn.LayerNorm(width),
                    nn.ReLU() if use_relu else nn.SiLU(),
                ]
            )
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                _lecun_uniform_init(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SAEncoder(nn.Module):
    """State-action encoder (maps to 64-dim embedding)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        network_width: int = 256,
        network_depth: int = 4,
        embed_dim: int = 64,
        use_relu: bool = False,
    ):
        super().__init__()
        activation = nn.ReLU if use_relu else nn.SiLU
        self.input = nn.Sequential(
            nn.Linear(state_dim + action_dim, network_width),
            nn.LayerNorm(network_width),
            activation(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(network_width, use_relu=use_relu)
                for _ in range(max(network_depth // 4, 1))
            ]
        )
        self.output = nn.Linear(network_width, embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.input:
            if isinstance(module, nn.Linear):
                _lecun_uniform_init(module.weight)
                nn.init.zeros_(module.bias)
        _lecun_uniform_init(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.input(x)
        x = self.blocks(x)
        return self.output(x)


class GoalEncoder(nn.Module):
    """Goal encoder psi (maps to 64-dim embedding)."""

    def __init__(
        self,
        goal_dim: int,
        network_width: int = 256,
        network_depth: int = 4,
        embed_dim: int = 64,
        use_relu: bool = False,
    ):
        super().__init__()
        activation = nn.ReLU if use_relu else nn.SiLU
        self.input = nn.Sequential(
            nn.Linear(goal_dim, network_width),
            nn.LayerNorm(network_width),
            activation(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(network_width, use_relu=use_relu)
                for _ in range(max(network_depth // 4, 1))
            ]
        )
        self.output = nn.Linear(network_width, embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.input:
            if isinstance(module, nn.Linear):
                _lecun_uniform_init(module.weight)
                nn.init.zeros_(module.bias)
        _lecun_uniform_init(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        x = self.input(goal)
        x = self.blocks(x)
        return self.output(x)


class ScalingCRLActor(nn.Module):
    """SAC-style Gaussian actor on concat(state, goal)."""

    LOG_STD_MAX = 2.0
    LOG_STD_MIN = -5.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        network_width: int = 256,
        network_depth: int = 4,
        use_relu: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        del device
        activation = nn.ReLU if use_relu else nn.SiLU
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, network_width),
            nn.LayerNorm(network_width),
            activation(),
            *[
                ResidualBlock(network_width, use_relu=use_relu)
                for _ in range(max(network_depth // 4, 1))
            ],
        )
        self.mean = nn.Linear(network_width, action_dim)
        self.log_std = nn.Linear(network_width, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                _lecun_uniform_init(module.weight)
                nn.init.zeros_(module.bias)
        _lecun_uniform_init(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        _lecun_uniform_init(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        del dones
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return torch.tanh(dist.rsample())

    def as_export_module(self) -> nn.Module:
        actor = self

        class _Wrapper(nn.Module):
            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                mean, _ = actor.forward(obs)
                return torch.tanh(mean)

        return _Wrapper()
