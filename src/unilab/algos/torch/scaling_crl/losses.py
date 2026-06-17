"""Contrastive critic losses for Scaling-CRL.

Aligned with ``scaling-crl/train.py`` ``update_critic`` (InfoNCE + logsumexp reg).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_distance(sa_repr: torch.Tensor, g_repr: torch.Tensor) -> torch.Tensor:
    """Negative Euclidean distance logits (B, B)."""
    diff = sa_repr[:, None, :] - g_repr[None, :, :]
    return -torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8)


def infonce_critic_loss(
    sa_repr: torch.Tensor,
    g_repr: torch.Tensor,
    *,
    logsumexp_penalty_coeff: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = contrastive_distance(sa_repr, g_repr)
    diag = torch.diag(logits)
    critic_loss = -(diag - torch.logsumexp(logits, dim=1)).mean()
    logsumexp = torch.logsumexp(logits + 1e-6, dim=1)
    critic_loss = critic_loss + logsumexp_penalty_coeff * (logsumexp**2).mean()
    metrics = {
        "critic_loss": critic_loss.detach(),
        "logsumexp": logsumexp.detach().mean(),
    }
    return critic_loss, metrics


def actor_critic_distance(sa_repr: torch.Tensor, g_repr: torch.Tensor) -> torch.Tensor:
    """Q(s,a,g) = -||phi(s,a) - psi(g)||_2 (per-sample)."""
    return -torch.sqrt(torch.sum((sa_repr - g_repr) ** 2, dim=-1) + 1e-8)
