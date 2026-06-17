"""Alias package for Scaling-CRL requested deliverable path."""

from unilab.algos.torch.scaling_crl import (
    GoalEncoder,
    SAEncoder,
    ScalingCRLActor,
    ScalingCRLLearner,
)

__all__ = [
    "GoalEncoder",
    "SAEncoder",
    "ScalingCRLActor",
    "ScalingCRLLearner",
]
