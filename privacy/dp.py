"""Opacus DP-SGD wrapper with correct multi-round epsilon accounting.

Trap this avoids (flagged explicitly in the project plan as the single most
common correctness bug in "FL + DP-SGD" projects): privacy loss composes
across ALL communication rounds a client participates in, not per-round in
isolation. Flower's client_fn is stateless across rounds (a fresh client
object can be built each round), so we can't rely on one PrivacyEngine
instance persisting across rounds either. Instead:

1. Before training starts, calibrate ONE noise_multiplier for the client's
   TOTAL step budget across every round it will do (`total_steps`).
2. Reuse that exact noise_multiplier every round (re-wrapping the fresh
   per-round model/optimizer/loader each time, but always with this same
   noise level) -- so the accountant's assumption (fixed noise per step)
   holds across the whole run.
3. Report the final epsilon once, after all rounds, from
   (noise_multiplier, sample_rate, total_steps, delta) -- a deterministic
   function of the whole training run, not a per-round guess.
"""
from dataclasses import dataclass

import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

DEFAULT_DELTA = 1e-5


@dataclass
class DPConfig:
    target_epsilon: float  # float("inf") means "no DP" (noise_multiplier=0)
    delta: float
    sample_rate: float
    total_steps: int
    noise_multiplier: float
    max_grad_norm: float = 1.0


def make_dp_config(
    target_epsilon: float,
    sample_rate: float,
    total_steps: int,
    delta: float = DEFAULT_DELTA,
    max_grad_norm: float = 1.0,
) -> DPConfig:
    if target_epsilon == float("inf") or total_steps == 0:
        noise_multiplier = 0.0
    else:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=total_steps,
        )
    return DPConfig(target_epsilon, delta, sample_rate, total_steps, noise_multiplier, max_grad_norm)


def wrap_for_round(model: nn.Module, optimizer, data_loader, dp_config: DPConfig):
    """Wrap model/optimizer/data_loader for one round of DP-SGD local
    training. GradSampleModule wraps `model` in place (shares the same
    parameter tensors), so `model` itself reflects trained weights
    afterwards -- callers don't need to unwrap anything."""
    privacy_engine = PrivacyEngine()
    return privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
    )


def compute_final_epsilon(dp_config: DPConfig) -> float:
    """Report epsilon for the *entire* training run (all rounds), not per-round."""
    if dp_config.noise_multiplier == 0.0:
        return float("inf")
    accountant = RDPAccountant()
    for _ in range(dp_config.total_steps):
        accountant.step(noise_multiplier=dp_config.noise_multiplier, sample_rate=dp_config.sample_rate)
    return accountant.get_epsilon(delta=dp_config.delta)
