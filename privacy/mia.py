"""Yeom et al. (2018) loss-threshold membership-inference attack.

For a trained model, per-sample loss is used as the membership signal: a
record with unusually LOW loss is more likely to have been a training
member (the model fit it better). We report Attack AUC (sweep the
threshold, ROC over -loss as score) rather than a single accuracy number:
0.5 = attacker no better than a coin flip (ideal privacy), 1.0 = perfect
leakage.

Chosen over a full Shokri et al. (2017) multi-shadow-model attack as the
primary method because it needs zero extra shadow-model training -- much
cheaper on free-tier compute.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def per_sample_losses(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        losses.append(criterion(model(xb), yb).cpu().numpy())
    return np.concatenate(losses)


def loss_threshold_attack_auc(member_losses: np.ndarray, non_member_losses: np.ndarray) -> float:
    """ROC-AUC using -loss as the membership score (lower loss -> more
    likely a member), sweeping every possible threshold rather than fixing one."""
    scores = np.concatenate([-member_losses, -non_member_losses])
    labels = np.concatenate([np.ones_like(member_losses), np.zeros_like(non_member_losses)])
    return float(roc_auc_score(labels, scores))


def run_mia(model: nn.Module, member_loader, non_member_loader, device: torch.device) -> dict:
    member_losses = per_sample_losses(model, member_loader, device)
    non_member_losses = per_sample_losses(model, non_member_loader, device)
    return {
        "attack_auc": loss_threshold_attack_auc(member_losses, non_member_losses),
        "member_loss_mean": float(member_losses.mean()),
        "non_member_loss_mean": float(non_member_losses.mean()),
        "n_members": int(len(member_losses)),
        "n_non_members": int(len(non_member_losses)),
    }
