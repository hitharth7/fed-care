"""Tabular MLP for the CDC Diabetes Health Indicators binary task.

No BatchNorm by design: Opacus's per-sample gradient computation (used later
for DP-SGD) doesn't support BatchNorm. LayerNorm is used instead, so the
same architecture works unmodified once privacy is layered on in Step 7.
"""
import torch
import torch.nn as nn


class DiabetesMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw logits, BCEWithLogitsLoss applies sigmoid
