"""FlowerClient wrapping DiabetesMLP.

One class handles all three FL conditions plus optional DP, composably:
- mu == 0: vanilla FedAvg local update.
- mu  > 0: FedProx (proximal term mu/2 * ||w - w_global||^2 added to the
  local loss, pulling each client toward the global model while still
  letting it drift toward its own population).
- dp_config given: wraps the round's training in Opacus DP-SGD, using a
  noise_multiplier precalibrated for the client's *total* multi-round step
  budget (see privacy/dp.py) -- composes with either mu setting above.

Kept as one class rather than one subclass per condition because Step 7
sweeps all three FL conditions across an epsilon grid.
"""
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from privacy.dp import DPConfig, wrap_for_round


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


class DiabetesFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        local_epochs: int = 1,
        lr: float = 1e-3,
        client_id: int = -1,
        mu: float = 0.0,
        dp_config: DPConfig | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.client_id = client_id
        self.mu = mu
        self.dp_config = dp_config

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        global_params = (
            [p.detach().clone() for p in self.model.parameters()] if self.mu > 0 else None
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        model, train_loader = self.model, self.train_loader
        if self.dp_config is not None and self.dp_config.noise_multiplier > 0:
            model, optimizer, train_loader = wrap_for_round(model, optimizer, train_loader, self.dp_config)

        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(self.local_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                if global_params is not None:
                    prox = sum((p - gp).pow(2).sum() for p, gp in zip(model.parameters(), global_params))
                    loss = loss + (self.mu / 2) * prox
                loss.backward()
                optimizer.step()

        # GradSampleModule (if DP was used) wraps self.model in place, so
        # self.model's own state_dict already reflects the trained weights.
        return get_parameters(self.model), len(self.train_loader.dataset), {"client_id": self.client_id}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, n, metrics = evaluate_model(self.model, self.val_loader, self.device)
        metrics["client_id"] = self.client_id
        return loss, n, metrics


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: torch.device) -> tuple[float, int, dict]:
    """Shared eval used by the Flower client, the fully-local baseline loop,
    and the per-client / per-condition result tables."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_probs, all_targets = [], []
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(yb.cpu().numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds = (probs >= 0.5).astype(np.float32)
    n = len(loader.dataset)
    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "auc": float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5,
    }
    return total_loss / n, n, metrics
