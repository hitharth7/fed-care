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

Two more things layered on for the drift-aware controller
(fl/drift_controller.py), both OFF by default so existing experiments
(Steps 5-8) are unaffected:
- fit()'s `config` dict may carry "mu" / "target_epsilon" overrides for
  THIS round only (falls back to the constructor's mu/dp_config when
  absent) -- this is how DriftAwareFedAvg actually controls each hospital.
- track_controller_metrics=True makes fit() also report "drift" (how far
  this round's local update moved from the global model it started from)
  and "overfit_gap" (this hospital's own train accuracy minus its own val
  accuracy) -- the two signals the controller reacts to.
"""
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from privacy.dp import DPConfig, make_dp_config, wrap_for_round


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
        track_controller_metrics: bool = False,
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
        self.track_controller_metrics = track_controller_metrics

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        # Networked clients (demo/) reuse ONE persistent client/model object
        # across every round -- unlike simulation mode, where a fresh model
        # is built each round. evaluate() leaves the model in .eval() mode,
        # and Opacus's validator rejects wrapping a model that isn't in
        # .train() mode, so this must happen before any DP wrapping below.
        self.model.train()

        # A controller (fl/drift_controller.py) can override mu / privacy
        # for just this round via config; absent that, use the fixed
        # values this client was constructed with (Steps 5-8's behavior).
        mu = float(config.get("mu", self.mu))
        dp_config = self.dp_config
        controller_target_epsilon = config.get("target_epsilon")
        if controller_target_epsilon is not None:
            batch_size = self.train_loader.batch_size or 128
            n = len(self.train_loader.dataset)
            steps_this_round = -(-n // batch_size) * self.local_epochs  # ceil
            dp_config = make_dp_config(
                target_epsilon=float(controller_target_epsilon),
                sample_rate=batch_size / n,
                total_steps=steps_this_round,
            )

        # Always snapshot the pre-training weights: needed for FedProx's
        # proximal term when mu > 0, and for the controller's drift signal
        # when track_controller_metrics is on. Cheap either way (a clone,
        # not a training pass).
        global_params = [p.detach().clone() for p in self.model.parameters()]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        model, train_loader = self.model, self.train_loader
        dp_wrapped = dp_config is not None and dp_config.noise_multiplier > 0
        if dp_wrapped:
            model, optimizer, train_loader = wrap_for_round(model, optimizer, train_loader, dp_config)

        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(self.local_epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                if mu > 0:
                    prox = sum((p - gp).pow(2).sum() for p, gp in zip(model.parameters(), global_params))
                    loss = loss + (mu / 2) * prox
                loss.backward()
                optimizer.step()

        # GradSampleModule (if DP was used) wraps self.model in place, so
        # self.model's own state_dict already reflects the trained weights.
        # Simulation-mode clients get a fresh model every round, but
        # networked clients (demo/) reuse this SAME model object across
        # rounds -- Opacus refuses to attach hooks to an already-wrapped
        # model, so they must be removed here or the very next round's DP
        # wrap (if any) raises "Trying to add hooks twice to the same model".
        if dp_wrapped:
            model.remove_hooks()

        metrics = {"client_id": self.client_id}

        if self.track_controller_metrics:
            post_flat = torch.cat([p.detach().flatten() for p in self.model.parameters()])
            pre_flat = torch.cat([p.flatten() for p in global_params])
            drift = float(torch.norm(post_flat - pre_flat) / (torch.norm(pre_flat) + 1e-8))
            _, _, train_metrics = evaluate_model(self.model, self.train_loader, self.device)
            _, _, val_metrics = evaluate_model(self.model, self.val_loader, self.device)
            overfit_gap = train_metrics["accuracy"] - val_metrics["accuracy"]
            metrics.update({"drift": drift, "overfit_gap": overfit_gap})

        if controller_target_epsilon is not None and dp_config.noise_multiplier > 0:
            metrics.update({
                "noise_multiplier": dp_config.noise_multiplier,
                "sample_rate": dp_config.sample_rate,
                "steps": dp_config.total_steps,
            })

        return get_parameters(self.model), len(self.train_loader.dataset), metrics

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
