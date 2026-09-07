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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from privacy.dp import DPConfig, make_dp_config, wrap_for_round


def compute_pos_weight(loader) -> float:
    """n_neg/n_pos for this client's OWN shard, for BCEWithLogitsLoss.

    Local positive rates at alpha=0.5 range from 0.8% (client 0) to 65%
    (client 1) against a 13.9% pooled rate. Unweighted BCE drives each
    client's update toward its own majority class, which is why 4 of 5
    clients score f1=0 and the model's probabilities sit far below 0.5.
    """
    y = loader.dataset.tensors[1]
    n_pos = float(y.sum())
    return float(len(y) - n_pos) / max(n_pos, 1.0)


def make_criterion(pos_weight: float | None, device: torch.device) -> nn.BCEWithLogitsLoss:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared forward pass -- used by evaluate_model and tune_threshold.
    Returns logits (not probabilities) so the caller can compute BCE loss
    directly without a lossy sigmoid round-trip."""
    model.eval()
    all_logits, all_targets = [], []
    for xb, yb in loader:
        all_logits.append(model(xb.to(device)).cpu())
        all_targets.append(yb)
    return torch.cat(all_logits), torch.cat(all_targets)


def tune_threshold(model: nn.Module, loader, device: torch.device, n_grid: int = 200) -> float:
    """Pick this client's decision threshold on data it OWNS (its train shard).

    A single 0.5 cut is wrong for any hospital whose class prior differs from
    the federation's: the alpha=0.5 global model puts EVERY one of client 1's
    (65% positive) patients on the negative side, scoring exactly its negative
    fraction (0.3472) despite ranking them well (AUC 0.82).

    Selected by balanced accuracy, not accuracy -- raw accuracy is maximized
    by predicting the local majority class, which is the failure being fixed.
    """
    logits, targets_t = predict_logits(model, loader, device)
    probs = torch.sigmoid(logits).numpy().ravel()
    targets = targets_t.numpy().ravel()
    if len(np.unique(targets)) < 2:
        return 0.5
    grid = np.unique(np.quantile(probs, np.linspace(0.001, 0.999, n_grid)))
    scores = [balanced_accuracy_score(targets, (probs >= t).astype(np.float32)) for t in grid]
    return float(grid[int(np.argmax(scores))])


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
        use_pos_weight: bool = False,
        per_client_threshold: bool = False,
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
        # Both default off so the original (threshold-0.5, unweighted) results
        # stay exactly reproducible as the "before" condition.
        self.pos_weight = compute_pos_weight(train_loader) if use_pos_weight else None
        self.per_client_threshold = per_client_threshold

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

        criterion = make_criterion(self.pos_weight, self.device)
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
        # Tuned on this client's own TRAIN shard, then applied to its val set --
        # no test-set leakage, and it only uses data the hospital already holds.
        threshold = (
            tune_threshold(self.model, self.train_loader, self.device)
            if self.per_client_threshold
            else 0.5
        )
        loss, n, metrics = evaluate_model(self.model, self.val_loader, self.device, threshold)
        metrics["client_id"] = self.client_id
        return loss, n, metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader, device: torch.device, threshold: float = 0.5
) -> tuple[float, int, dict]:
    """Shared eval used by the Flower client, the fully-local baseline loop,
    and the per-client / per-condition result tables.

    Reports balanced_accuracy alongside accuracy: under this dataset's 86/14
    imbalance, an all-negative predictor scores 0.94-0.99 accuracy on the
    negative-heavy clients while being a coin flip (balanced accuracy 0.50),
    so accuracy alone hides a globally degenerate model.
    """
    logits, targets_t = predict_logits(model, loader, device)
    loss = float(nn.BCEWithLogitsLoss()(logits, targets_t))
    probs = torch.sigmoid(logits).numpy().ravel()
    targets = targets_t.numpy().ravel()
    preds = (probs >= threshold).astype(np.float32)
    n = len(loader.dataset)
    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "auc": float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5,
        "threshold": float(threshold),
    }
    return loss, n, metrics
