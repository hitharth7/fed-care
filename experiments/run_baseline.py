"""Centralized baseline: train DiabetesMLP on the full pooled dataset.

This is the accuracy ceiling everything else (local-only, FedAvg, FedProx,
with/without DP) gets compared against. Sanity check: accuracy should land
near published kernels for this dataset (~75-86%) — if wildly off, something
is wrong before FL/DP code is even introduced.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.loaders import load_features_targets
from models.mlp import DiabetesMLP

SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    all_probs, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        probs = torch.sigmoid(model(xb)).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(yb.numpy())
    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds = (probs >= 0.5).astype(np.float32)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1": float(f1_score(targets, preds)),
        "auc": float(roc_auc_score(targets, probs)),
    }


def train_baseline(
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    feature_loader=load_features_targets,
    results_filename: str = "baseline_metrics.json",
) -> tuple:
    """`feature_loader`/`results_filename` default to the diabetes dataset so
    every existing call site reproduces its exact prior behavior unchanged;
    pass `data.loaders.load_heart_features_targets` for the second specialty."""
    set_seed()
    device = get_device()
    print(f"device: {device}")

    X, y = feature_loader()
    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
    )

    def to_loader(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, True)
    val_loader = to_loader(X_val, y_val, False)
    test_loader = to_loader(X_test, y_test, False)

    model = DiabetesMLP(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_accuracy": [], "val_f1": [], "val_auc": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(epoch_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["auc"])
        print(
            f"epoch {epoch + 1}/{epochs}  loss={epoch_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}  "
            f"val_auc={val_metrics['auc']:.4f}"
        )

    test_metrics = evaluate(model, test_loader, device)
    print("test metrics:", test_metrics)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / results_filename, "w") as f:
        json.dump({"history": history, "test": test_metrics}, f, indent=2)

    return model, history, test_metrics


if __name__ == "__main__":
    train_baseline()
