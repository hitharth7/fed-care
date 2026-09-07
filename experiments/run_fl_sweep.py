"""FL core: fully-local vs vanilla FedAvg, on Dirichlet non-IID hospital shards.

Step 5 scope only: {local, FedAvg} conditions. FedProx personalization
(Step 6) and DP (Step 7) extend this same script's data pipeline.

Two runs:
1. Sanity check (Verification Plan #2): FedAvg at high alpha (near-IID)
   should converge close to the centralized baseline (Step 4: ~86.6% acc,
   0.83 AUC) -- proves the Flower loop itself is correct before non-IID
   or privacy is layered on.
2. Main comparison: moderate non-IID (alpha=0.5), fully-local vs FedAvg,
   reporting per-client accuracy -- especially the worst-served client.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data.loaders import load_features_targets
from data.partition import build_experiment_splits
from fl.client import (
    DiabetesFlowerClient,
    compute_pos_weight,
    evaluate_model,
    make_criterion,
    set_parameters,
    tune_threshold,
)
from fl.strategies import make_fedavg_strategy
from models.mlp import DiabetesMLP
from privacy.dp import DPConfig, compute_final_epsilon, make_dp_config, wrap_for_round

SEED = 42
NUM_CLIENTS = 5
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


def make_loader(X: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def prepare_data(alpha: float, num_clients: int = NUM_CLIENTS, batch_size: int = 128):
    """Returns (splits, client_loaders[(train, val)], input_dim).

    Feature scaling is fit once on the pooled training pool (all clients'
    train+val, excluding the global test set) -- a deliberate simplification:
    a real deployment would need federated feature statistics, but that's a
    separate (solved) problem and out of scope here.
    """
    X_df, y_s = load_features_targets()
    splits = build_experiment_splits(y_s, num_clients, alpha, seed=SEED)

    X = X_df.values.astype(np.float32)
    y = y_s.values.astype(np.float32)

    train_pool_idx = np.concatenate(
        [c["train"] for c in splits["clients"]] + [c["val"] for c in splits["clients"]]
    )
    scaler = StandardScaler().fit(X[train_pool_idx])
    X = scaler.transform(X).astype(np.float32)

    client_loaders = [
        (
            make_loader(X, y, c["train"], batch_size, shuffle=True),
            make_loader(X, y, c["val"], batch_size, shuffle=False),
        )
        for c in splits["clients"]
    ]
    return splits, client_loaders, X.shape[1]


def client_dp_configs(
    client_loaders, target_epsilon: float, batch_size: int, local_epochs: int, num_rounds: int
) -> list[DPConfig]:
    """One DPConfig per client, each calibrated for that client's OWN total
    step budget (its dataset size differs -> its sample_rate differs) across
    the FULL multi-round run -- see privacy/dp.py for why this must be done
    once up front rather than per round."""
    configs = []
    for train_loader, _ in client_loaders:
        n = len(train_loader.dataset)
        steps_per_round = -(-n // batch_size) * local_epochs  # ceil
        total_steps = steps_per_round * num_rounds
        sample_rate = batch_size / n
        configs.append(make_dp_config(target_epsilon, sample_rate, total_steps))
    return configs


def run_fully_local(
    client_loaders,
    input_dim: int,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    dp_configs: list[DPConfig] | None = None,
    return_models: bool = False,
    use_pos_weight: bool = False,
    per_client_threshold: bool = False,
):
    """Each hospital trains alone -- no communication. Lower bound.
    If `dp_configs` is given, client i's local training is DP-SGD wrapped
    with dp_configs[i] (total_steps there must equal epochs * steps/epoch,
    NOT multiplied by num_rounds -- there are no FL rounds here).
    If `return_models` is True, returns (per_client_metrics, models) so
    Step 8's MIA audit can attack these exact trained models without
    retraining them.
    `use_pos_weight`/`per_client_threshold` default off so this reproduces
    the original results exactly; both must be passed to get a fair
    comparison against a `run_fl_condition` call with the same flags on."""
    per_client_metrics = []
    models = []
    for i, (train_loader, val_loader) in enumerate(client_loaders):
        set_seed(SEED + i)
        model = DiabetesMLP(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader = train_loader
        dp_config = dp_configs[i] if dp_configs else None
        if dp_config is not None and dp_config.noise_multiplier > 0:
            model, optimizer, loader = wrap_for_round(model, optimizer, loader, dp_config)
        pos_weight = compute_pos_weight(train_loader) if use_pos_weight else None
        criterion = make_criterion(pos_weight, device)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        threshold = tune_threshold(model, train_loader, device) if per_client_threshold else 0.5
        _, n, metrics = evaluate_model(model, val_loader, device, threshold)
        metrics["client_id"] = i
        metrics["n_train"] = len(train_loader.dataset)
        if dp_config is not None:
            metrics["epsilon"] = compute_final_epsilon(dp_config)
        per_client_metrics.append(metrics)
        models.append(model)
        print(f"[local] client {i}: {metrics}")
    if return_models:
        return per_client_metrics, models
    return per_client_metrics


def run_fl_condition(
    client_loaders,
    input_dim: int,
    device: torch.device,
    num_rounds: int = 10,
    local_epochs: int = 1,
    lr: float = 1e-3,
    mu: float = 0.0,
    dp_configs: list[DPConfig] | None = None,
    use_pos_weight: bool = False,
    per_client_threshold: bool = False,
):
    """FedAvg (mu=0) or FedProx (mu>0) over `num_rounds` rounds, optionally
    with per-client DP-SGD (dp_configs[i], calibrated for num_rounds).

    `use_pos_weight`/`per_client_threshold` default off, reproducing the
    original results exactly. Pass both True to correct for label shift:
    each client weights its local BCE loss by its own class imbalance, and
    picks its evaluation threshold on its own train shard rather than using
    a fixed 0.5 (see fl/client.py -- a fixed threshold, calibrated to the
    POOLED prior, is why the alpha=0.5 FedAvg run collapses client 1, whose
    local positive rate of 65% is nothing like the pooled 14%).

    Returns (history, final_per_client_metrics, final_global_model). The
    final global model is the single shared model FedAvg/FedProx actually
    deploys (FedProx changes local training dynamics via the proximal term,
    but -- per Li et al. 2020 -- still converges to one shared model, not a
    distinct model per client); Step 8's MIA audit attacks this model.
    """
    num_clients = len(client_loaders)
    sink: dict = {}
    param_sink: dict = {}

    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        model = DiabetesMLP(input_dim=input_dim)
        train_loader, val_loader = client_loaders[cid]
        dp_config = dp_configs[cid] if dp_configs else None
        return DiabetesFlowerClient(
            model,
            train_loader,
            val_loader,
            device,
            local_epochs=local_epochs,
            lr=lr,
            client_id=cid,
            mu=mu,
            dp_config=dp_config,
            use_pos_weight=use_pos_weight,
            per_client_threshold=per_client_threshold,
        ).to_client()

    strategy = make_fedavg_strategy(num_clients, per_round_sink=sink, param_sink=param_sink)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
        ray_init_args={"log_to_driver": False},
    )

    final_per_client = [{**metrics, "n_val": n} for n, metrics in sink["latest"]]
    final_per_client.sort(key=lambda m: m["client_id"])
    if dp_configs:
        for m in final_per_client:
            m["epsilon"] = compute_final_epsilon(dp_configs[m["client_id"]])

    final_model = DiabetesMLP(input_dim=input_dim)
    set_parameters(final_model, param_sink["params"])
    return history, final_per_client, final_model


def run_fedavg(
    client_loaders,
    input_dim: int,
    device: torch.device,
    num_rounds: int = 10,
    local_epochs: int = 1,
    lr: float = 1e-3,
    use_pos_weight: bool = False,
    per_client_threshold: bool = False,
):
    """Vanilla FedAvg -- thin wrapper over run_fl_condition(mu=0)."""
    return run_fl_condition(
        client_loaders,
        input_dim,
        device,
        num_rounds,
        local_epochs,
        lr,
        mu=0.0,
        use_pos_weight=use_pos_weight,
        per_client_threshold=per_client_threshold,
    )


def summarize(condition: str, per_client_metrics: list[dict]) -> dict:
    """Tracks accuracy, balanced accuracy, and AUC -- accuracy alone can be
    misleading here: under this dataset's 86/14 imbalance, an all-negative
    predictor scores 0.94-0.99 accuracy on the negative-heavy clients while
    being a coin flip (balanced accuracy 0.50), and under moderate non-IID +
    pooled FedAvg/FedProx, per-client predictions can collapse to the
    majority class (f1=0) for every client, pinning accuracy at a fixed
    floor regardless of mu even though the underlying probability outputs
    (and therefore AUC) still move -- confirmed empirically while debugging
    the mu grid (Step 6): AUC varied clearly across mu while accuracy stayed
    bit-identical. balanced_accuracy is only present once per-client
    thresholds are used (fl/client.py); it falls back to accuracy otherwise
    since a fixed-0.5 balanced accuracy of ~0.50 for every client would
    otherwise swamp this table's signal."""
    accs = [m["accuracy"] for m in per_client_metrics]
    bals = [m.get("balanced_accuracy", m["accuracy"]) for m in per_client_metrics]
    aucs = [m["auc"] for m in per_client_metrics]
    worst_acc = min(per_client_metrics, key=lambda m: m["accuracy"])
    worst_auc = min(per_client_metrics, key=lambda m: m["auc"])
    print(
        f"\n=== {condition}: mean_acc={np.mean(accs):.4f}  mean_bal_acc={np.mean(bals):.4f}  "
        f"mean_auc={np.mean(aucs):.4f}  "
        f"worst_client(acc)={worst_acc['client_id']} (acc={worst_acc['accuracy']:.4f})  "
        f"worst_client(auc)={worst_auc['client_id']} (auc={worst_auc['auc']:.4f}) ==="
    )
    return {
        "condition": condition,
        "per_client": per_client_metrics,
        "mean_accuracy": float(np.mean(accs)),
        "mean_balanced_accuracy": float(np.mean(bals)),
        "mean_auc": float(np.mean(aucs)),
        "worst_client_id": int(worst_acc["client_id"]),
        "worst_client_accuracy": float(worst_acc["accuracy"]),
        "worst_client_auc": float(worst_auc["auc"]),
    }


def main():
    device = get_device()
    print(f"device: {device}")

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    # --- 1. Sanity check: near-IID FedAvg should track the centralized baseline ---
    print("\n>>> Sanity check: FedAvg at alpha=100 (near-IID)")
    _, client_loaders_iid, input_dim = prepare_data(alpha=100.0)
    _, sanity_metrics, _ = run_fedavg(client_loaders_iid, input_dim, device, num_rounds=10)
    all_results["sanity_fedavg_alpha100"] = summarize("fedavg_alpha100", sanity_metrics)

    baseline_path = RESULTS_DIR / "baseline_metrics.json"
    if baseline_path.exists():
        baseline_acc = json.loads(baseline_path.read_text())["test"]["accuracy"]
        gap = baseline_acc - all_results["sanity_fedavg_alpha100"]["mean_accuracy"]
        print(f"centralized baseline acc={baseline_acc:.4f}  near-IID FedAvg mean acc={all_results['sanity_fedavg_alpha100']['mean_accuracy']:.4f}  gap={gap:.4f}")
        all_results["sanity_fedavg_alpha100"]["baseline_gap"] = float(gap)

    # --- 2. Main comparison: moderate non-IID (alpha=0.5), local vs FedAvg ---
    print("\n>>> Main comparison: alpha=0.5 (moderate non-IID)")
    _, client_loaders_noniid, input_dim = prepare_data(alpha=0.5)

    local_metrics = run_fully_local(client_loaders_noniid, input_dim, device, epochs=10)
    all_results["local_alpha0.5"] = summarize("local_alpha0.5", local_metrics)

    _, fedavg_metrics, _ = run_fedavg(client_loaders_noniid, input_dim, device, num_rounds=10)
    all_results["fedavg_alpha0.5"] = summarize("fedavg_alpha0.5", fedavg_metrics)

    with open(RESULTS_DIR / "fl_step5_local_vs_fedavg.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR / 'fl_step5_local_vs_fedavg.json'}")
    return all_results


def plot_condition_comparison(comparison: dict, save_path: Path) -> None:
    """Two panels, not one: accuracy alone can hide what's happening here --
    at alpha=0.5, FedAvg/FedProx often collapse every client's predictions
    to the majority class (f1=0), pinning accuracy at a fixed floor
    regardless of mu, while AUC (using continuous probabilities) still moves.
    Both panels are needed to see the real picture."""
    import matplotlib.pyplot as plt

    conditions = ["local", "fedavg", "fedprox_best"]
    labels = ["Fully local", "Vanilla FedAvg", f"FedProx (mu={comparison['best_mu']})"]
    num_clients = len(comparison["local"]["per_client"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.25
    x = np.arange(num_clients)
    for metric, ax in zip(["accuracy", "auc"], axes):
        for i, cond in enumerate(conditions):
            per_client = sorted(comparison[cond]["per_client"], key=lambda m: m["client_id"])
            values = [m[metric] for m in per_client]
            ax.bar(x + (i - 1) * width, values, width, label=labels[i])
        ax.set_xticks(x)
        ax.set_xticklabels([f"client_{i}" for i in range(num_clients)])
        ax.set_ylabel(metric)
        ax.set_title(f"Per-client {metric}")
    axes[1].legend()
    fig.suptitle("local vs FedAvg vs FedProx (alpha=0.5)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_step6_fedprox(device: torch.device, alpha: float = 0.5, num_rounds: int = 10, mus=(0.001, 0.01, 0.1, 1.0)) -> tuple[dict, float]:
    """FedProx mu grid search (validation split = each client's own val set),
    then compare best-mu FedProx against Step 5's local/FedAvg results on
    the same worst-served-client metric."""
    _, client_loaders, input_dim = prepare_data(alpha=alpha)

    grid_results = {}
    for mu in mus:
        print(f"\n>>> FedProx mu={mu}")
        _, metrics, _ = run_fl_condition(client_loaders, input_dim, device, num_rounds=num_rounds, mu=mu)
        grid_results[str(mu)] = summarize(f"fedprox_mu{mu}", metrics)

    # Selected by mean AUC, not worst-client accuracy: at alpha=0.5 every mu
    # value collapses every client's predictions to the majority class
    # (f1=0), which pins accuracy at an identical floor regardless of mu --
    # AUC is where mu's actual effect on the model is visible (see summarize()).
    best_mu = max(grid_results, key=lambda k: grid_results[k]["mean_auc"])
    print(f"\nBest mu by mean AUC (accuracy is saturated/uninformative here -- see summarize() docstring): {best_mu}")

    step5_path = RESULTS_DIR / "fl_step5_local_vs_fedavg.json"
    step5 = json.loads(step5_path.read_text()) if step5_path.exists() else {}

    comparison = {
        "local": step5.get("local_alpha0.5"),
        "fedavg": step5.get("fedavg_alpha0.5"),
        "fedprox_best": grid_results[best_mu],
        "best_mu": float(best_mu),
        "mu_grid": grid_results,
    }
    with open(RESULTS_DIR / "fl_step6_fedprox.json", "w") as f:
        json.dump(comparison, f, indent=2)

    plot_condition_comparison(comparison, RESULTS_DIR / "fl_step6_worst_client_comparison.png")
    print(f"Saved to {RESULTS_DIR / 'fl_step6_fedprox.json'} and fl_step6_worst_client_comparison.png")
    return comparison, float(best_mu)


def run_step5c_corrected(device: torch.device, alpha: float = 0.5, num_rounds: int = 10) -> dict:
    """Step 5's local-vs-FedAvg comparison, both corrected for label shift
    (pos_weight + per-client threshold), so the comparison is fair in BOTH
    directions -- Step 5's original numbers compare a threshold-broken
    FedAvg against threshold-broken local models (every local model except
    client 1 also scores f1=0 at 0.5, hidden by their own negative-heavy
    priors), so neither condition's original number is a trustworthy
    answer to "does collaboration help". This is the run that actually
    answers it: does FedAvg's shared model, once its output is read
    correctly, beat the hospital training alone on its own patients?
    """
    _, client_loaders, input_dim = prepare_data(alpha=alpha)

    local_metrics = run_fully_local(
        client_loaders, input_dim, device, epochs=10, use_pos_weight=True, per_client_threshold=True
    )
    local_summary = summarize("local_corrected", local_metrics)

    _, fedavg_metrics, _ = run_fedavg(
        client_loaders, input_dim, device, num_rounds=num_rounds, use_pos_weight=True, per_client_threshold=True
    )
    fedavg_summary = summarize("fedavg_corrected", fedavg_metrics)

    print(f"\n{'client':>7} {'local_acc':>10} {'fedavg_acc':>11} {'local_bal':>10} {'fedavg_bal':>11} {'local_f1':>9} {'fedavg_f1':>10}")
    print("-" * 72)
    for lm, fm in zip(
        sorted(local_metrics, key=lambda m: m["client_id"]),
        sorted(fedavg_metrics, key=lambda m: m["client_id"]),
    ):
        print(
            f"{lm['client_id']:>7} {lm['accuracy']:>10.4f} {fm['accuracy']:>11.4f} "
            f"{lm['balanced_accuracy']:>10.4f} {fm['balanced_accuracy']:>11.4f} "
            f"{lm['f1']:>9.4f} {fm['f1']:>10.4f}"
        )

    verdict = "FedAvg beats local" if fedavg_summary["mean_balanced_accuracy"] > local_summary["mean_balanced_accuracy"] else "local still beats FedAvg"
    print(f"\nmean balanced accuracy: local={local_summary['mean_balanced_accuracy']:.4f}  "
          f"fedavg={fedavg_summary['mean_balanced_accuracy']:.4f}  -> {verdict}")

    result = {"local_corrected": local_summary, "fedavg_corrected": fedavg_summary}
    with open(RESULTS_DIR / "fl_step5c_corrected.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {RESULTS_DIR / 'fl_step5c_corrected.json'}")
    return result


if __name__ == "__main__":
    main()
    device = get_device()
    run_step6_fedprox(device)
    run_step5c_corrected(device)
