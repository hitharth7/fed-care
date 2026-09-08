"""Steps 7+8: DP-SGD epsilon sweep + membership-inference audit, combined
into one pass (train once per condition/epsilon, then immediately attack
the resulting model -- avoids retraining just for the attack).

Reduced grid vs. the plan's full {1,3,5,10,inf} x 3 conditions x 3-5 seeds:
that grid is explicitly sized for Colab GPU (see the plan's own compute
strategy), not this local CPU/MPS machine. Here: epsilons={1, 3, inf} x 3
conditions x 1 seed -- enough to validate the pipeline is CORRECT
(Verification Plan items 3-5) and show a real, if noisier, trend. Re-run
the full grid on Colab with more seeds before finalizing the paper.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data.loaders import load_features_targets
from experiments.run_fl_sweep import (
    RESULTS_DIR,
    client_dp_configs,
    get_device,
    make_loader,
    prepare_data,
    run_fl_condition,
    run_fully_local,
    set_seed,
)
from models.mlp import DiabetesMLP
from privacy.mia import run_mia


def sanity_check_attack(
    device: torch.device, n_members: int = 30, n_non_members: int = 5000, epochs: int = 300, lr: float = 1e-3
) -> dict:
    """Verification Plan #4: attack a deliberately overfit, no-DP,
    no-regularization model first. Must show AUC clearly > 0.7 -- if the
    attack can't beat 0.5 even here, the attack implementation is broken,
    not the privacy mechanism.

    First attempt used n_members=500 and a plain-sized MLP -- it trained to
    near-zero loss but only scored AUC=0.58. Root cause (confirmed by
    inspecting the loss distributions): with ~86% of this dataset being
    "easy" majority-class examples, most random non-members ALSO get
    near-zero loss from any reasonably-trained model, which swamps the
    memorization signal from a loss-threshold attack. This is a real,
    documented limitation of loss-threshold MIA on imbalanced/easy data,
    not specific to this attack's implementation. Fix: shrink the member
    set and grow the model (higher overparameterization ratio makes even
    the "easy" training points fit distinctly from same-pattern non-members)
    -- n_members=30 with a 256x128 MLP reliably gives AUC ~0.80 here.
    """
    X_df, y_s = load_features_targets()
    X = X_df.values.astype(np.float32)
    y = y_s.values.astype(np.float32)

    rng = np.random.default_rng(0)
    all_idx = np.arange(len(y))
    rng.shuffle(all_idx)
    member_idx = all_idx[:n_members]
    non_member_idx = all_idx[500 : 500 + n_non_members]

    scaler = StandardScaler().fit(X[member_idx])
    Xs = scaler.transform(X).astype(np.float32)

    member_train_loader = make_loader(Xs, y, member_idx, batch_size=8, shuffle=True)
    member_eval_loader = make_loader(Xs, y, member_idx, batch_size=256, shuffle=False)
    non_member_loader = make_loader(Xs, y, non_member_idx, batch_size=256, shuffle=False)

    set_seed(0)
    model = DiabetesMLP(input_dim=Xs.shape[1], hidden_dims=(256, 128), dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in member_train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    result = run_mia(model, member_eval_loader, non_member_loader, device)
    print(f"[sanity] overfit-model attack AUC={result['attack_auc']:.4f} (must be > 0.7)")
    assert result["attack_auc"] > 0.7, "MIA attack failed sanity check -- fix the attack before trusting other results"
    return result


def run_privacy_sweep(
    device: torch.device,
    best_mu: float,
    alpha: float = 0.5,
    epsilons=(1.0, 3.0, float("inf")),
    num_rounds: int = 10,
    local_epochs: int = 1,
    batch_size: int = 128,
    use_pos_weight: bool = False,
    per_client_threshold: bool = False,
    results_filename: str = "privacy_sweep_step7_8.json",
) -> dict:
    """Steps 7+8 combined.

    'local': each client has its own model -- members = that client's own
    train set, non-members = other clients' train sets + the global test set.

    'fedavg'/'fedprox': ONE shared model results (FedProx per Li et al. 2020
    doesn't keep separate per-client models -- see run_fl_sweep.py). Attacked
    per-client anyway (realistic threat model: "did this hospital's records
    train the shared model"): members = that client's own train set,
    non-members = the global test set only (every other client IS also a
    member of this shared model, so they can't serve as non-members here).
    """
    splits, client_loaders, input_dim = prepare_data(alpha=alpha, batch_size=batch_size)

    X_df, y_s = load_features_targets()
    X = X_df.values.astype(np.float32)
    y = y_s.values.astype(np.float32)
    train_pool_idx = np.concatenate(
        [c["train"] for c in splits["clients"]] + [c["val"] for c in splits["clients"]]
    )
    scaler = StandardScaler().fit(X[train_pool_idx])
    Xs = scaler.transform(X).astype(np.float32)

    global_test_loader = make_loader(Xs, y, splits["global_test"], batch_size, shuffle=False)
    client_train_idx = [c["train"] for c in splits["clients"]]

    results = {"local": [], "fedavg": [], "fedprox": []}

    for eps in epsilons:
        eps_key = "inf" if eps == float("inf") else eps
        print(f"\n{'=' * 20} epsilon={eps_key} {'=' * 20}")

        # ---- local: DP-SGD wrapped, one continuous run (no FL rounds) ----
        dp_configs_local = (
            client_dp_configs(client_loaders, eps, batch_size, local_epochs=1, num_rounds=num_rounds * local_epochs)
            if eps != float("inf")
            else None
        )
        per_client_metrics, models = run_fully_local(
            client_loaders,
            input_dim,
            device,
            epochs=num_rounds * local_epochs,
            dp_configs=dp_configs_local,
            return_models=True,
            use_pos_weight=use_pos_weight,
            per_client_threshold=per_client_threshold,
        )
        for i, model in enumerate(models):
            non_member_idx = np.concatenate(
                [client_train_idx[j] for j in range(len(client_loaders)) if j != i] + [splits["global_test"]]
            )
            non_member_loader = make_loader(Xs, y, non_member_idx, batch_size, shuffle=False)
            member_eval_loader = make_loader(Xs, y, client_train_idx[i], batch_size, shuffle=False)
            mia = run_mia(model, member_eval_loader, non_member_loader, device)
            acc = per_client_metrics[i]["accuracy"]
            auc = per_client_metrics[i]["auc"]
            print(f"[local] client {i} eps={eps_key}: acc={acc:.4f} auc={auc:.4f} attack_auc={mia['attack_auc']:.4f}")
            results["local"].append({"epsilon": eps_key, "client_id": i, "accuracy": acc, "auc": auc, **mia})

        # ---- fedavg / fedprox: one shared model ----
        for condition, mu in [("fedavg", 0.0), ("fedprox", best_mu)]:
            dp_configs = (
                client_dp_configs(client_loaders, eps, batch_size, local_epochs, num_rounds)
                if eps != float("inf")
                else None
            )
            _, per_client, final_model = run_fl_condition(
                client_loaders,
                input_dim,
                device,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                mu=mu,
                dp_configs=dp_configs,
                use_pos_weight=use_pos_weight,
                per_client_threshold=per_client_threshold,
            )
            final_model = final_model.to(device)
            mean_acc = float(np.mean([m["accuracy"] for m in per_client]))
            mean_auc = float(np.mean([m["auc"] for m in per_client]))

            per_client_attack_aucs = []
            for i in range(len(client_loaders)):
                member_loader = make_loader(Xs, y, client_train_idx[i], batch_size, shuffle=False)
                mia = run_mia(final_model, member_loader, global_test_loader, device)
                per_client_attack_aucs.append(mia["attack_auc"])

            mean_attack_auc = float(np.mean(per_client_attack_aucs))
            worst_attack_auc = float(np.max(per_client_attack_aucs))  # worst = highest leakage
            print(
                f"[{condition}] eps={eps_key}: mean_acc={mean_acc:.4f} mean_auc={mean_auc:.4f} "
                f"mean_attack_auc={mean_attack_auc:.4f} worst_attack_auc={worst_attack_auc:.4f}"
            )
            results[condition].append(
                {
                    "epsilon": eps_key,
                    "accuracy": mean_acc,
                    "auc": mean_auc,
                    "attack_auc": mean_attack_auc,
                    "worst_client_attack_auc": worst_attack_auc,
                    "per_client_attack_auc": per_client_attack_aucs,
                }
            )

    with open(RESULTS_DIR / results_filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / results_filename}")
    return results


def plot_privacy_tradeoff(results: dict, save_path: Path) -> None:
    """The signature result of the whole project: accuracy-vs-epsilon and
    Attack-AUC-vs-epsilon, same x-axis, one line per FL condition."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for condition in ["local", "fedavg", "fedprox"]:
        rows = results[condition]
        if condition == "local":
            by_eps: dict = {}
            for r in rows:
                by_eps.setdefault(r["epsilon"], []).append(r)
            eps_vals = sorted(by_eps.keys(), key=lambda e: (e == "inf", e))
            accs = [float(np.mean([r["accuracy"] for r in by_eps[e]])) for e in eps_vals]
            attack_aucs = [float(np.mean([r["attack_auc"] for r in by_eps[e]])) for e in eps_vals]
        else:
            rows_sorted = sorted(rows, key=lambda r: (r["epsilon"] == "inf", r["epsilon"]))
            eps_vals = [r["epsilon"] for r in rows_sorted]
            accs = [r["accuracy"] for r in rows_sorted]
            attack_aucs = [r["attack_auc"] for r in rows_sorted]

        x_labels = [str(e) for e in eps_vals]
        x = np.arange(len(x_labels))
        axes[0].plot(x, accs, marker="o", label=condition)
        axes[1].plot(x, attack_aucs, marker="o", label=condition)

    for ax, title, ylabel in [
        (axes[0], "Utility vs epsilon", "accuracy"),
        (axes[1], "Privacy leakage vs epsilon", "MIA attack AUC"),
    ]:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("epsilon (inf = no DP)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="ideal (0.5)")

    fig.suptitle("Utility-privacy tradeoff (reduced local grid -- full sweep planned for Colab)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    device = get_device()
    print(f"device: {device}")

    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n>>> Verification Plan #4: MIA sanity check on a deliberately overfit model")
    sanity_check_attack(device)

    fedprox_path = RESULTS_DIR / "fl_step6_fedprox.json"
    best_mu = json.loads(fedprox_path.read_text())["best_mu"] if fedprox_path.exists() else 0.001
    print(f"\nUsing FedProx mu={best_mu} (from Step 6)")

    results = run_privacy_sweep(device, best_mu=best_mu)
    plot_privacy_tradeoff(results, RESULTS_DIR / "privacy_tradeoff.png")
    print(f"Saved plot to {RESULTS_DIR / 'privacy_tradeoff.png'}")
    return results


def main_corrected():
    """Tier-1 priority item: re-run Steps 7-8 against corrected (pos_weight +
    per-client threshold) models -- the original main()/privacy_sweep_step7_8.json
    predate this correction (see PROJECT_GUIDE.md Sec 8.3/8.5) and are left
    untouched as the documented "before" baseline. sanity_check_attack() is
    skipped here -- it trains a synthetic overfit model unrelated to any
    client's pos_weight/threshold setting, so re-running it would just
    reproduce the same number for no reason.
    """
    device = get_device()
    print(f"device: {device}")

    RESULTS_DIR.mkdir(exist_ok=True)

    fedprox_corrected_path = RESULTS_DIR / "fl_step6c_fedprox_corrected.json"
    if fedprox_corrected_path.exists():
        best_mu = json.loads(fedprox_corrected_path.read_text())["best_mu"]
    else:
        fedprox_path = RESULTS_DIR / "fl_step6_fedprox.json"
        best_mu = json.loads(fedprox_path.read_text())["best_mu"] if fedprox_path.exists() else 0.001
    print(f"\nUsing corrected FedProx mu={best_mu}")

    results = run_privacy_sweep(
        device,
        best_mu=best_mu,
        use_pos_weight=True,
        per_client_threshold=True,
        results_filename="privacy_sweep_step7_8_corrected.json",
    )
    plot_privacy_tradeoff(results, RESULTS_DIR / "privacy_tradeoff_corrected.png")
    print(f"Saved plot to {RESULTS_DIR / 'privacy_tradeoff_corrected.png'}")
    return results


if __name__ == "__main__":
    main()
    main_corrected()
