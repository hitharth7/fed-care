"""FL sweep for the second specialty: UCI Heart Disease (Cleveland).

Unlike run_fl_sweep.py, this goes straight to the corrected setup
(pos_weight + per-client threshold) -- there is no need to reproduce the
uncorrected/broken-threshold result a second time, we already know why it's
wrong (see PROJECT_GUIDE.md Sec 8.2b) and diagnosing it again here would
just spend time re-confirming a mechanism, not learning anything new.

Scale warning: 297 rows total across 5 simulated hospitals (Dirichlet
alpha=0.5) means per-client shards can be as small as a few dozen rows --
enough to prove the pipeline is genuinely dataset-agnostic and to give the
router something real to serve, but these numbers should be read as
directional, not as rigorous as the diabetes results (253k rows). Kept at
5 hospitals (not fewer) so hospital IDs mean the same physical hospital
across both specialties for the router.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders import load_heart_features_targets
from experiments.run_fl_sweep import (
    RESULTS_DIR,
    get_device,
    prepare_data,
    run_fedavg,
    run_fl_condition,
    run_fully_local,
    summarize,
)

ALPHA = 0.5
BATCH_SIZE = 16  # small dataset -- 128 (the diabetes default) would exceed most shards
NUM_ROUNDS = 15  # a few more than diabetes's 10 -- fewer, smaller batches per round
MUS = (0.001, 0.01, 0.1, 1.0)


def main():
    device = get_device()
    print(f"device: {device}")
    RESULTS_DIR.mkdir(exist_ok=True)

    _, client_loaders, input_dim = prepare_data(
        alpha=ALPHA, batch_size=BATCH_SIZE, feature_loader=load_heart_features_targets
    )
    print("client shard sizes:", [len(tl.dataset) for tl, _ in client_loaders])

    print("\n>>> Local (corrected)")
    local_metrics = run_fully_local(
        client_loaders, input_dim, device, epochs=NUM_ROUNDS,
        use_pos_weight=True, per_client_threshold=True,
    )
    local_summary = summarize("heart_local_corrected", local_metrics)

    print("\n>>> FedAvg (corrected)")
    _, fedavg_metrics, _ = run_fedavg(
        client_loaders, input_dim, device, num_rounds=NUM_ROUNDS,
        use_pos_weight=True, per_client_threshold=True,
    )
    fedavg_summary = summarize("heart_fedavg_corrected", fedavg_metrics)

    grid_results = {}
    for mu in MUS:
        print(f"\n>>> FedProx (corrected) mu={mu}")
        _, metrics, _ = run_fl_condition(
            client_loaders, input_dim, device, num_rounds=NUM_ROUNDS, mu=mu,
            use_pos_weight=True, per_client_threshold=True,
        )
        grid_results[str(mu)] = summarize(f"heart_fedprox_corrected_mu{mu}", metrics)
    best_mu = max(grid_results, key=lambda k: grid_results[k]["mean_balanced_accuracy"])
    print(f"\nBest mu by mean balanced accuracy: {best_mu}")

    conditions = {
        "local_corrected": local_summary,
        "fedavg_corrected": fedavg_summary,
        "fedprox_corrected_best": grid_results[best_mu],
        "best_mu": float(best_mu),
        "mu_grid": grid_results,
    }

    print(f"\n{'condition':>24} {'mean_bal_acc':>13}")
    print("-" * 40)
    for name in ("local_corrected", "fedavg_corrected", "fedprox_corrected_best"):
        print(f"{name:>24} {conditions[name]['mean_balanced_accuracy']:>13.4f}")

    with open(RESULTS_DIR / "fl_heart_corrected.json", "w") as f:
        json.dump(conditions, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'fl_heart_corrected.json'}")
    return conditions


if __name__ == "__main__":
    main()
