"""One-time setup for the live networked demo.

Materializes each simulated hospital's own local data file on disk, plus a
small shared feature-scaling config, so that demo/run_hospital.py can start
a hospital process that ONLY EVER reads its own file at demo time -- no
hospital process touches another hospital's data once the demo starts. This
mirrors how a real deployment would provision each hospital's local
database once, ahead of training (the one-time pooled pass here to compute
the Dirichlet partition and scaler is the same documented simplification
used throughout this project -- see PROJECT_GUIDE.md -- done here as an
offline setup step rather than something any hospital process does live).

Run this once before the live demo:
    python demo/prepare_demo_data.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler

from data.loaders import load_features_targets
from data.partition import build_experiment_splits

NUM_HOSPITALS = 5
ALPHA = 0.5
SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "hospital_data"


def main():
    X_df, y_s = load_features_targets()
    X = X_df.values.astype(np.float32)
    y = y_s.values.astype(np.float32)

    splits = build_experiment_splits(y_s, NUM_HOSPITALS, ALPHA, seed=SEED)

    train_pool_idx = np.concatenate(
        [c["train"] for c in splits["clients"]] + [c["val"] for c in splits["clients"]]
    )
    scaler = StandardScaler().fit(X[train_pool_idx])
    Xs = scaler.transform(X).astype(np.float32)

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "scaler_stats.json", "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

    print(f"Materializing {NUM_HOSPITALS} separate hospital data files (alpha={ALPHA}, seed={SEED})...\n")
    for i, c in enumerate(splits["clients"]):
        np.savez(OUT_DIR / f"hospital_{i}_train.npz", X=Xs[c["train"]], y=y[c["train"]])
        np.savez(OUT_DIR / f"hospital_{i}_val.npz", X=Xs[c["val"]], y=y[c["val"]])
        print(f"  Hospital {i}: {len(c['train'])} train records, {len(c['val'])} val records -> hospital_{i}_{{train,val}}.npz")

    print(f"\nDone. Files written to {OUT_DIR}/")
    print("Each hospital's run_hospital.py process will only ever read its own file from here on.")


if __name__ == "__main__":
    main()
