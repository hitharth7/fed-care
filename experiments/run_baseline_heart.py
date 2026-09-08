"""Centralized baseline for the second specialty: UCI Heart Disease (Cleveland).

Same DiabetesMLP architecture, same train_baseline() loop as run_baseline.py --
only the data loader and a smaller batch size (297 rows total vs 253k for
diabetes) differ. Proves the pipeline is genuinely dataset-agnostic rather
than tuned to one dataset's shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders import load_heart_features_targets
from experiments.run_baseline import train_baseline

if __name__ == "__main__":
    train_baseline(
        epochs=40,  # more epochs than diabetes -- 297 rows means far fewer gradient steps/epoch
        batch_size=32,
        feature_loader=load_heart_features_targets,
        results_filename="baseline_metrics_heart.json",
    )
