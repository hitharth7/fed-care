"""Train and persist the models the query router serves at inference time.

Run this once (or whenever the underlying training code changes) -- the
router itself only ever LOADS these artifacts, it never trains live on a
query. Writes to router/artifacts/<specialty>/, which stays untracked (like
every other trained-weight file in this repo, see .gitignore's *.pt/*.ckpt)
since it's fully deterministic given SEED=42 and regenerating is simpler
than a stale-cache risk.

Design B (a conversation-driven decision, not in the original plan docs):
the router answers with the SHARED federated model -- the best-tested
condition per specialty -- and separately keeps each hospital's
independently-trained LOCAL model only to explain "which hospital's
population does this patient most resemble". A local model never makes the
actual call: local models are structurally the least accurate of the three
conditions tested for diabetes (paper/report.md Sec 4.2c), so routing the
real prediction through them would trade accuracy for narrative.

Heart disease uses plain FedAvg, not the mu-grid "winner" from
run_fl_sweep_heart.py -- that grid's margin between conditions (0.7833 vs
0.7833, an exact tie) is noise from a 297-row dataset, not a real personalization
signal, so claiming a "tuned" FedProx result there would overstate what was
actually found.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.loaders import load_features_targets, load_heart_features_targets
from experiments.run_fl_sweep import get_device, prepare_data, run_fedavg, run_fl_condition, run_fully_local

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

SPECIALTIES = {
    "diabetes": {
        "display_name": "Diabetes Risk",
        "feature_loader": load_features_targets,
        "condition": "fedprox",
        "mu": 0.1,  # the corrected mu-grid winner -- see fl_step6c_fedprox_corrected.json
        "num_rounds": 10,
        "batch_size": 128,
        "local_epochs": 10,
        "scale_note": "253,680 patients across 5 simulated hospitals -- full-scale result.",
    },
    "heart": {
        "display_name": "Heart Disease Risk",
        "feature_loader": load_heart_features_targets,
        "condition": "fedavg",
        "mu": 0.0,
        "num_rounds": 15,
        "batch_size": 16,
        "local_epochs": 15,
        "scale_note": "297 patients (UCI Cleveland) across 5 simulated hospitals -- "
        "demonstration scale, not a validated clinical result.",
    },
}


# Plain-English labels for coded fields -- these are survey/clinical codes
# (BRFSS for diabetes, UCI Cleveland for heart), not free-form numbers, so a
# raw "Age=9" or "cp=3" is meaningless without the codebook. Only fields with
# a genuinely small, well-documented set of codes get a dropdown; continuous
# fields (BMI, blood pressure, cholesterol, etc.) stay as numeric inputs.
YES_NO = {0: "No", 1: "Yes"}
FIELD_OPTIONS = {
    # diabetes (CDC BRFSS)
    "HighBP": YES_NO, "HighChol": YES_NO, "CholCheck": YES_NO, "Smoker": YES_NO,
    "Stroke": YES_NO, "HeartDiseaseorAttack": YES_NO, "PhysActivity": YES_NO,
    "Fruits": YES_NO, "Veggies": YES_NO, "HvyAlcoholConsump": YES_NO,
    "AnyHealthcare": YES_NO, "NoDocbcCost": YES_NO, "DiffWalk": YES_NO,
    "Sex": {0: "Female", 1: "Male"},
    "GenHlth": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "Age": {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
        7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+",
    },
    "Education": {
        1: "Never attended / kindergarten only", 2: "Grades 1-8", 3: "Grades 9-11",
        4: "Grade 12 / GED", 5: "Some college (1-3 yrs)", 6: "College graduate (4+ yrs)",
    },
    "Income": {
        1: "< $10k", 2: "$10k-15k", 3: "$15k-20k", 4: "$20k-25k", 5: "$25k-35k",
        6: "$35k-50k", 7: "$50k-75k", 8: "$75k+",
    },
    # heart (UCI Cleveland)
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No (<=120 mg/dl)", 1: "Yes (>120 mg/dl)"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
    "exang": YES_NO,
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}


def _feature_ranges(feature_loader) -> dict:
    """Real min/max/step/default per field, straight from the actual data --
    these are survey/clinical codes (0/1 flags, 1-5 scales, category codes),
    not free-form numbers, so the frontend form needs real bounds rather
    than accepting anything a number input allows."""
    X, _ = feature_loader()
    out = {}
    for col in X.columns:
        vals = X[col]
        is_int = (vals % 1 == 0).all()
        out[col] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "step": 1 if is_int else 0.1,
            "default": int(vals.median()) if is_int else float(vals.median()),
        }
        if col in FIELD_OPTIONS:
            out[col]["options"] = [{"value": v, "label": lbl} for v, lbl in FIELD_OPTIONS[col].items()]
    return out


def train_specialty(name: str, cfg: dict, alpha: float = 0.5) -> None:
    print(f"\n{'=' * 60}\nTraining specialty: {name} ({cfg['display_name']})\n{'=' * 60}")
    device = get_device()

    splits, client_loaders, input_dim, scaler, feature_names = prepare_data(
        alpha=alpha,
        batch_size=cfg["batch_size"],
        feature_loader=cfg["feature_loader"],
        return_scaler=True,
    )
    n_per_hospital = [len(tl.dataset) for tl, _ in client_loaders]
    print(f"hospitals: {n_per_hospital} train rows each")

    # --- shared federated model: this is what actually answers a query ---
    if cfg["condition"] == "fedprox":
        _, per_client, global_model = run_fl_condition(
            client_loaders,
            input_dim,
            device,
            num_rounds=cfg["num_rounds"],
            mu=cfg["mu"],
            use_pos_weight=True,
            per_client_threshold=True,
        )
    else:
        _, per_client, global_model = run_fedavg(
            client_loaders,
            input_dim,
            device,
            num_rounds=cfg["num_rounds"],
            use_pos_weight=True,
            per_client_threshold=True,
        )
    thresholds = {str(int(m["client_id"])): float(m["threshold"]) for m in per_client}
    print(f"per-hospital thresholds (federated model): {thresholds}")

    # --- each hospital's own independently-trained model: explanation only ---
    _, local_models = run_fully_local(
        client_loaders,
        input_dim,
        device,
        epochs=cfg["local_epochs"],
        use_pos_weight=True,
        per_client_threshold=True,
        return_models=True,
    )

    out_dir = ARTIFACTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(global_model.state_dict(), out_dir / "global_model.pt")
    for hid, model in enumerate(local_models):
        torch.save(model.state_dict(), out_dir / f"local_model_{hid}.pt")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "display_name": cfg["display_name"],
                "feature_names": feature_names,
                "input_dim": input_dim,
                "num_hospitals": len(client_loaders),
                "hospital_train_sizes": n_per_hospital,
                "thresholds": thresholds,
                "condition": cfg["condition"],
                "mu": cfg["mu"],
                "scale_note": cfg["scale_note"],
                "feature_ranges": _feature_ranges(cfg["feature_loader"]),
            },
            f,
            indent=2,
        )
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    for name, cfg in SPECIALTIES.items():
        train_specialty(name, cfg)
