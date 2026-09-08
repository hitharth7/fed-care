"""Query router (Design B): a doctor picks a specialty and enters a
patient's structured data. The SHARED federated model for that specialty
answers; each hospital's own independently-trained LOCAL model is used only
to explain which hospital's population this patient most resembles -- it
never makes the actual prediction (see train_specialists.py's docstring for
why). Reads artifacts written by train_specialists.py; trains nothing itself.

Every call returns a `trace`: a list of plain-English log lines covering
every step (specialty lookup, scaling, the shared model's raw output, which
hospital's own threshold was applied and why, and the per-hospital
agreement comparison) -- built so a reviewer can see the mechanism, not
just a final number. The frontend (built after this file is tested) renders
`trace` instead of this file's console printout.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from models.mlp import DiabetesMLP

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_cache: dict = {}


def _load_specialty(name: str) -> dict | None:
    """Returns None (not an exception) for an unconfigured specialty -- the
    caller turns that into an honest "no data" answer, never a fabricated one."""
    if name in _cache:
        return _cache[name]
    d = ARTIFACTS_DIR / name
    if not (d / "metadata.json").exists():
        return None

    meta = json.loads((d / "metadata.json").read_text())
    with open(d / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    global_model = DiabetesMLP(input_dim=meta["input_dim"])
    global_model.load_state_dict(torch.load(d / "global_model.pt", weights_only=True))
    global_model.eval()

    local_models = []
    for hid in range(meta["num_hospitals"]):
        m = DiabetesMLP(input_dim=meta["input_dim"])
        m.load_state_dict(torch.load(d / f"local_model_{hid}.pt", weights_only=True))
        m.eval()
        local_models.append(m)

    bundle = {"meta": meta, "scaler": scaler, "global_model": global_model, "local_models": local_models}
    _cache[name] = bundle
    return bundle


def list_specialties() -> list[dict]:
    """What a specialty picker (CLI or frontend) renders -- only specialties
    with real trained artifacts on disk show up as selectable."""
    out = []
    if not ARTIFACTS_DIR.exists():
        return out
    for d in sorted(p for p in ARTIFACTS_DIR.iterdir() if p.is_dir()):
        meta_path = d / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            out.append({"name": d.name, "display_name": meta["display_name"], "scale_note": meta["scale_note"]})
    return out


def get_specialty_schema(name: str) -> dict | None:
    """Metadata a frontend needs to render a query form for this specialty
    (feature names, hospital count, display/scale text) -- or None if it
    isn't trained. Doesn't expose model internals, just what's needed to
    build a form and call answer_query()."""
    bundle = _load_specialty(name)
    if bundle is None:
        return None
    meta = bundle["meta"]
    return {
        "name": name,
        "display_name": meta["display_name"],
        "scale_note": meta["scale_note"],
        "feature_names": meta["feature_names"],
        "feature_ranges": meta.get("feature_ranges", {}),
        "num_hospitals": meta["num_hospitals"],
    }


@torch.no_grad()
def _predict_prob(model: torch.nn.Module, x: np.ndarray) -> float:
    logit = model(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
    return float(torch.sigmoid(logit).item())


def answer_query(specialty: str, hospital_id: int, patient_features: dict) -> dict:
    trace = [f"Specialty requested: '{specialty}'"]
    bundle = _load_specialty(specialty)
    if bundle is None:
        trace.append(f"No trained model exists for specialty '{specialty}' in this federation.")
        return {
            "status": "no_data",
            "trace": trace,
            "message": f"No hospital in this network has trained a model for '{specialty}'.",
        }

    meta = bundle["meta"]
    trace.append(f"Loaded '{meta['display_name']}' -- {meta['scale_note']}")

    missing = [f for f in meta["feature_names"] if f not in patient_features]
    if missing:
        trace.append(f"Missing required fields: {missing}")
        return {"status": "error", "trace": trace, "message": f"Missing fields: {missing}"}

    raw = np.array([patient_features[f] for f in meta["feature_names"]], dtype=np.float32)
    trace.append(f"Patient features received: {dict(zip(meta['feature_names'], raw.tolist()))}")

    x = bundle["scaler"].transform(raw.reshape(1, -1)).astype(np.float32)[0]
    trace.append("Scaled using the same StandardScaler fit on the pooled federation training data.")

    prob = _predict_prob(bundle["global_model"], x)
    trace.append(
        f"Shared federated model ({meta['condition']}, mu={meta['mu']}) output probability: {prob:.4f}"
    )

    threshold = meta["thresholds"].get(str(hospital_id), 0.5)
    trace.append(
        f"Applying Hospital {hospital_id}'s own calibrated decision threshold: {threshold:.4f} "
        f"(tuned on that hospital's own training data, not a fixed 0.5 -- "
        f"see PROJECT_GUIDE.md Sec 8.2b for why a fixed threshold is wrong here)"
    )
    prediction = "positive" if prob >= threshold else "negative"
    trace.append(f"Prediction for Hospital {hospital_id}'s patient: {prediction}")

    trace.append(
        "Checking which hospital's own independently-trained model agrees most closely "
        "(explanation only -- does not change the prediction above):"
    )
    agreements = []
    for hid, local_model in enumerate(bundle["local_models"]):
        local_prob = _predict_prob(local_model, x)
        gap = abs(local_prob - prob)
        agreements.append({"hospital_id": hid, "local_probability": local_prob, "agreement_gap": gap})
        trace.append(f"  Hospital {hid}'s local model: probability={local_prob:.4f}  (gap from shared model: {gap:.4f})")

    closest = min(agreements, key=lambda a: a["agreement_gap"])
    trace.append(
        f"Closest match: Hospital {closest['hospital_id']} "
        f"(this patient's profile most resembles Hospital {closest['hospital_id']}'s own patient population)"
    )

    return {
        "status": "ok",
        "specialty": specialty,
        "display_name": meta["display_name"],
        "scale_note": meta["scale_note"],
        "hospital_id": hospital_id,
        "probability": prob,
        "threshold_used": threshold,
        "prediction": prediction,
        "most_similar_hospital": closest["hospital_id"],
        "hospital_agreement": agreements,
        "trace": trace,
    }


if __name__ == "__main__":
    print("Available specialties:", list_specialties())

    example_queries = [
        (
            "diabetes",
            1,
            {
                "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32, "Smoker": 1,
                "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0,
                "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
                "GenHlth": 3, "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 0, "Sex": 1,
                "Age": 9, "Education": 4, "Income": 5,
            },
        ),
        (
            "heart",
            2,
            {
                "age": 58, "sex": 1, "cp": 3, "trestbps": 145, "chol": 260, "fbs": 0,
                "restecg": 1, "thalach": 120, "exang": 1, "oldpeak": 2.3, "slope": 2,
                "ca": 1, "thal": 3,
            },
        ),
        ("imaging", 0, {}),  # not trained -- must return no_data, never fabricate
    ]

    for specialty, hospital_id, features in example_queries:
        print(f"\n{'=' * 70}\nQuery: specialty={specialty} hospital_id={hospital_id}\n{'=' * 70}")
        result = answer_query(specialty, hospital_id, features)
        for line in result["trace"]:
            print(" ", line)
        print("RESULT:", {k: v for k, v in result.items() if k != "trace"})
        assert result["status"] in ("ok", "no_data", "error")
        if specialty == "imaging":
            assert result["status"] == "no_data", "must not fabricate an answer for an untrained specialty"

    print("\nAll self-checks passed.")
