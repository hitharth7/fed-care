# FedCare — Project Guide

This is the technical reference for the codebase: what every file does, how the pieces fit together, how to run it, and what's been found so far. For the *planning* documents (scope decisions, literature review, week-by-week roadmap) see [`fedcare_project_plan.md`](fedcare_project_plan.md) and [`fedcare_master_project_guide.md`](fedcare_master_project_guide.md). For the *write-up* (IEEE short-paper structure, results, discussion) see [`paper/report.md`](paper/report.md). This guide sits between the two: it explains the *implementation*.

---

## 1. What FedCare is, in one page

FedCare simulates five hospitals collaboratively training a diabetes-diagnostic model without sharing raw patient data, via **Flower**-based federated learning (FL). Three things are layered on top of plain FedAvg, in this order:

1. **Non-IID simulation** — hospitals don't have identical patient populations. Simulated via Dirichlet partitioning over the CDC Diabetes Health Indicators dataset.
2. **FedProx personalization** — a proximal term in each client's local loss, meant to let each hospital's model drift toward its own population without losing the benefit of collaboration.
3. **Differential privacy (Opacus DP-SGD)** — formal noise-based protection against a client's updates leaking patient-level information.

The project's distinguishing move (vs. a standard FL tutorial): it doesn't stop at reporting a theoretical DP epsilon. It **attacks its own trained models** with a membership-inference attack (MIA) to empirically measure how much actually leaks, and pairs that against the accuracy cost, for every FL condition.

**Core research question**: given that hospitals legitimately have different patient populations, can FedProx recover the accuracy lost to DP, without reopening the privacy leak DP is meant to close?

**Answer found so far** (see §8): not quite what was expected — FedProx turns out *not* to recover the worst-served hospital's accuracy at all (a verified negative result), and separately, federated pooling itself (independent of DP) turns out to be the dominant privacy mechanism in this setup. Both are real, defensible findings — see `paper/report.md` for the full discussion.

---

## 2. Repo structure

```
fedcare/
  data/
    loaders.py         # download + cache CDC Diabetes Health Indicators (UCI id=891)
    partition.py        # Dirichlet non-IID partitioner + experiment split builder
    raw/                 # gitignored — cached dataset CSV lands here on first run
  models/
    mlp.py               # DiabetesMLP: the one model used everywhere (no BatchNorm)
  fl/
    client.py             # DiabetesFlowerClient: FedAvg/FedProx/+DP, all in one class
    strategies.py          # FedAvg server-side aggregation strategy + result-capturing hooks
  privacy/
    dp.py                  # Opacus wrapper with correct multi-round epsilon accounting
    mia.py                  # Yeom et al. loss-threshold membership-inference attack
  experiments/
    run_baseline.py          # Step 4: centralized baseline
    run_fl_sweep.py           # Steps 5-6: local vs FedAvg vs FedProx
    run_mia_audit.py           # Steps 7-8: DP epsilon sweep + MIA audit
  notebooks/
    01_eda.ipynb                # dataset EDA + Dirichlet partition visualization
  demo/
    prepare_demo_data.py         # one-time: materializes each hospital's own data file on disk
    run_server.py                 # real networked FL server + live dashboard web server (for live panel demos)
    run_hospital.py                # real networked FL client, one process per hospital
    dashboard/
      index.html                    # the live visual dashboard (open in a browser)
    README.md                       # how to run the live demo + what to say about it
  results/                       # all metrics (.json) and figures (.png) land here
  paper/
    report.md                     # the write-up — IEEE short-paper structure, real numbers
  fedcare_project_plan.md           # original scope/architecture/roadmap doc
  fedcare_master_project_guide.md    # literature review + extended-scope doc
  PROJECT_GUIDE.md                    # this file
  requirements.txt
  .gitignore
  README.md
```

Every `experiments/*.py` script is runnable standalone (`python experiments/run_baseline.py`) and writes its outputs to `results/`. Nothing in `data/`, `models/`, `fl/`, or `privacy/` is meant to be run directly — they're imported by the experiment scripts. `demo/` is a separate presentation layer (see §5.11) — it doesn't feed into any research result.

---

## 3. Setup

```bash
cd fedcare
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

That's it — no API keys, no accounts needed for anything in this repo currently. The dataset downloads automatically from UCI on first run (see §7 gotcha about SSL certs if that fails on macOS).

**Versions this was built and tested against**: Python 3.13, torch 2.14, flwr 1.36.0 (pinned — see §7), opacus 1.6.0, ray 2.55.1 (pulled in via `flwr[simulation]`).

---

## 4. How to run it

Run in this order — each script depends on the previous one's output being in `results/`:

```bash
source .venv/bin/activate

python experiments/run_baseline.py    # ~1-2 min.  Writes results/baseline_metrics.json
python experiments/run_fl_sweep.py    # ~2-3 min.  Writes fl_step5_*.json, fl_step6_*.json + a figure
python experiments/run_mia_audit.py   # ~5-10 min. Writes privacy_sweep_step7_8.json + a figure
```

`run_fl_sweep.py` reads `results/baseline_metrics.json` (to print the sanity-check gap) but doesn't hard-fail if it's missing. `run_mia_audit.py` reads `results/fl_step6_fedprox.json` to pick up the best FedProx mu found in Step 6 (falls back to `mu=0.001` if that file doesn't exist yet).

Runtimes above are for a 2024 MacBook (Apple MPS backend, no CUDA). On Colab with a T4 GPU, the FL/DP scripts should be faster; the *reduced* grids used in `run_mia_audit.py` are specifically sized to be tractable locally (see §8.4) — expect to edit that script's grid before running the full-scale version.

### Interactive EDA

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb  # headless re-run
# or: jupyter lab notebooks/01_eda.ipynb                                    # interactive
```

---

## 5. Pipeline walkthrough (module by module)

### 5.1 `data/loaders.py`

`load_diabetes_data()` fetches the **CDC Diabetes Health Indicators** dataset (BRFSS 2015, UCI id **891**, 253,680 rows × 21 features + binary target `Diabetes_binary`) via the `ucimlrepo` package, and caches it to `data/raw/diabetes.csv` (gitignored — ~11MB) so it only downloads once. `load_features_targets()` returns `(X, y)` split.

**Gotcha baked in**: macOS's python.org Framework builds don't wire a CA bundle into `ssl` by default, which breaks the plain `urllib` fetch `ucimlrepo` uses internally. `_ensure_ssl_certs()` points `SSL_CERT_FILE` at `certifi`'s bundle before the first fetch, so this is transparent — you shouldn't hit it, but if you ever see `SSLCertVerificationError` here, that's the cause.

### 5.2 `data/partition.py`

Two functions:

- **`dirichlet_partition(y, num_clients, alpha, seed)`** — the core non-IID mechanism (Hsu et al., 2019). For each class label, shuffles that class's row indices and splits them across `num_clients` using proportions drawn from `Dir(alpha, ..., alpha)`. Low `alpha` → extreme skew (a client can end up ~100% one class); high `alpha` (e.g. 100) → near-IID, converging to the pooled dataset's own class balance.

- **`build_experiment_splits(y, num_clients, alpha, ...)`** — what every experiment script actually calls. Builds:
  - a **global held-out test set** (stratified, never touched by any client's training) — this doubles as the MIA "definitely non-member" pool in Step 8.
  - for each client, a **local train/val split** of its Dirichlet shard.

  Everything is deterministic given `(num_clients, alpha, seed)` — no caching to disk needed; re-running with the same arguments reproduces the identical split, which is why the MIA audit can regenerate "who trained on what" without needing saved index files.

**Verified in `notebooks/01_eda.ipynb`**: at `alpha=0.1`, some clients end up with as few as a handful of rows (an extreme stress case, matching the plan's own risk table). The project settled on **`alpha=0.5`** as the primary "moderate non-IID" condition for FL comparisons, and `alpha=100` as the "near-IID" sanity-check condition.

### 5.3 `models/mlp.py`

`DiabetesMLP`: 2 hidden layers (64, 32) with **LayerNorm** (not BatchNorm) + ReLU + Dropout(0.2), configurable via `hidden_dims`. **No BatchNorm anywhere in this codebase, by design** — Opacus's per-sample gradient computation doesn't support it, and this was decided in Step 4 (before any FL/DP code existed) specifically to avoid a rework later. Verified compatible via `opacus.validators.ModuleValidator.validate()` (returns no errors).

### 5.4 `fl/client.py`

`DiabetesFlowerClient` is the single class that implements **all three FL conditions plus optional DP**, composed via constructor arguments rather than one subclass per condition:

| Condition | How it's expressed |
|---|---|
| Vanilla FedAvg | `mu=0.0`, `dp_config=None` (defaults) |
| FedProx | `mu > 0.0` — adds `mu/2 * ||w - w_global||²` to the local loss, pulling the client toward the round's starting global params |
| + Differential Privacy | `dp_config` given — wraps that round's model/optimizer/loader with Opacus (`privacy/dp.py`) before training |

Any combination of `mu` and `dp_config` is valid — Step 7's DP sweep runs all three FL conditions with DP simultaneously using this same class. `get_parameters`/`set_parameters` do the ndarray ↔ state_dict conversion Flower's `NumPyClient` interface expects; `evaluate_model()` (shared with the non-Flower "fully local" training path in `run_fl_sweep.py`) computes accuracy/F1/AUC identically regardless of which path produced the model.

### 5.5 `fl/strategies.py`

`make_fedavg_strategy()` builds a `flwr.server.strategy.FedAvg` configured so all 5 clients participate every round (no subsampling — pointless with only 5 simulated hospitals). Two things it adds beyond Flower's defaults, both via closures over caller-supplied dicts:

- **`per_round_sink`** — captures the *raw* per-client `(num_examples, metrics)` list from the final round. Flower's own `History` object only keeps the *aggregated* (weighted-average) metric per round, but this project's key metric is **per-client accuracy, especially the worst-served client** — that requires the unaggregated breakdown.
- **`param_sink`** — captures the final global model's raw parameters via a no-op centralized `evaluate_fn` hook. Needed so the MIA audit (Step 8) can attack the actual trained model object, not just its reported metrics.

### 5.6 `privacy/dp.py`

Wraps Opacus `PrivacyEngine` with the one thing the project plan explicitly flags as *the* most common correctness bug in "FL + DP-SGD" projects: **privacy loss composes across every communication round a client participates in, not per-round in isolation.**

Because Flower's `client_fn` is stateless (a fresh client object can be built every round — see §5.4), you can't just keep one `PrivacyEngine` alive across rounds. The fix implemented here:

1. **Before training starts**, `make_dp_config()` calibrates **one** `noise_multiplier` for the client's *total* step budget across every round it will do (`opacus.accountants.utils.get_noise_multiplier`, given the target epsilon and the *full* multi-round step count).
2. **Every round**, `wrap_for_round()` re-wraps that round's fresh model/optimizer/loader with Opacus, but always using that *same* precalibrated `noise_multiplier` — so the noise level per step is consistent across the whole run, which is what the accounting math assumes.
3. **After all rounds**, `compute_final_epsilon()` reports the actual epsilon for the *entire* run from `(noise_multiplier, sample_rate, total_steps, delta)` — a deterministic function of the whole training run, computed once, not accumulated incorrectly per round.

Verified empirically: `GradSampleModule` (what Opacus wraps your model in) shares the same underlying parameter tensors as the original model — training in-place updates are visible on the original `model` object afterward, so no extra unwrapping step is needed to read out trained weights.

### 5.7 `privacy/mia.py`

The Yeom et al. (2018) **loss-threshold membership-inference attack**. For a trained model, computes per-sample BCE loss on a "member" pool (data the model trained on) and a "non-member" pool (data it never saw), then reports **Attack AUC** — ROC-AUC treating `-loss` as the membership score, swept across every threshold rather than fixing one. 0.5 = attacker no better than a coin flip (ideal privacy); 1.0 = perfect leakage.

Chosen over a full Shokri et al. (2017) multi-shadow-model attack as the primary method because it needs zero extra shadow-model training — much cheaper on free-tier compute (a lightweight shadow-model version remains a viable stretch add-on, not attempted here).

### 5.8 `experiments/run_baseline.py` — Step 4

Trains `DiabetesMLP` on the full pooled dataset (stratified 70/15/15 train/val/test), standardized features, BCE loss. This is the **accuracy ceiling** everything else is compared against. **Result: 86.6% test accuracy, AUC 0.83, F1 0.24** — matches published kernels for this dataset (~75-86% accuracy), confirming the pipeline is sound before any FL/DP complexity is introduced. F1 is low because of the dataset's ~86%/14% class imbalance, not a bug.

### 5.9 `experiments/run_fl_sweep.py` — Steps 5-6

The main FL comparison script. Key functions:

- **`prepare_data(alpha, ...)`** — builds the Dirichlet split, fits a `StandardScaler` on the pooled training pool (a deliberate simplification — a real deployment would need federated feature statistics, out of scope here), returns per-client `DataLoader` pairs.
- **`run_fully_local(...)`** — each client trains alone, no communication (the "lower bound" condition). Optionally DP-wrapped; optionally returns the trained model objects (`return_models=True`, used by the MIA audit).
- **`run_fl_condition(..., mu=0.0, dp_configs=None)`** — the generic FedAvg/FedProx(+DP) runner via Flower's `start_simulation`. Returns `(history, final_per_client_metrics, final_global_model)`.
- **`summarize(...)`** — tracks **both accuracy and AUC** per condition (see §8.2 for why accuracy alone is misleading here) and reports the worst client by each.
- **`run_step6_fedprox(...)`** — grid-searches `mu ∈ {0.001, 0.01, 0.1, 1.0}`, selects the best by **mean AUC** (not worst-client accuracy — see §8.2), and produces the two-panel comparison figure.

Running `main()` does two things in sequence: (1) the near-IID FedAvg sanity check against the centralized baseline, then (2) the `alpha=0.5` local-vs-FedAvg comparison, followed by the FedProx mu grid search.

### 5.10 `experiments/run_mia_audit.py` — Steps 7-8

Combines the DP epsilon sweep and the MIA audit into one pass per (condition, epsilon) — trains once, then immediately attacks the resulting model, avoiding a separate retraining step.

- **`sanity_check_attack(...)`** — Verification Plan item #4: attack a deliberately overfit, no-DP, no-regularization model first; must show AUC > 0.7 or the attack itself is presumed broken. Uses a small member set (30 samples) and an intentionally over-large model (256×128 hidden layers) — see the function's docstring for why a naive first attempt (500 members, normal-sized model) only reached AUC 0.58 despite near-perfect memorization, and why that's a real, documented property of loss-threshold attacks on imbalanced data rather than an attack bug.
- **`run_privacy_sweep(...)`** — for each epsilon in the grid, trains all three FL conditions (with DP where `epsilon != inf`) and runs the MIA attack against each. For `local`, each client has its own model (members = own train set, non-members = other clients' train sets + global test set). For `fedavg`/`fedprox`, there's one shared model (members = that client's own train set, non-members = the global test set only — every other client is *also* a member of this shared model, so can't serve as a non-member).
- **`plot_privacy_tradeoff(...)`** — the paired accuracy-vs-epsilon / attack-AUC-vs-epsilon figure, one line per FL condition.

### 5.11 `demo/` — real networked demo, for presenting to a panel

Everything above runs in Flower's **simulation mode**: all 5 hospitals execute as virtual clients inside one Python process (or, more precisely, one Ray-managed process pool — see §6). That's correct and standard for producing research results, but there's nothing to visibly point at in a live demo.

`demo/` is a separate presentation layer using the *exact same* model and client code (`models/mlp.py`, `fl/client.py`), but deployed as real, separate OS processes communicating over an actual gRPC network socket — the same transport Flower uses in production. It does not feed into any research result in `paper/report.md`; it exists purely so a panel can watch federated learning happen rather than take it on faith.

- **`prepare_demo_data.py`** — one-time setup. Computes the same Dirichlet partition as the research pipeline (`alpha=0.5`, 5 hospitals, `seed=42`) and **materializes each hospital's data as its own separate `.npz` file** on disk (`demo/hospital_data/hospital_{0..4}_{train,val}.npz`) — a genuinely separate, inspectable artifact per hospital, not just an in-memory slice. This mirrors how a real deployment would provision each hospital's local database once, ahead of training.
- **`run_server.py`** — a real Flower server (`flwr.server.start_server`) with narrated per-round logging of every hospital's reported accuracy before/after aggregation. Also starts a small local HTTP server (`http.server`, stdlib, default port 8090, background thread) serving `dashboard/index.html` and a `dashboard_state.json` it rewrites atomically after every fit/evaluate aggregation round — this is the *only* connection between the server and the dashboard; the dashboard is a read-only mirror of real strategy callbacks, not a separate data path.
- **`run_hospital.py`** — a real Flower client (`flwr.client.start_client`) that loads *only* its own `.npz` file (never another hospital's), trains via the same `DiabetesFlowerClient`, and prints what it's doing at each step. Run once per hospital, in its own process/terminal.
- **`dashboard/index.html`** — a single self-contained page (inline CSS/JS, Chart.js from CDN for the accuracy-over-rounds chart) polling `dashboard_state.json` every 800ms: a central "Global Model" node with 5 hospital nodes around it, connecting lines that light up when a hospital sends weights, per-hospital accuracy color-coded, and a live activity log. Verified end-to-end with a real running server + 5 real hospital processes (not just the JSON, the *rendered page* too, via headless Chrome screenshots and DOM dumps against the live server). This caught two real rendering bugs worth knowing about if you touch this file: `backdrop-filter: blur()` on the card backgrounds rendered as fully invisible in headless Chrome (fixed by using an opaque-enough base color instead of relying on blur compositing for visibility), and a `resize` event listener that wiped and recreated all hospital cards was firing on spurious resize events and could leave the visualization empty (removed — `ensureCards()` already recomputes layout every poll tick, so it was unnecessary). See `demo/dashboard_preview.png` for a real screenshot from an actual run.

See `demo/README.md` for exact run instructions, multi-machine setup, troubleshooting, and suggested talking points. Verified end-to-end: all 5 hospitals connecting, training, and reporting accuracy that matches Step 5's simulation-mode results exactly (client 1: 34.7%, client 3: 97.7%, etc. — same seed, same partition, different transport).

---

## 6. Design decisions and why (quick reference)

| Decision | Reason |
|---|---|
| No BatchNorm anywhere | Opacus incompatibility — decided in Step 4, before FL/DP code existed |
| Flower `start_simulation` for research (not the newer `ServerApp`/`flwr run`) | Simpler `NumPyClient` API, easier to reason about for a team new to PyTorch. Pinned `flwr==1.36.0` in `requirements.txt` since this API is deprecated and could be removed in a future Flower release |
| `demo/` uses `start_server`/`start_client` (also deprecated, also kept) | Same simplicity reasoning as above, applied to the real-networking side — the newer `flower-superlink`/`flower-supernode` CLI deployment model is more production-grade but heavier to set up for a one-off panel demo |
| One `DiabetesFlowerClient` class for all conditions | FedProx and DP both compose as optional constructor args rather than needing 4+ subclasses (plain / FedProx / DP / FedProx+DP) |
| Feature scaling fit on pooled training pool, not per-client | A real deployment would need federated feature statistics (a separate, solved problem) — out of scope for this project |
| `build_experiment_splits()` never persists indices to disk | Deterministic given `(num_clients, alpha, seed)` — regenerating is simpler and less error-prone than a stale-cache risk |
| μ selected by mean AUC, not worst-client accuracy | Accuracy is saturated/uninformative at `alpha=0.5` — see §8.2 |
| Reduced epsilon grid locally (`{1, 3, inf}`, 1 seed) | The plan's full grid (`{1,3,5,10,inf} × 3 conditions × 3-5 seeds`) is explicitly sized for Colab GPU, not a laptop — this local run validates correctness, not final numbers |

---

## 7. Known gotchas

- **SSL cert error on first dataset download (macOS)**: already handled in `data/loaders.py` (see §5.1) — you shouldn't hit this, but if you do, it's the python.org Framework build's missing CA bundle, not a network issue.
- **`flwr.simulation.start_simulation` prints a loud deprecation warning** every run. Expected — see the "Design decisions" table above. It still works correctly on the pinned version.
- **`pip install flwr[simulation]` is required**, not just `flwr` — the bare package raises `ImportError` on `ray` when you actually try to run a simulation. Already in `requirements.txt`.
- **Ray/MPS interaction**: multiple concurrent Ray actors *can* look like they're producing suspiciously-identical results — if you ever see a parameter sweep where every value in the sweep gives bit-identical results, don't assume it's a Ray caching bug before checking whether the underlying *metric* (e.g., thresholded accuracy) is just saturated. This exact pattern happened during Step 6 development (see `paper/report.md` §4.2) and took real debugging effort to distinguish from an actual bug — AUC moved correctly the entire time; only accuracy was pinned.

---

## 8. Findings so far

### 8.1 Baseline (Step 4)

86.6% test accuracy, AUC 0.83, F1 0.24 — matches published kernels, confirms the pipeline before FL/DP complexity.

### 8.2 Local vs FedAvg vs FedProx (Steps 5-6)

Sanity check passed first: FedAvg at `alpha=100` (near-IID) reaches within ~0.002 of the centralized baseline — confirms the FL loop itself before non-IID/privacy is introduced.

At `alpha=0.5` (moderate non-IID):

| Condition | Mean accuracy | Client 1's accuracy (worst) |
|---|---|---|
| Fully local | 91.3% | **78.9%** (F1 0.85 — genuinely learned its own distribution) |
| Vanilla FedAvg | 82.4% | **34.7%** (F1 0.0 — collapsed to majority-class prediction) |
| FedProx (any μ tested, incl. wider follow-up check) | 82.4% | **34.7%** (unchanged) |

**FedProx does not recover client 1's accuracy at any μ ∈ {0.001, 0.01, 0.1, 1.0, 10.0} or at local_epochs ∈ {1, 5}** — verified as a real, structural finding (μ does change the underlying model — AUC shifts measurably — but not enough to flip any prediction across the 0.5 threshold for this client). See `paper/report.md` §4.2 for the full investigation.

### 8.3 Privacy audit (Steps 7-8)

MIA sanity check passes (AUC 0.80 against a deliberately overfit model). Two findings:

1. **Fully local models leak measurably more than the pooled FedAvg/FedProx model, even without any DP.** Local per-client attack AUC: 0.52-0.63 (mean ~0.59). FedAvg/FedProx shared model: attack AUC ~0.485 (statistically ≈0.5, i.e. no detectable leakage) at *every* epsilon tested, including no DP at all.
2. DP noise further reduces the (already small) local-condition leakage, but the effect is modest at this sample size (single seed, 3-point grid).

### 8.4 What's not done yet

- The full multi-seed, full-epsilon-grid sweep (`{1,3,5,10,inf} × 3 conditions × 3-5 seeds`) on Colab GPU — this local run used a reduced grid specifically to validate the pipeline's correctness (Verification Plan items 2-5 all pass), not to produce final paper-quality numbers.
- Drift-aware controller and imaging-modality (PneumoniaMNIST) stretch goals — deferred per the project's own scope decision (core-first, stretch goals only if ahead of schedule).

---

## 9. Where to look for what

| Question | Where |
|---|---|
| "Why was X scoped in/out?" | `fedcare_project_plan.md` |
| "What papers support this?" | `fedcare_master_project_guide.md` §8 |
| "What were the final results / how do I cite this?" | `paper/report.md` |
| "How does module X work / how do I run it?" | This file |
| "What are the raw numbers behind result Y?" | `results/*.json` |
| "How do I show a panel that this actually works, live?" | `demo/README.md` |
