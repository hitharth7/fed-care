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

**Answer found so far** (see §8): more layered than it first looked. The initial FedAvg run appeared to collapse one hospital's accuracy (78.9% local → 34.7% federated), unrecoverable by FedProx at any mu — but that collapse turned out to be substantially a fixed-threshold artifact under per-client label shift, not a training failure (that hospital's AUC was 0.82 throughout). Correcting for it and re-comparing local vs. FedAvg fairly on both sides shows federation gives a modest *net* benefit, but not to the most atypical hospitals — see §8.2 for the full diagnosis and corrected numbers. Separately, federated pooling itself (independent of DP) turns out to be the dominant privacy mechanism in this setup (§8.3), though those results predate the correction and should be re-run. See `paper/report.md` for the full discussion.

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
    drift_controller.py     # rule-based adaptive controller: drift/leakage-risk -> mu/epsilon/aggregation weight
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

Runtimes above are for a 2024 MacBook (Apple MPS backend, no CUDA). On Colab with a T4 GPU, the FL/DP scripts should be faster; the *reduced* grids used in `run_mia_audit.py` are specifically sized to be tractable locally (see §8.5) — expect to edit that script's grid before running the full-scale version.

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

Any combination of `mu` and `dp_config` is valid — Step 7's DP sweep runs all three FL conditions with DP simultaneously using this same class. `get_parameters`/`set_parameters` do the ndarray ↔ state_dict conversion Flower's `NumPyClient` interface expects; `evaluate_model()` (shared with the non-Flower "fully local" training path in `run_fl_sweep.py`) computes accuracy/balanced_accuracy/F1/AUC/**PR-AUC/sensitivity/specificity/MCC/Brier/ECE** identically regardless of which path produced the model (§8.6 Phase 1 — the fuller metric set was an external-review finding: accuracy/balanced-accuracy/AUC alone under-report calibration and minority-class performance).

**`use_calibration_split=True` (§8.6 Phase 1)** — fixes a real leakage the external review caught: `tune_threshold()` was picking the decision cutoff on the same examples (`train_loader`) that fit the model's weights. When enabled, `_split_for_calibration()` holds out 20% of the training shard (deterministic, seeded) purely for threshold tuning; the optimizer never sees it. **Hard-incompatible with DP by design, not oversight** — `client_dp_configs()` pre-calibrates each client's privacy accounting against the *full* training-set size before the client exists, so shrinking the fit set would desync the realized sample_rate from the epsilon it was calibrated for. Two guards enforce this: one at construction (`dp_config` given directly) and one inside `fit()` (the drift controller can inject DP mid-run via `config["target_epsilon"]` even when `dp_config` was `None` at construction — both paths are covered). Default `False`, so every existing result is provably unaffected (when off, `_fit_loader`/`_calib_loader` are literally the same object as `train_loader`, not just behaviorally equivalent) — confirmed by re-running client 1's corrected local balanced accuracy and matching the pre-existing 0.7453 exactly.

**Added after diagnosing the alpha=0.5 collapse (§8.2)** — two optional, independently-toggled corrections for per-client label shift, both off by default so every existing result stays exactly reproducible:

- `compute_pos_weight(loader)` + `make_criterion(pos_weight, device)` — `use_pos_weight=True` weights each client's local `BCEWithLogitsLoss` by its own `n_negative/n_positive`, so training targets that client's own class balance instead of the pooled dataset's.
- `predict_logits()` + `tune_threshold()` — `per_client_threshold=True` has each client pick its evaluation threshold by balanced accuracy on its own *training* shard (never validation or the global test set), then apply that threshold when evaluating. `evaluate_model()` takes an explicit `threshold` argument (default 0.5) and now also reports `balanced_accuracy` and the `threshold` used, in the returned metrics dict.

Both flags thread through `DiabetesFlowerClient.__init__`, `run_fully_local()`, and `run_fl_condition()`/`run_fedavg()` in `experiments/run_fl_sweep.py`.

Two more things layered on for the drift-aware controller (§5.4a), both **off by default** so Steps 5-8 are completely unaffected:
- `fit()`'s `config` dict may carry `"mu"` / `"target_epsilon"` overrides for *just that round* (falls back to the constructor's fixed `mu`/`dp_config` when absent — this is how a controller actually controls a client without needing a different class).
- `track_controller_metrics=True` makes `fit()` also report `"drift"` (relative L2 distance the local update moved from the global model it started from) and `"overfit_gap"` (that client's own train accuracy minus its own val accuracy, right after local training) — the two signals the controller reacts to.

**A real bug worth knowing about if you touch this file**: networked clients (`demo/`) reuse *one* persistent client/model object across every round, unlike simulation-mode clients, which get a fresh model each round via `client_fn`. This surfaced two real, now-fixed issues once DP was wired into the demo: (1) `evaluate()` leaves the model in `.eval()` mode, and Opacus's validator refuses to wrap a model that isn't in `.train()` mode — fixed by calling `self.model.train()` at the very top of `fit()`, before anything else. (2) Opacus refuses to attach hooks to an already-wrapped model — fixed by calling `model.remove_hooks()` at the end of `fit()` whenever that round used DP, restoring a clean model for the next round's wrap. Neither bug could have shown up in Steps 5-8's testing (simulation mode never reuses a model across rounds); both were caught by testing the real networked demo directly rather than assuming the simulation-tested code would behave identically.

### 5.4a `fl/drift_controller.py`

The drift-aware controller — `fedcare_master_project_guide.md` §6's "Selected New Idea," previously deferred as a stretch goal (per the "core first" scope decision), now built and wired into the live demo.

After every round, for each hospital: **drift** (how far that hospital's local update moved from the shared model, relative L2 distance) and **overfit_gap** (that hospital's own train accuracy minus val accuracy — a cheap, always-available proxy for membership-inference leakage risk, since running Step 8's actual attack every round would be far too slow for a live demo) decide that hospital's settings for the *next* round, via fixed, simple thresholds (not learned — deliberately, per the project's own risk register: "keep it lightweight and rule-based"):

- high drift → raise `mu` (pull that hospital back toward the group more strongly) + down-weight its contribution to *this* round's aggregation
- low drift → relax `mu` (a well-aligned hospital gets more room to personalize)
- high overfit gap → tighten privacy (lower `target_epsilon` → more DP noise) for that hospital next round
- low overfit gap → relax privacy back toward the baseline

`ClientControllerState.cumulative_epsilon()` composes the *true* privacy cost across every round so far (feeding each round's possibly-different noise level into one `RDPAccountant` sequentially) — deliberately tracked and surfaced separately from the per-round `target_epsilon`, since conflating the two would repeat exactly the kind of misleading-epsilon mistake `privacy/dp.py` (Step 7) was built to avoid. Verified live: over a 20+ round networked run, cumulative epsilon grew monotonically and consistently (e.g. 7.99 → 10.82 → 13.16 → 15.25 → 17.17 across 5 rounds at a fixed per-round target of 8.0) — exactly the expected multi-round composition behavior.

`DriftAwareFedAvg(fl.server.strategy.FedAvg)` is the strategy that makes this real: `configure_fit` injects each client's current `mu`/`target_epsilon` into their per-round config (keyed by Flower's own `client.cid`, confirmed stable across rounds in both simulation and real networked mode — see the design-decisions table); `aggregate_fit` reads back `drift`/`overfit_gap` from the returned metrics, updates that client's controller state, and down-weights high-drift clients' contribution before calling the standard FedAvg aggregation. `demo/run_server.py` wires an `on_decision` callback into this to mirror every decision into the dashboard.

**Honest scope note**: validated as a real, correctly-functioning system (tested live, including the DP composition and per-client mu/weight adjustments behaving as designed) — not yet run through the same multi-seed research comparison as Steps 5-8. Whether it actually *improves* worst-client accuracy or leakage over the static approach is the natural next experiment, not yet run.

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

- **`run_step5c_corrected(...)`** — re-runs local vs. vanilla FedAvg at `alpha=0.5` with `use_pos_weight=True, per_client_threshold=True` on *both* conditions, since Step 5's original local numbers have the same fixed-threshold artifact hiding in them (clients 0/2/3 also score f1=0.0 there) and aren't a fair baseline to compare a corrected FedAvg against. Writes `results/fl_step5c_corrected.json`. Runs automatically after `run_step6_fedprox()` when the script is executed directly. **Not yet done**: the mu grid itself hasn't been re-run with correction — see §8.5.

### 5.10 `experiments/run_mia_audit.py` — Steps 7-8

Combines the DP epsilon sweep and the MIA audit into one pass per (condition, epsilon) — trains once, then immediately attacks the resulting model, avoiding a separate retraining step.

- **`sanity_check_attack(...)`** — Verification Plan item #4: attack a deliberately overfit, no-DP, no-regularization model first; must show AUC > 0.7 or the attack itself is presumed broken. Uses a small member set (30 samples) and an intentionally over-large model (256×128 hidden layers) — see the function's docstring for why a naive first attempt (500 members, normal-sized model) only reached AUC 0.58 despite near-perfect memorization, and why that's a real, documented property of loss-threshold attacks on imbalanced data rather than an attack bug.
- **`run_privacy_sweep(...)`** — for each epsilon in the grid, trains all three FL conditions (with DP where `epsilon != inf`) and runs the MIA attack against each. For `local`, each client has its own model (members = own train set, non-members = other clients' train sets + global test set). For `fedavg`/`fedprox`, there's one shared model (members = that client's own train set, non-members = the global test set only — every other client is *also* a member of this shared model, so can't serve as a non-member).
- **`plot_privacy_tradeoff(...)`** — the paired accuracy-vs-epsilon / attack-AUC-vs-epsilon figure, one line per FL condition.

### 5.11 `demo/` — real networked demo, for presenting to a panel

Everything above runs in Flower's **simulation mode**: all 5 hospitals execute as virtual clients inside one Python process (or, more precisely, one Ray-managed process pool — see §6). That's correct and standard for producing research results, but there's nothing to visibly point at in a live demo.

`demo/` is a separate presentation layer using the *exact same* model and client code (`models/mlp.py`, `fl/client.py`), but deployed as real, separate OS processes communicating over an actual gRPC network socket — the same transport Flower uses in production. It does not feed into any research result in `paper/report.md`; it exists purely so a panel can watch federated learning happen rather than take it on faith.

- **`prepare_demo_data.py`** — one-time setup. Computes the same Dirichlet partition as the research pipeline (`alpha=0.5`, 5 hospitals, `seed=42`) and **materializes each hospital's data as its own separate `.npz` file** on disk (`demo/hospital_data/hospital_{0..4}_{train,val}.npz`) — a genuinely separate, inspectable artifact per hospital, not just an in-memory slice. This mirrors how a real deployment would provision each hospital's local database once, ahead of training.
- **`run_server.py`** — a real Flower server (`flwr.server.start_server`) with narrated per-round logging of every hospital's reported accuracy before/after aggregation. By default builds `DriftAwareFedAvg` (§5.4a) rather than plain `FedAvg` — `--no-adaptive` switches back to plain FedAvg for a before/after comparison. Also starts a small local HTTP server (`http.server`, stdlib, default port 8090, background thread) serving `dashboard/index.html` and a `dashboard_state.json` it rewrites atomically after every fit/evaluate aggregation round *and* after every controller decision (via the `on_decision` callback) — this is the *only* connection between the server and the dashboard; the dashboard is a read-only mirror of real strategy/controller callbacks, not a separate data path.
- **`run_hospital.py`** — a real Flower client (`flwr.client.start_client`) that loads *only* its own `.npz` file (never another hospital's), trains via the same `DiabetesFlowerClient` with `track_controller_metrics=True` by default (`--no-adaptive` disables it, matching the server flag), and prints what it's doing at each step — including the controller-assigned `mu`/`target_epsilon` it received and its own drift/overfit_gap once training completes. Run once per hospital, in its own process/terminal.
- **`dashboard/index.html`** — a single self-contained page (inline CSS/JS, Chart.js from CDN for the accuracy-over-rounds chart) polling `dashboard_state.json` every 800ms: a central "Global Model" node with 5 hospital nodes around it, connecting lines that light up when a hospital sends weights, per-hospital accuracy color-coded, a per-hospital controller panel (drift, overfit gap, mu, per-round target epsilon, true cumulative epsilon, and the latest decision's plain-English reason), and a live activity log. Verified end-to-end with a real running server + 5 real hospital processes (not just the JSON, the *rendered page* too, via headless Chrome screenshots and DOM dumps against the live server). This caught real bugs worth knowing about if you touch this file: `backdrop-filter: blur()` on the card backgrounds rendered as fully invisible in headless Chrome (fixed by using an opaque-enough base color instead of relying on blur compositing for visibility); a `resize` event listener that wiped and recreated all hospital cards was firing on spurious resize events and could leave the visualization empty (removed — `ensureCards()` already recomputes layout every poll tick, so it was unnecessary); and, after the controller made cards significantly taller, the absolutely-positioned cards at the bottom of the pentagon layout genuinely overflowed the `.stage` container and overlapped the panels below it (fixed with a much larger `margin-bottom` on `.stage` — absolutely-positioned children don't reserve layout space, so only a real margin guarantees clearance regardless of exact card height). See `demo/dashboard_preview.png` for a real screenshot from an actual run.

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
| Overfit-gap proxy instead of a real MIA attack for the controller's leakage signal | Running Step 8's actual attack every round would be far too slow for a live demo; train-vs-val accuracy gap is a cheap, always-available, defensible stand-in (the classic precondition for the memorization MIA measures directly) |
| Controller's `target_epsilon` and `cumulative_epsilon` shown as two separate dashboard numbers | Conflating "what this round wants to cost" with "true total cost so far" is exactly the kind of misleading-epsilon mistake `privacy/dp.py` was built to avoid — a per-round-adaptive controller doesn't get a pass on that discipline |

---

## 7. Known gotchas

- **SSL cert error on first dataset download (macOS)**: already handled in `data/loaders.py` (see §5.1) — you shouldn't hit this, but if you do, it's the python.org Framework build's missing CA bundle, not a network issue.
- **`flwr.simulation.start_simulation` prints a loud deprecation warning** every run. Expected — see the "Design decisions" table above. It still works correctly on the pinned version.
- **`pip install flwr[simulation]` is required**, not just `flwr` — the bare package raises `ImportError` on `ray` when you actually try to run a simulation. Already in `requirements.txt`.
- **Ray/MPS interaction**: multiple concurrent Ray actors *can* look like they're producing suspiciously-identical results — if you ever see a parameter sweep where every value in the sweep gives bit-identical results, don't assume it's a Ray caching bug before checking whether the underlying *metric* (e.g., thresholded accuracy) is just saturated. This exact pattern happened during Step 6 development (see `paper/report.md` §4.2) and took real debugging effort to distinguish from an actual bug — AUC moved correctly the entire time; only accuracy was pinned.
- **DP-SGD + a persistent (networked, multi-round) client object**: if you see `IllegalModuleConfigurationError('Model needs to be in training mode')` or `ValueError: Trying to add hooks twice to the same model` from Opacus, it means a client's `self.model` is being DP-wrapped a second time without being reset first — only possible in `demo/` (real networked clients reuse one model object across all rounds; simulation-mode clients don't). Already fixed in `fl/client.py`'s `fit()` (`self.model.train()` before wrapping, `model.remove_hooks()` after) — mentioned here in case a future change to that method reintroduces it.

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

**Diagnosis (§8.2b) found the collapse above is substantially a fixed-threshold artifact, not a training failure**: client 1's AUC under FedAvg is 0.822 (the model ranks its patients well), and **balanced accuracy is 0.500 — chance — for every one of the 5 clients**, not only client 1, once the accuracy metric's imbalance-driven floor is accounted for. Clients 0/2/3's 93.9–99.3% accuracy above is itself an artifact of 94–99% of their patients being the negative class.

#### 8.2b Correction and the fair local-vs-FedAvg comparison

Fix (`fl/client.py`, §5.4): per-client `pos_weight` in the training loss + a per-client decision threshold tuned on each client's own training shard. Client 1, before/after (AUC unchanged — only the readout changed): accuracy 0.347 → 0.758, F1 0.000 → 0.810.

Section 8.2's "fully local" numbers have the same artifact hiding in them (clients 0/2/3 also score f1=0.0 there), so re-ran **local vs. vanilla FedAvg with the same correction on both** (`run_step5c_corrected()`, `results/fl_step5c_corrected.json`):

| Client | Local bal. acc | FedAvg bal. acc |
|---|---|---|
| 0 | 0.743 | **0.783** |
| 1 | **0.745** | 0.744 |
| 2 | 0.737 | **0.750** |
| 3 | 0.764 | **0.770** |
| 4 | **0.733** | 0.721 |
| **Mean** | 0.744 | **0.753** |

**Federation wins on average** (0.753 vs 0.744), but not uniformly: clients 0/2/3 (typical local class priors) gain from pooling; clients 1 and 4 (the two furthest from the federation's typical patient mix) do marginally *worse* under the shared model than alone. This is the expected signature of non-IID heterogeneity limiting FedAvg for atypical clients specifically — and it's a sharper, more defensible finding than the original "collapse." See `paper/report.md` §4.2b–4.2c.

**Not yet tested**: the FedProx μ grid hasn't been re-run with this correction — whether corrected FedProx closes the remaining gap for clients 1/4 (which corrected vanilla FedAvg does not) is the natural next experiment (§8.5).

### 8.3 Privacy audit (Steps 7-8)

MIA sanity check passes (AUC 0.80 against a deliberately overfit model). Two findings:

1. **Fully local models leak measurably more than the pooled FedAvg/FedProx model, even without any DP.** Local per-client attack AUC: 0.52-0.63 (mean ~0.59). FedAvg/FedProx shared model: attack AUC ~0.485 (statistically ≈0.5, i.e. no detectable leakage) at *every* epsilon tested, including no DP at all.
2. DP noise further reduces the (already small) local-condition leakage, but the effect is modest at this sample size (single seed, 3-point grid).

**These models predate the §8.2b correction** (no `pos_weight`, fixed 0.5 threshold) — re-running this sweep against corrected models is pending (§8.5); a model that actually learns the positive class rather than one whose probabilities cluster near 0 may show different memorization behavior.

### 8.4 The drift-aware controller (built, demo-only so far)

The previously-deferred stretch goal is now built (`fl/drift_controller.py`, §5.4a) and running live in `demo/`, adaptively tuning each hospital's FedProx `mu`, DP `target_epsilon`, and aggregation weight based on measured drift and an overfit-gap leakage-risk proxy. Verified as a real, correctly-functioning system over live 20+ round networked runs — but **not yet run through Steps 5-8's rigor**: no multi-seed comparison against the static approach, no `experiments/` script, no numbers in `paper/report.md`. It demonstrably reacts correctly to what it observes; whether that reaction actually improves worst-client accuracy or leakage over the static Step 5-8 approach is the natural next experiment. **It also predates the §8.2b correction** — it's tuning `mu`/epsilon on top of the same unweighted-loss, fixed-0.5-threshold models, so its drift/overfit-gap signals should be re-validated once the controller's clients also use `pos_weight`/per-client thresholds.

### 8.5 What's not done yet

- **Re-run the DP/MIA sweep (§8.3) against corrected models** (`use_pos_weight=True, per_client_threshold=True`) — highest priority, since §8.3's current numbers are evaluated against the pre-fix models.
- **Re-run the FedProx μ grid with the same correction** — tests whether personalization closes the clients 1/4 gap that corrected vanilla FedAvg (§8.2b) does not.
- If it doesn't: explicit per-client personalization (FedPer — local classifier head, shared feature extractor — or Ditto) targeted specifically at clients 1 and 4.
- The full multi-seed, full-epsilon-grid sweep (`{1,3,5,10,inf} × 3 conditions × 3-5 seeds`) on Colab GPU — this local run used a reduced grid specifically to validate the pipeline's correctness (Verification Plan items 2-5 all pass), not to produce final paper-quality numbers.
- A proper research-pipeline validation of the drift-aware controller, re-run against corrected (§8.2b) models (see §8.4) — and the imaging-modality (PneumoniaMNIST) stretch goal, deferred per the project's own scope decision (core-first, stretch goals only if ahead of schedule).

### 8.6 External review findings and phased response

An outside review of the project (pre-submission assessment, not a peer reviewer) found two problems worth recording permanently:

1. **`DriftAwareFedAvg` (§8.4/§5.4a) has never been run through the offline experiment pipeline.** It exists only in `demo/`. Zero result files, zero comparison against FedAvg/FedProx, despite being the project's actual novel contribution. This is the single most important gap — everything else is secondary until it's closed.
2. **FedProx is not personalization.** It produces one shared global model (Li et al. 2020); the paper's title/research question claiming "personalized FL" is not supported by what's implemented. Either reframe (drop the claim) or implement real personalization (Ditto/FedPer) and revisit the earlier decision (§8.5) to skip it.

Response plan, in priority order (each phase gets its own debrief entry below as it completes):

- **Phase 1 (done, this entry)** — measurement fixes that had to land *before* any benchmarking, or every subsequent run would need repeating: (a) added PR-AUC/sensitivity/specificity/MCC/Brier/ECE to `evaluate_model` — accuracy/balanced-accuracy/AUC alone under-report calibration and minority-class behavior under 86/14 imbalance; (b) fixed `tune_threshold()`'s train/calibration leakage via `use_calibration_split`, deliberately incompatible with DP (two guards — see §5.4) rather than risk silently corrupting the privacy accounting under time pressure; (c) `run_fl_condition` now accepts `strategy_factory`, so any Flower strategy — including `DriftAwareFedAvg` — can run through the exact same harness as FedAvg/FedProx, which is the prerequisite for closing gap #1 above. **Verified, not just written**: new metrics tested against synthetic data including the single-class edge case heart disease's tiny shards actually hit; ECE checked against known-calibrated/known-miscalibrated inputs; the calibration split confirmed disjoint/proportioned/deterministic; both incompatibility guards confirmed to actually raise; and a regression check reproduced client 1's exact pre-existing balanced accuracy (0.7453) with the new code at default settings, confirming no prior result was disturbed.
- **Phase 2 (next)** — the actual missing experiment: run `DriftAwareFedAvg` through this harness, add a static-μ control condition (isolates whether *adaptivity* helps or the controller just found a good fixed μ by luck), and benchmark against FedAvg/FedProx/FedAvgM/FedAdam/FedYogi/QFedAvg/DPFedAvgAdaptive — the last four are already installed via Flower, zero implementation cost.
- **Phase 3** — ablations (drift-only vs. overfit-only vs. both vs. no down-weighting), hyperparameter sensitivity for the hardcoded thresholds (`DRIFT_HIGH=0.20` etc. in `fl/drift_controller.py`), and the threshold-vs-federation ablation (does per-client calibration explain most of the §8.2b gain, independent of federation itself?).
- **Phase 4** — writing: retitle away from unsupported "personalized" claim, retire "federation is itself a privacy mechanism" (the corrected local-vs-federated MIA gap nearly closed after §8.2b's fix, undermining that claim — see §8.3), full DP hyperparameter table, limitations section.

**Learning worth keeping**: the external review's confound argument on the privacy finding is stronger than this document's own earlier explanation (§8.3) — local vs. federated MIA leakage differing is at least partly explained by effective training-set size (local trains on one small shard; federated is shaped by the full pool), not necessarily by federation conferring privacy. Don't restate the strong version of that claim until it's tested with a controlled comparison.

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
