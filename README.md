# FedCare

Personalized federated learning for healthcare diagnostics, with empirically-audited privacy.

See [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) for the full technical guide (what every module does, how to run everything, findings so far). See [`fedcare_project_plan.md`](fedcare_project_plan.md) for the original scope/architecture/roadmap, and [`fedcare_master_project_guide.md`](fedcare_master_project_guide.md) for the literature review and extended (drift-controller) angle.

**Core research question**: given that hospitals legitimately have different patient populations (non-IID), can personalization (FedProx) recover the accuracy lost to differential privacy, without reopening the privacy leak DP was meant to close (measured via membership-inference attack)?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repo structure

```
data/            # dataset loading + Dirichlet non-IID partitioning
models/          # MLP (tabular), CNN (stretch goal: imaging)
fl/              # Flower client + FedAvg/FedProx strategies + drift-aware controller
privacy/         # Opacus DP-SGD wrapper, membership-inference audit
experiments/     # runnable scripts: baseline, FL sweep, MIA audit
notebooks/       # EDA and results plotting only (no core logic)
demo/            # real networked FL demo (separate processes) -- for presenting live to a panel
results/         # summarized metrics/plots (raw data gitignored)
paper/           # IEEE-format report/paper source
```

## Running the pipeline

```bash
python experiments/run_baseline.py    # centralized baseline (accuracy ceiling)
python experiments/run_fl_sweep.py    # local vs FedAvg vs FedProx (Steps 5-6)
python experiments/run_mia_audit.py   # DP epsilon sweep + membership-inference audit (Steps 7-8)
```

Each writes metrics/figures to `results/`. See [`paper/report.md`](paper/report.md) for the write-up (IEEE short-paper structure) with all current numbers filled in.

## Presenting this live (panel demo)

The scripts above run all 5 hospitals as an in-process Flower simulation — correct for research, but nothing to visibly point at. [`demo/`](demo/README.md) runs the exact same model/client code as real, separate processes over a real network socket: one terminal per hospital, each loading only its own data file, sending only model weights. See [`demo/README.md`](demo/README.md) for setup and talking points.

## Status

Core pipeline (Steps 1-9) is built and runs end-to-end on a local CPU/MPS validation grid: dataset EDA, Dirichlet non-IID partitioning, centralized baseline (86.6% acc), local vs FedAvg vs FedProx comparison, a reduced DP epsilon sweep, and a membership-inference audit (sanity-checked against an overfit model, AUC 0.80). Two headline findings, both reported honestly in `paper/report.md` rather than smoothed over:

1. Vanilla FedAvg can badly harm a hospital whose local population diverges from the global pool (worst-client accuracy 78.9% trained alone vs 34.7% under FedAvg) -- and FedProx does **not** recover this at any tested mu, a verified negative result, not a bug.
2. Federated pooling itself, independent of DP, is the dominant privacy mechanism here: the shared FedAvg/FedProx model shows near-zero measured MIA leakage at every epsilon including no DP, while fully local per-hospital models leak measurably more.

**Also built**: the drift-aware controller (`fl/drift_controller.py`, previously a deferred stretch goal) — adaptively tunes each hospital's personalization strength, DP privacy level, and aggregation weight based on measured drift and a leakage-risk proxy, live in the demo (see [`demo/README.md`](demo/README.md#the-drift-aware-controller)). Verified as a real, correctly-functioning system; not yet run through a proper multi-seed research comparison.

**Not yet done**: the full multi-seed, full-epsilon-grid sweep on Colab GPU (this local run used a reduced 3-point grid, 1 seed, to validate correctness — see `paper/report.md` Section 3.6 and 4.3), a research-pipeline validation of the drift-aware controller, and the imaging stretch goal (deferred per project scope decision).
