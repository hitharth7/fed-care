# FedCare: Personalized Federated Learning for Healthcare Diagnostics with Empirically-Audited Privacy

## Context

This is the 7th-semester capstone project for a 2-person team at VIT. Idea #10 from the brainstorm doc (`implementation_plan.md`) — "Privacy-Preserving Federated Learning for Healthcare Diagnostics" — was flagged as a real-world-relevant but **overused template** (FedAvg + DP-SGD on a health dataset is a standard Flower/Opacus tutorial exercise; any examiner familiar with FL has seen it). The goal of this plan is to keep the accessible, well-scoped core, but bolt on a genuine empirical contribution so the project is defensible as novel rather than "followed a tutorial."

Constraints established with the user, driving every scope decision below:
- **Compute**: free tier only (Colab/Kaggle free GPU, no paid credits).
- **Timeline**: 14-16 weeks, standard VIT review cadence (proposal → mid-review → final).
- **Team skill**: solid ML theory, but new to PyTorch — the plan front-loads a short PyTorch ramp-up before FL code.
- **Novel angle chosen**: **personalized FL (FedProx) for non-IID hospitals** + **membership-inference privacy auditing** (Byzantine-robustness was explicitly declined — good call, it would've added a third unrelated threat model and diluted the story).
- **Dataset**: tabular primary, imaging as an explicitly droppable stretch goal.
- **Publication goal**: write it up as a short paper (IEEE format); patenting was declined after flagging that pure algorithms are excluded from patentability in India under Patents Act §3(k) unless tied to a specific technical/hardware effect.
- **Demo**: local only, no hosted deployment — so no infra/deployment cost or complexity is in scope.
- **Work split**: not requested — plan is purely technical/timeline, no per-person task assignment.

**The single sentence that makes this novel**: *"We built a federated system that adapts to each hospital's own patient population instead of forcing one global model, and we don't just claim patient privacy via a theoretical ε — we measure it, by attacking our own model."*

---

## Why this scope (not more, not less)

The original analysis suggested Byzantine-robustness + MIA auditing as the strongest combo. The user chose **personalization + MIA** instead — this is equally strong and arguably more coherent: Byzantine-robustness answers "what if a hospital is malicious," which is a different question from "how do we handle honest hospitals with different patient demographics." Personalization + privacy-auditing both live on the same core FL system and reinforce a single research question:

> *Given that hospitals legitimately have different patient populations (non-IID), can personalization recover the accuracy lost to differential privacy, without reopening the privacy leak that DP was meant to close?*

That framing — personalization vs. privacy audited together, not as two separate bolt-ons — is the paper's contribution. Everything below is built to produce that one result cleanly.

---

## System Architecture

```
                     ┌─────────────────────────────┐
                     │   Central Server (Flower)    │
                     │   Strategy: FedAvg / FedProx  │
                     └──────┬───────┬───────┬───────┘
                            │       │       │
              ┌─────────────┘       │       └─────────────┐
              ▼                     ▼                     ▼
    ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
    │ Simulated Hospital │ │ Simulated Hospital │ │ Simulated Hospital │  (5 virtual
    │  Client 1 (shard)  │ │  Client 2 (shard)  │ │  Client N (shard)  │   Flower clients,
    │                    │ │                    │ │                    │   Dirichlet-skewed)
    │  Local train loop  │ │  Local train loop  │ │  Local train loop  │
    │  + Opacus DP-SGD   │ │  + Opacus DP-SGD   │ │  + Opacus DP-SGD   │
    └────────┬───────────┘ └────────┬───────────┘ └────────┬───────────┘
             │                      │                      │
             └──────────  after training  ──────────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │  Membership-Inference Audit   │
                     │  (attack each trained model:   │
                     │   can it tell "member" from     │
                     │   "non-member" record?)         │
                     └──────────────────────────────┘
```

Key implementation decision: **use Flower's built-in simulation mode (virtual clients in one process), not literal Docker containers per hospital.** Docker-per-hospital adds real networking/ops complexity for zero research payoff, and actively works against a free-tier-compute, PyTorch-still-learning team. Flower's simulation engine is standard practice in published FL research — this is a legitimate simplification, not a corner cut. Docker can be mentioned as a one-paragraph "how this would be deployed in production" discussion in the report without ever needing to build it.

---

## Components to Build

### 1. Data & Non-IID partitioning (`data/`)
- **Dataset**: CDC Diabetes Health Indicators (BRFSS 2015, public on Kaggle/UCI, ~253k rows, 21 features, binary target). Chosen over Pima Indians Diabetes (only 768 rows — too small to give each of 5 simulated hospitals enough data for meaningful local training *and* shadow-model/MIA training) and over MIMIC-III (requires credentialed access — not worth the overhead for a semester timeline).
- **Non-IID simulation**: use **Dirichlet-distribution partitioning** (standard method in FL literature, e.g. Hsu et al. 2019) over the label, controlled by concentration parameter α. Low α → highly skewed "hospitals" (some see almost only one class); high α → near-IID. This is the standard, citable way to simulate "different hospitals have different patient populations" without needing a dataset that literally has a hospital-ID column.
- Deliverable: a bar-chart of per-client label distribution at a couple of α values — visually proves the non-IID challenge is real, good report figure.

### 2. Centralized baseline (`models/`, `experiments/run_baseline.py`)
- A small MLP (2-3 hidden layers). **No BatchNorm** — Opacus's per-sample gradient computation doesn't support BatchNorm; use LayerNorm/GroupNorm or skip normalization. Deciding this now avoids a rework later.
- Train on the full pooled dataset → this is the accuracy ceiling everything else is compared against.
- Sanity check: accuracy should land in the range reported by public kernels on this dataset (~75-86%) — if wildly off, something's wrong before FL code is even introduced.

### 3. Federated core (`fl/client.py`, `fl/strategies.py`)
- `FlowerClient` wrapping the MLP (`get_parameters` / `fit` / `evaluate`).
- Three conditions to compare, all on the *same* non-IID partitions:
  1. **Fully local** — each hospital trains alone, no collaboration (lower bound).
  2. **Vanilla FedAvg** — one shared global model (the "everyone gets the same average" baseline).
  3. **FedProx personalization** — proximal term `μ‖w - w_global‖²` added to each client's local loss, allowing each hospital's model to drift toward its own population while still being pulled toward the global average. Small grid search over μ ∈ {0.001, 0.01, 0.1, 1} on a validation split before locking the final value.
- Metric that matters most: **per-client accuracy, especially the worst-served client** — the personalization story is "FedProx closes the gap for the hospital that vanilla FedAvg serves worst," not just "average accuracy goes up."

### 4. Differential privacy (`privacy/dp.py`)
- Wrap local training with Opacus `PrivacyEngine`.
- **Trap to avoid**: privacy loss composes across *all* communication rounds a client participates in, not per-round in isolation. Opacus's accountant must be configured with the *total* local steps across all rounds, or the reported ε will be wrong. Flag this explicitly before implementation — it's the single most common correctness bug in "FL + DP-SGD" projects.
- Sweep target ε ∈ {1, 3, 5, 10, ∞} crossed with the 3 FL conditions above → 15 configs, each run over 3-5 seeds for error bars.
- Deliverable: accuracy-vs-ε Pareto frontier, one line per FL condition — shows whether personalization buys back some of the accuracy DP costs.

### 5. Membership-inference privacy audit (`privacy/mia.py`) — the key differentiator
- Primary attack: **Yeom et al. (2018) loss-threshold attack** — for a trained model, compute per-sample loss; a record with unusually low loss is more likely to have been a training member. Report **Attack AUC** (sweep the threshold, plot ROC) rather than a single accuracy number: 0.5 = attacker no better than a coin flip (ideal privacy), 1.0 = perfect leakage.
  - Chosen over a full Shokri et al. (2017) multi-shadow-model attack as the *primary* method because it needs zero extra shadow-model training — much cheaper on free-tier compute. A lightweight 2-3-shadow-model version is a good stretch add-on if time/compute allow, not a blocker.
- **Member/non-member pools**: for a given client's trained model, "members" = that client's own training shard, "non-members" = other clients' shards + a global held-out test set never used in training anywhere. This is realistic for the FL threat model (an attacker trying to tell if a specific patient's record was used by a specific hospital).
- Sanity check before trusting any result: attack a deliberately overfit, no-DP, no-regularization model first — it should show a *clearly* elevated AUC (>0.7). If the attack can't beat 0.5 even against an overfit model, the attack implementation is broken, not the privacy mechanism.
- Run this attack against every config from the DP sweep → **Attack-AUC-vs-ε curve**, plotted alongside the accuracy-vs-ε curve on the same x-axis. This paired plot (utility cost vs. measured privacy gain, same ε axis) is the signature result of the whole project.

### 6. Stretch goal: imaging modality (`models/cnn.py`)
- Only attempt in weeks 12-13 if the tabular pipeline is fully done early. Repeat partitioning → FedAvg/FedProx → DP → MIA on **PneumoniaMNIST** (via the `medmnist` package — 28x28 grayscale, ~5.8k images, binary label, deliberately tiny so it trains fast even on CPU/free Colab).
- Explicitly droppable — call this out to the team early so nobody burns week 13 panicking about a CNN instead of polishing the paper.

### 7. Paper / report (`paper/`)
- IEEE-format short paper: Abstract → Intro (problem + threat model) → Related Work (FedAvg, FedProx, DP-SGD, membership inference) → Method (architecture + the 3-axis experiment grid) → Results (all plots/tables) → Discussion ("at what ε does personalization stop being able to buy back accuracy, and does the privacy audit still pass at that point?") → Conclusion.
- Doubles as the VIT project report. Target venue: check with your faculty advisor for a suitable student symposium/IEEE student conference track, or post to arXiv as a preprint to timestamp the contribution even without a formal venue.

---

## Suggested Repo Structure

```
fedcare/
  data/
    loaders.py        # download + preprocess CDC Diabetes Health Indicators
    partition.py       # Dirichlet non-IID partitioner
  models/
    mlp.py             # tabular MLP, Opacus-compatible (no BatchNorm)
    cnn.py              # stretch-goal small CNN for PneumoniaMNIST
  fl/
    client.py           # FlowerClient wrapper
    strategies.py       # FedAvg / FedProx strategy config
  privacy/
    dp.py               # Opacus PrivacyEngine wrapper + epsilon accounting
    mia.py               # loss-threshold (+ optional shadow-model) MIA attack
  experiments/
    run_baseline.py
    run_fl_sweep.py      # {local, FedAvg, FedProx} x {epsilon} grid runner
    run_mia_audit.py
  notebooks/
    01_eda.ipynb
    02_results_plots.ipynb
  results/               # summarized metrics/plots (raw data gitignored)
  paper/                 # LaTeX/Overleaf source
  README.md
```

Small `.py` modules + thin notebooks for orchestration/plots — avoids notebook-only spaghetti while staying simple enough for a team still ramping up on PyTorch.

---

## Week-by-Week Roadmap (14-16 weeks)

| Weeks | Phase | Deliverable |
|---|---|---|
| 1-2 | **Foundations**: PyTorch fundamentals (official 60-min blitz + 1-2 small non-FL training exercises), read FedAvg/FedProx/DP-SGD/MIA papers, set up repo + Colab/Kaggle environment, download + EDA the dataset | Literature survey + dataset EDA — feeds **Review 1 (proposal)** |
| 3-4 | Centralized baseline MLP; Dirichlet non-IID partitioner + visualization | Working baseline + non-IID skew figure |
| 5-7 | Flower `FlowerClient`; vanilla FedAvg vs. fully-local; implement FedProx; μ grid search; per-client accuracy comparison | 3-condition comparison table/plot |
| 8-9 | Opacus DP-SGD integration (careful multi-round epsilon accounting); ε sweep across all 3 FL conditions | Accuracy-vs-ε Pareto frontier — feeds **mid-review** |
| 10-11 | Membership-inference audit (loss-threshold attack); sanity-check on overfit model; run against full ε sweep | Attack-AUC-vs-ε curve (the signature result) |
| 12-13 | **If ahead**: imaging stretch goal on PneumoniaMNIST. **If behind**: skip it — add more seeds/ablations to tabular results instead | Either a second-modality result, or tighter error bars on the primary result |
| 14-16 | Write IEEE-format paper/report, polish README + architecture diagrams, prepare local live-demo script, slides | Final paper + **final review** demo |

Built-in slack: the imaging stretch goal (weeks 12-13) is the buffer — if the team falls behind at any point, drop it without touching the core deliverable.

---

## Cost Analysis (target: $0)

| Item | Choice | Cost |
|---|---|---|
| Compute | Google Colab free (T4 GPU) + Kaggle free (30 GPU-hrs/week) as backup when Colab throttles | $0 |
| Dataset | CDC Diabetes Health Indicators (public), MedMNIST (open license) | $0 |
| FL framework | Flower (flwr), simulation mode | $0 |
| DP library | Opacus | $0 |
| Everything else | PyTorch, scikit-learn, pandas, matplotlib, GitHub | $0 |
| Paper venue | arXiv preprint (free) + ask advisor about student-conference tracks | $0-minimal |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Dirichlet α too extreme → FedAvg fails to converge at all | Tune α empirically; validate that near-IID (high α) FedAvg tracks the centralized baseline before pushing α lower |
| Opacus + BatchNorm incompatibility | Design the MLP without BatchNorm from day one (LayerNorm/GroupNorm or none) |
| Federated DP-SGD epsilon accounting done per-round instead of across all rounds → invalid ε reporting | Explicitly account for total local steps across *all* communication rounds in the Opacus accountant config; sanity-check ε=∞ run matches non-DP accuracy |
| MIA attack looks like it "succeeds" or "fails" for the wrong reason | Sanity-check the attack against a deliberately overfit no-DP model first (must show AUC > 0.7) before trusting any DP-condition result |
| Colab session disconnects mid-sweep | Structure experiments as resumable scripts (checkpoint every N rounds), not one long notebook cell; fall back to Kaggle when Colab is throttled |
| Team still ramping up on PyTorch | Weeks 1-2 are explicitly reserved for a structured mini-curriculum before any FL code is written |

---

## Verification Plan (how you'll know each phase is actually correct, not just "runs without crashing")

1. Centralized baseline accuracy lands near published benchmarks for this dataset (~75-86%).
2. FedAvg at high Dirichlet α (near-IID) converges close to the centralized baseline — proves the FL loop itself is correctly implemented before non-IID or privacy is layered on.
3. Opacus run at target ε = ∞ (no added noise) reproduces the non-DP FedAvg accuracy — proves the DP wrapper isn't silently corrupting training.
4. MIA attack against an intentionally overfit, no-DP model shows AUC clearly above 0.5 (>0.7) — proves the attack implementation itself works before it's used to judge privacy.
5. Across the ε sweep, Attack AUC should trend monotonically toward 0.5 as ε decreases (more privacy) — if this trend is absent or inverted, revisit the accounting or attack code before trusting the headline plot.
