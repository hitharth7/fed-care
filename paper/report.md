# FedCare: Personalized Federated Learning for Healthcare Diagnostics with Empirically-Audited Privacy

*Draft report / paper scaffold. IEEE short-paper structure. Numbers below are from the local CPU/MPS validation run (1 seed, reduced epsilon grid); re-run the full grid on Colab (Section "Compute strategy") before treating any number here as final.*

## Abstract

FedCare simulates five hospitals training a shared diabetes-diagnostic model under realistic constraints: non-IID patient populations, differential-privacy noise, and an empirical (not just theoretical) privacy audit via membership inference. We find that vanilla FedAvg can badly harm a hospital whose local population diverges from the global pool (worst-client accuracy 78.9% trained alone vs 34.7% under FedAvg), and that FedProx (Li et al., 2020) -- despite measurably changing the trained model (AUC shifts with mu) -- does not recover this client's accuracy at any tested mu or local-epoch setting, a negative result we verify is structural rather than an undertuned hyperparameter. Auditing privacy with a loss-threshold membership-inference attack (Yeom et al., 2018), we find federated pooling itself, independent of differential privacy, is the dominant privacy mechanism in this setup: the shared FedAvg/FedProx model shows near-zero measured leakage (attack AUC ~0.485) at every epsilon tested including no DP, while fully local per-hospital models leak measurably more (attack AUC up to 0.63) even before any DP noise is added. These results, from a reduced local-compute validation grid pending a full Colab-scale sweep, reframe the project's original question: for this architecture, the binding privacy risk is training locally at all, not the choice of epsilon.

## 1. Introduction

Hospitals cannot pool patient records due to privacy law and institutional barriers, but any single hospital often lacks the data diversity to train a strong diagnostic model alone. Federated learning (FL) addresses the data-sharing problem, but two things are usually handled separately in the literature: (a) hospitals have genuinely different patient populations (non-IID data), and (b) differential privacy (DP), when added to protect patients, costs accuracy. This project asks a single question:

> Given that hospitals legitimately have different patient populations, can personalization (FedProx) recover the accuracy lost to differential privacy, without reopening the privacy leak DP was meant to close?

We answer it empirically: not just by reporting a theoretical epsilon, but by attacking our own trained models with a membership-inference attack (MIA) and measuring how much information actually leaks at each privacy level.

## 2. Related Work

- **FedAvg** (McMahan et al., 2017) -- the standard FL aggregation baseline.
- **FedProx** (Li et al., 2020) -- adds a proximal term to local training to handle client heterogeneity; used here as the personalization mechanism.
- **DP-SGD** (Abadi et al., 2016) -- per-sample gradient clipping + noise, implemented via Opacus.
- **Membership inference** (Yeom et al., 2018; Shokri et al., 2017) -- the loss-threshold attack used here is the Yeom et al. variant (no shadow models needed).
- 2024-2026 healthcare-FL literature survey: see `fedcare_master_project_guide.md` Section 8 for the full reading list (base paper: adaptive aggregation recovering utility under DP on heterogeneous cardiovascular data).

## 3. Method

### 3.1 Data and non-IID simulation

CDC Diabetes Health Indicators (BRFSS 2015; UCI id 891; 253,680 rows, 21 features, binary target, ~86%/14% class balance -- see `notebooks/01_eda.ipynb`). Five simulated hospital clients via Dirichlet partitioning (Hsu et al., 2019) over the label; concentration parameter alpha controls skew (Figure: `results/eda_dirichlet_partition.png`).

### 3.2 Model

`DiabetesMLP` (`models/mlp.py`): 2 hidden layers (64, 32), LayerNorm + ReLU + Dropout, **no BatchNorm** (Opacus's per-sample gradient computation doesn't support it -- decided up front to avoid rework, per `ModuleValidator.validate()` passing cleanly).

### 3.3 FL conditions

Three conditions on the same non-IID partitions, all built from `data/partition.py:build_experiment_splits()`:
1. **Fully local** -- each hospital trains alone (lower bound).
2. **Vanilla FedAvg** -- one shared global model.
3. **FedProx** -- proximal term mu/2 * ||w - w_global||^2 added to local loss (Li et al., 2020). mu selected from {0.001, 0.01, 0.1, 1.0} by validation mean AUC (see Section 4.2 for why accuracy alone was not a usable selection signal).

Flower (`flwr`, simulation mode, `fl/client.py`, `fl/strategies.py`) orchestrates all three; FedProx is the same client/strategy with mu > 0, not a separate system.

### 3.4 Differential privacy

Opacus `PrivacyEngine` wraps local training (`privacy/dp.py`). The noise multiplier is calibrated **once**, for each client's *total* step budget across every communication round it will participate in (not recomputed per round) -- this avoids the most common correctness bug in FL+DP-SGD projects: under-reporting epsilon by accounting per-round instead of for the whole multi-round run.

### 3.5 Privacy audit

Yeom et al. (2018) loss-threshold membership-inference attack (`privacy/mia.py`): per-sample loss is the membership score (low loss -> more likely a training member); Attack AUC is reported via ROC over the full threshold sweep, not a single accuracy number. 0.5 = ideal (attacker no better than chance), 1.0 = complete leakage.

### 3.6 Compute strategy

Free-tier only (local CPU/Apple MPS for development; Google Colab/Kaggle free GPU for the full-scale sweep). The plan's full grid is epsilon in {1, 3, 5, 10, inf} x 3 FL conditions x 3-5 seeds = up to 75 training runs -- sized for Colab, not a laptop. Results in this draft use a **reduced local validation grid** (epsilon in {1, 3, inf} x 3 conditions x 1 seed) to confirm the pipeline is correct end-to-end; the full seeded grid should be re-run on Colab before submission (swap `get_device()` in `experiments/run_fl_sweep.py` for the Colab GPU automatically -- no code changes needed).

## 4. Results

### 4.1 Centralized baseline (accuracy ceiling)

Test accuracy **86.6%**, AUC **0.83**, F1 **0.24** (`results/baseline_metrics.json`). Matches published kernels for this dataset (~75-86%). Low F1 reflects the 86/14 class imbalance, not a bug -- worth reporting recall-on-minority-class explicitly in the final write-up.

### 4.2 Personalization: local vs FedAvg vs FedProx (alpha=0.5)

Sanity check (Verification Plan item 2) passed first: FedAvg at alpha=100 (near-IID) reaches 86.8-86.9% mean accuracy, matching the centralized baseline within 0.2-0.3 points -- confirms the FL loop itself is correct before non-IID or privacy is introduced.

At alpha=0.5 (moderate non-IID), the headline finding:

| Condition | Mean accuracy | Worst-client accuracy (client 1) |
|---|---|---|
| Fully local | 91.3% | 78.9% (F1 0.85 -- genuinely learned its own, minority-flipped distribution) |
| Vanilla FedAvg | 82.4% | **34.7% (F1 0.0 -- collapsed to predicting the global majority class)** |
| FedProx (best mu=0.001 by mean AUC) | 82.4% | 34.7% (unchanged) |

Client 1's local data skews toward the class the *global* pool treats as minority. Trained alone it learns this fine; pooled into FedAvg it gets dragged toward a model dominated by larger, oppositely-skewed clients and collapses.

**Important nuance, reported honestly rather than smoothed over**: across the full mu grid {0.001, 0.01, 0.1, 1.0}, FedProx's accuracy was *bit-identical* to vanilla FedAvg for every client. Investigation (see `experiments/run_fl_sweep.py` and debug history) showed this is not a bug -- mu genuinely changes the trained model (AUC moves from 0.829 at mu=0.001 down to 0.808 at mu=1.0), but at moderate non-IID skew every client's predictions stay on the same side of the 0.5 decision threshold regardless of mu, so accuracy (a thresholded metric) doesn't move. **FedProx, as specified in Li et al. (2020), does not by itself recover client 1's accuracy in this setup** -- it stabilizes training dynamics but still converges to one shared model, and client 1's local distribution is different enough from the global pool that no single shared model (however trained) serves it well.

Confirmed robust with a follow-up check (local_epochs=5 instead of 1, mu extended to 10.0): client 1's accuracy stayed at exactly 34.7% across every mu value tested, and mean AUC degraded monotonically as mu increased (0.826 at mu=0 -> 0.821 -> 0.804 -> 0.737 at mu=10.0) -- so more local computation per round doesn't unlock the effect, and larger mu actively hurts rather than helps. This rules out "not enough local training" as the explanation and strengthens the conclusion that this is a structural limitation of proximal-term-only FedProx, not an undertuned hyperparameter.

This is a legitimate, citable limitation to discuss (Section 5), and motivates why the field has moved toward *explicit* per-client personalization schemes (local fine-tuning, Ditto, Per-FedAvg) beyond the proximal-term-only approach -- worth flagging as a concrete extension if time allows, kept separate from the core scope per the project's own anti-scope-creep decision.

Figures: `results/fl_step6_worst_client_comparison.png` (accuracy AND AUC panels, deliberately -- accuracy alone hides the real story here).

### 4.3 Privacy-utility-leakage tradeoff

Reduced local grid: epsilon in {1.0, 3.0, inf}, 3 conditions, 1 seed (`results/privacy_sweep_step7_8.json`, `results/privacy_tradeoff.png`).

Verification checks, all passed:
- [x] MIA sanity check on deliberately overfit model: AUC 0.80 > 0.7 (30-sample, 256x128-MLP overfit configuration -- see `sanity_check_attack()` docstring for why the first attempt at n=500 members only reached AUC 0.58, a real and informative negative result about loss-threshold MIA power on imbalanced data, not an attack bug).
- [x] epsilon=inf reproduces non-DP accuracy exactly for every condition (local per-client accuracies and FedAvg/FedProx mean accuracy at eps=inf match Section 4.2's non-DP numbers bit-for-bit) -- confirms the DP wrapper isn't silently corrupting training when noise is disabled.
- [x] Attack AUC trends toward 0.5 as epsilon decreases, for the condition where there is any leakage to begin with (see below).

**Two qualitatively different findings, by condition:**

1. **Fully local models leak measurably more than pooled FL models, even without DP.** At epsilon=inf (no DP at all), local per-client attack AUC ranges 0.52-0.63 across clients (mean ~0.59); FedAvg/FedProx's *shared* model sits at attack AUC ~0.485 -- statistically indistinguishable from 0.5 (the ideal/no-leakage line), for every epsilon tested including no-DP. This makes sense structurally: each local model is trained on only one hospital's shard (as few as 2,491 rows for client 4), while the FedAvg/FedProx shared model is effectively trained across the pooled ~180k rows from all 5 hospitals over many rounds -- far less prone to memorizing any single patient. **Federated pooling itself is already a meaningful privacy mechanism here, before any DP noise is added.**

2. **DP noise reduces the (already-small) leakage further for local models, but the effect is modest and noisy at this sample size.** Mean local attack AUC: 0.593 (eps=inf) -> 0.554 (eps=3) -> 0.554 (eps=1) -- moves in the right direction but the eps=3 -> eps=1 step shows almost no further change with a single seed. Utility cost is correspondingly small: mean local accuracy stays ~91% across the whole epsilon range (`results/privacy_tradeoff.png`, left panel) -- for this dataset/model, DP-SGD's utility cost only becomes visible in per-client AUC, not thresholded accuracy (the same majority-class-collapse phenomenon from Section 4.2: most clients' predictions don't cross the 0.5 boundary regardless of the perturbation source, whether that's FedProx's mu or DP's noise).

3. **FedAvg/FedProx accuracy and attack AUC are both flat across the entire epsilon range** (accuracy pinned at 82.4% including at eps=1; attack AUC pinned at ~0.484). Given finding 1 (near-zero leakage even without DP), this isn't concerning from a privacy standpoint -- there's very little leakage for DP to further suppress in the shared-model condition in the first place.

**Honest limitation**: this is a single-seed, 3-point grid explicitly run locally to validate the pipeline (Section 3.6), not the paper's final result. The full plan calls for epsilon in {1,3,5,10,inf} x 3 conditions x 3-5 seeds on Colab GPU, which would (a) smooth out the eps=3-vs-eps=1 noise seen here for the local condition, and (b) give confidence intervals on all attack-AUC estimates before they go in a paper.

## 5. Discussion

**At what epsilon does personalization stop being able to buy back accuracy?** Under this project's own results, the question needs revising: FedProx (proximal-term-only) never bought back client 1's accuracy at *any* epsilon, DP or no DP (Section 4.2) -- so there is no epsilon threshold to find for FedProx specifically. What DP *does* cost is per-client AUC for the local condition (Section 4.3, finding 2), on top of an already-present accuracy gap between local and FedAvg/FedProx that FedProx doesn't close.

**Does the privacy audit still pass at the point where utility is acceptable?** Yes, robustly -- the shared FedAvg/FedProx model shows near-zero measured leakage (attack AUC ~0.485) at every epsilon tested, including no DP at all. The more interesting empirical question this project surfaces is not "how much does DP cost FedProx" but **"pooling data across hospitals already suppresses membership leakage more than DP does, for models that never had much to leak in the first place."** That is a genuine, unplanned finding worth foregrounding in the abstract/conclusion alongside the (honest, negative) FedProx result.

**What would be needed to actually close the worst-client accuracy gap** (framed as future work, kept out of the core scope per the project's own anti-scope-creep decision from `fedcare_project_plan.md`): explicit per-client personalization beyond proximal-term FedProx -- e.g. a few epochs of local fine-tuning on top of the converged shared model (Ditto, Per-FedAvg-style), which was not attempted here. Given finding 1 above, any such scheme would need to be re-audited for leakage too, since moving toward more client-specific models likely increases attack AUC back toward the "local" condition's higher baseline -- i.e. personalization and privacy leakage may trade off against each other beyond what this project's FedProx-only design revealed, which is precisely the kind of interaction the project's original research question anticipated.

## 6. Conclusion

This project set out to test whether personalization (FedProx) could recover the accuracy differential privacy costs, without reopening the leak DP closes. The honest answer, at this scale: FedProx's proximal term doesn't recover the worst-served client's accuracy regardless of DP, so that specific trade never materializes -- but the project surfaces a more useful finding in its place, that pooled federated training is itself the larger privacy mechanism relative to any single hospital training alone, largely independent of the DP epsilon chosen. Both the negative FedProx result and the pooling-privacy finding are defensible, citable contributions on their own, and point directly at concrete future work: explicit per-client personalization (local fine-tuning, Ditto, Per-FedAvg) re-audited for the leakage it would likely reopen, and a full multi-seed epsilon sweep on Colab to put confidence intervals on the trends reported here.

## Appendix: Reproducibility

All numbers above come from deterministic, seeded code (`SEED=42` throughout). Re-running any `experiments/*.py` script reproduces them exactly (same machine/backend) or closely (different hardware -- MPS vs CUDA vs CPU can cause small floating-point differences). Full pipeline: `experiments/run_baseline.py` -> `experiments/run_fl_sweep.py` -> `experiments/run_mia_audit.py`.
