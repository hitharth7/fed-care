# FedCare: Personalized Federated Learning for Healthcare Diagnostics with Empirically-Audited Privacy

*Draft report / paper scaffold. IEEE short-paper structure. Numbers below are from the local CPU/MPS validation run (1 seed, reduced epsilon grid); re-run the full grid on Colab (Section "Compute strategy") before treating any number here as final. **Update**: Section 4.2's original "FedAvg collapses client 1" finding was diagnosed and partially corrected after initial drafting -- see Sections 4.2b-4.2c. The DP/MIA sweep in Section 4.3 predates this correction and evaluates the pre-fix models; it has not yet been re-run against corrected models (flagged in Section 5).*

## Abstract

FedCare simulates five hospitals training a shared diabetes-diagnostic model under realistic constraints: non-IID patient populations, differential-privacy noise, and an empirical (not just theoretical) privacy audit via membership inference. We find that vanilla FedAvg can badly harm a hospital whose local population diverges from the global pool (worst-client accuracy 78.9% trained alone vs 34.7% under FedAvg), and that FedProx (Li et al., 2020) -- despite measurably changing the trained model (AUC shifts with mu) -- does not recover this client's accuracy at any tested mu or local-epoch setting, a negative result we verify is structural rather than an undertuned hyperparameter. Auditing privacy with a loss-threshold membership-inference attack (Yeom et al., 2018), we find federated pooling itself, independent of differential privacy, is the dominant privacy mechanism in this setup: the shared FedAvg/FedProx model shows near-zero measured leakage (attack AUC ~0.485) at every epsilon tested including no DP, while fully local per-hospital models leak measurably more (attack AUC up to 0.63) even before any DP noise is added. These results, from a reduced local-compute validation grid pending a full Colab-scale sweep, reframe the project's original question: for this architecture, the binding privacy risk is training locally at all, not the choice of epsilon. A follow-up diagnosis (Section 4.2b) found the reported "collapse" was substantially a fixed-threshold artifact under class-imbalanced label shift, not a training failure: client 1's AUC was 0.82 throughout, and balanced accuracy was 0.50 (chance) for *every* client, not only client 1, once accuracy's imbalance-driven floor was accounted for. Correcting for this (per-client loss weighting and per-client decision thresholds, applied fairly to both the local and federated conditions) shows vanilla FedAvg gives a modest net benefit over local training (mean balanced accuracy 0.753 vs. 0.744), concentrated in clients with typical local class priors -- the two most atypical clients still do marginally better alone, which sharpens rather than overturns the case for explicit personalization.

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

### 3.3b Correcting for per-client label shift

Added after the diagnosis in Section 4.2b. Two independent, optional corrections in `fl/client.py`, both off by default so every prior result remains exactly reproducible as the uncorrected baseline:

- **Per-client loss weighting** (`compute_pos_weight`, `use_pos_weight=True`): `BCEWithLogitsLoss(pos_weight=n_negative/n_positive)`, computed from each client's own training shard, so local training targets that client's own class balance rather than the pooled dataset's.
- **Per-client decision threshold** (`tune_threshold`, `per_client_threshold=True`): each client selects its evaluation threshold by balanced accuracy on its own training shard (never its validation or the global test set), then applies that threshold to its validation set. This is a post-hoc calibration step, not a change to the shared model's parameters.

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

### 4.2b Diagnosing the collapse: label shift, not a broken model

The collapse above is a real symptom, but "FedProx cannot fix client 1" understates what is happening. Two checks against the exact Section 4.2 models:

- Client 1's AUC under vanilla FedAvg is **0.822** -- the model ranks client 1's patients well.
- **Balanced accuracy** (mean of per-class recall; insensitive to a class-imbalanced majority baseline) is **0.500 -- chance level -- for every one of the 5 clients**, not only client 1, under both local and FedAvg/FedProx at the fixed 0.5 threshold. Clients 0, 2, 3 only look fine on raw accuracy (93.9-99.3%) because 94-99% of their patients are the negative class -- an all-negative predictor scores well by construction there. Client 1 is not a broken outlier; it is the only client whose imbalance direction makes the same underlying failure visible in accuracy.

Mechanism: local positive rates range from 0.8% (client 0) to 65% (client 1) against a pooled 13.9%. Unweighted `BCEWithLogitsLoss` on this pooled-style objective drives predicted probabilities well below 0.5 for nearly every input -- the corrected thresholds recovered below sit at 0.03-0.07, not 0.5. That is harmless for clients whose local prior points the same direction as the pooled skew, and catastrophic for client 1, whose local prior is inverted relative to the pool.

**Correction applied** (Section 3.3b): per-client `pos_weight` during training, plus a per-client decision threshold tuned on each client's own training shard. Client 1, before and after (AUC unchanged -- only the readout changed):

| | Accuracy | F1 | AUC |
|---|---|---|---|
| Fixed 0.5, unweighted (Section 4.2) | 0.347 | 0.000 | 0.823 |
| Corrected | 0.758 | 0.810 | 0.823 |

### 4.2c The fair comparison: is federation still worth it once both sides are corrected?

Section 4.2's "fully local" column has the same artifact hiding inside it -- clients 0, 2, 3 also score F1=0.0 there, masked by their own negative-heavy priors -- so comparing an uncorrected local baseline against an uncorrected FedAvg is not a fair test of this project's actual research question (does collaboration help?). Re-running **local and vanilla FedAvg with the same correction applied to both**:

| Client | Local acc | FedAvg acc | Local bal. acc | FedAvg bal. acc |
|---|---|---|---|---|
| 0 | 0.750 | 0.685 | 0.743 | **0.783** |
| 1 | **0.762** | 0.758 | **0.745** | 0.744 |
| 2 | 0.714 | 0.725 | 0.737 | **0.750** |
| 3 | 0.687 | 0.683 | 0.764 | **0.770** |
| 4 | **0.741** | 0.709 | **0.733** | 0.721 |
| **Mean** | 0.731 | 0.712 | 0.744 | **0.753** |

(`results/fl_step5c_corrected.json`.) Mean balanced accuracy favors federation, by a modest 0.9-point margin -- FedAvg's shared model is a net improvement over training alone once evaluated correctly. The benefit is **not uniform**: clients 0, 2, 3 (relatively unremarkable local priors) gain from pooling, while clients 1 and 4 -- the two furthest from the federation's typical patient mix (65% and 11.4% positive respectively) -- do marginally *worse* under the shared model than alone. This is the expected signature of non-IID heterogeneity limiting FedAvg specifically for atypical clients, and it sharpens Section 4.2's finding: the proximal term was never positioned to fix a *calibration/label-shift* problem (it acts in parameter space, not on the decision output), which is a more precise claim than "personalization failed."

**Open item**: only vanilla FedAvg (mu=0) has been re-run with correction; the mu grid (Section 4.2) has not, so whether a corrected FedProx outperforms corrected FedAvg specifically for clients 1 and 4 remains untested (Section 5).

### 4.3 Privacy-utility-leakage tradeoff

Reduced local grid: epsilon in {1.0, 3.0, inf}, 3 conditions, 1 seed (`results/privacy_sweep_step7_8.json`, `results/privacy_tradeoff.png`). **These models predate the Section 4.2b/4.2c correction** (no `pos_weight`, fixed 0.5 threshold) -- re-running this sweep against corrected models is pending; a model that actually learns the positive class (rather than one whose probabilities cluster near 0) may plausibly show different memorization behavior, so the leakage numbers below should be treated as provisional pending that re-run.

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

**Was FedAvg actually broken, or was the evaluation?** Mostly the evaluation (Section 4.2b): a fixed 0.5 threshold under per-client label shift pinned balanced accuracy at chance (0.50) for *every* client, not just client 1, and correcting for it flips the headline comparison -- FedAvg beats local training on average (Section 4.2c), just not for the two most atypical clients. The right open question is therefore no longer "does personalization recover client 1" but **"does corrected FedProx close the residual gap for clients 1 and 4 that corrected FedAvg does not"** -- untested here (Section 4.2c's open item) and the natural next experiment.

**At what epsilon does personalization stop being able to buy back accuracy?** Under this project's *uncorrected* results, the question needed revising: FedProx (proximal-term-only) never bought back client 1's accuracy at *any* epsilon, DP or no DP (Section 4.2) -- so there was no epsilon threshold to find for FedProx specifically. That still stands as a finding about the proximal term in parameter space; it does not resolve the calibration question above, which lives in output space and is orthogonal to it. What DP *does* cost is per-client AUC for the local condition (Section 4.3, finding 2), on top of an already-present accuracy gap between local and FedAvg/FedProx that FedProx doesn't close -- though Section 4.3's models predate the correction and should be re-run before this specific claim is finalized.

**Does the privacy audit still pass at the point where utility is acceptable?** Yes, robustly -- the shared FedAvg/FedProx model shows near-zero measured leakage (attack AUC ~0.485) at every epsilon tested, including no DP at all. The more interesting empirical question this project surfaces is not "how much does DP cost FedProx" but **"pooling data across hospitals already suppresses membership leakage more than DP does, for models that never had much to leak in the first place."** That is a genuine, unplanned finding worth foregrounding in the abstract/conclusion alongside the (honest, negative) FedProx result.

**What would be needed to actually close the worst-client accuracy gap** (framed as future work, kept out of the core scope per the project's own anti-scope-creep decision from `fedcare_project_plan.md`): explicit per-client personalization beyond proximal-term FedProx -- e.g. a few epochs of local fine-tuning on top of the converged shared model (Ditto, Per-FedAvg-style), which was not attempted here. Given finding 1 above, any such scheme would need to be re-audited for leakage too, since moving toward more client-specific models likely increases attack AUC back toward the "local" condition's higher baseline -- i.e. personalization and privacy leakage may trade off against each other beyond what this project's FedProx-only design revealed, which is precisely the kind of interaction the project's original research question anticipated.

## 6. Conclusion

This project set out to test whether personalization (FedProx) could recover the accuracy differential privacy costs, without reopening the leak DP closes. The initial answer looked stark -- FedAvg appeared to collapse one hospital's accuracy from 78.9% to 34.7%, unrecoverable by FedProx at any mu -- but diagnosis showed most of that collapse was a fixed-threshold artifact under per-client label shift, not a training failure: the model's ranking of that hospital's patients (AUC) was fine throughout, and *every* client, not just the one that looked broken, was at chance-level balanced accuracy under the unweighted, fixed-threshold setup. Correcting for this (per-client loss weighting and per-client thresholds, Section 3.3b) and re-running local vs. FedAvg fairly on both sides (Section 4.2c) gives a more honest three-part result: **(1)** collaboration does help, by a modest overall margin, once measured correctly; **(2)** that benefit is uneven -- it does not reach the hospitals whose patient population differs most from the federation's, which is exactly where explicit per-client personalization (FedPer, Ditto, local fine-tuning on the converged shared model -- none of which were attempted here) should be targeted next; **(3)** the proximal term alone, evaluated fairly, has not yet been shown to close that specific remaining gap, and should be re-tested under the same correction before drawing a final conclusion about FedProx's value in this setting. Separately, the project surfaces a finding that survives this revision untouched: pooled federated training is itself a meaningful privacy mechanism relative to any single hospital training alone, largely independent of the DP epsilon chosen (Section 4.3) -- though those models predate the correction above and should be re-run against corrected models before this claim is finalized. Concrete next steps, in priority order: re-run the privacy audit against corrected models; re-run the FedProx mu grid under the same correction to test whether it closes the clients 1/4 gap; implement explicit per-client personalization (FedPer/Ditto) if it does not; and run the full multi-seed epsilon sweep on Colab to put confidence intervals on every trend reported here.

## Appendix: Reproducibility

All numbers above come from deterministic, seeded code (`SEED=42` throughout). Re-running any `experiments/*.py` script reproduces them exactly (same machine/backend) or closely (different hardware -- MPS vs CUDA vs CPU can cause small floating-point differences). Full pipeline: `experiments/run_baseline.py` -> `experiments/run_fl_sweep.py` -> `experiments/run_mia_audit.py`.
