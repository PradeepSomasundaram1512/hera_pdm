# PAPER_AUDIT — HERA-PdM

All numbers below are copied from `results/` (via `paper/numbers.tex`); the paper itself contains no hand-typed result values.

## A. Final paper title
**HERA-PdM: Uncertainty-Calibrated Escalation in a Hierarchical Edge–Digital-Twin Predictive Maintenance Framework—An Empirical Study on Bearing Run-to-Failure Data**
(The working title was changed because the experiments did not support a framing of HERA-PdM as an improved system; see I.)

## B. Exact research question
Can conformally calibrated RUL intervals computed on an edge device serve as the escalation signal of a hierarchical (edge → fog/cloud) predictive-maintenance system — escalating only when the interval does not determine a unique maintenance action — and what does this cost or gain in RUL accuracy, interval calibration, unsafe/false maintenance decisions, escalation rate, communication and latency, compared with always-edge, always-cloud, point-estimate and alternative gating policies? (Sub-questions RQ1–RQ5 of the brief are answered in §V of the paper.)

## C. Exact novelty claim (as stated in the paper)
“To the best of our literature review, no study uses conformal RUL intervals computed on the edge as the escalation rule of a hierarchical PdM system, relates escalation to maintenance-decision boundaries, and evaluates it against the gating rules of existing systems on public run-to-failure data with communication and latency accounting. The contribution is methodological and incremental; the architecture itself is not new.”
The paper does **not** claim that HERA-PdM outperforms the baselines.

## D. Top 5 closest competing papers
1. Huang, Park, Paoletti, Simeone — *Reliable Inference in Edge-Cloud Model Cascades via Conformal Alignment*, arXiv:2510.17543 (2025, preprint).
2. Saleh, Dinh, Villanyi, Hy — *HAMA: A Hierarchical Adaptive Multi-Agent Architecture for Industrial IoT Predictive Maintenance*, IEEE Access 14 (2026), 10.1109/ACCESS.2026.3731096.
3. Ren, Liu, Wang, Lü, Deen — *Cloud–Edge-Based Lightweight TCN for RUL Prediction in IIoT*, IEEE IoT-J 8(16) (2021), 10.1109/JIOT.2020.3008170.
4. Yao et al. — *EC-Distill-ZeroDiag: cloud-edge collaborative zero-shot industrial fault diagnosis via LLM distillation*, J. Intell. Manuf. (2026), 10.1007/s10845-026-02897-1.
5. Wang, Vidal, Pozo — *Uncertainty-Aware Bearing RUL Prediction Based on Conformal Prediction*, PHME 9(1) (2026), 10.36001/phme.2026.v9i1.4902.
(Also close: Hou et al. arXiv:2503.14453 — conformal edge–cloud offloading for probabilistic solvers; Moccardi et al. 2025 — weighted conformal RUL.)

## E. How HERA-PdM differs from each
1. **Huang et al.**: conformal edge→cloud deferral for *classification prediction sets* in a generic setting; HERA applies conformal deferral to *RUL regression* and defines escalation by *maintenance-decision regions*, with a bound on unsafe edge decisions, evaluated on PdM data.
2. **HAMA**: edge gate is a z-score anomaly pre-filter feeding fog anomaly-detection ensembles; RUL is a point estimate on C-MAPSS; no calibrated intervals. HERA's “anomaly-trigger” baseline reproduces this gating principle.
3. **Ren et al.**: edge and cloud RUL both run; no uncertainty and no routing rule. HERA routes by calibrated uncertainty and adds decision/HIL layers.
4. **EC-Distill-ZeroDiag**: fault *diagnosis*; escalation driven by an LLM-student's self-reflection confidence without coverage guarantee. HERA uses conformal intervals and no LLM.
5. **Wang et al.**: conformal bearing RUL intervals (XJTU-SY) without a decision layer or hierarchy; HERA uses calibration as the control signal of escalation and maintenance actions.

## F. Dataset used
FEMTO-ST / PRONOSTIA bearing run-to-failure data (IEEE PHM 2012 Prognostic Challenge; Nectoux et al. 2012), from the NASA PCoE data repository mirror. 17 bearings (6 Learning_set + 11 Full_Test_Set), 3 operating conditions, 2 accelerometers at 25.6 kHz, 24,889 snapshots; 28 hand-crafted features per snapshot; piecewise-linear RUL capped at 3000 s. (C-MAPSS was downloaded first and then discarded at the user's request to use a different dataset from their prior work.)

## G. Models implemented
* Edge: Edge-CNN (2.7k params), **Edge-GRU (2.5k params, selected)**; window 16.
* Fog/cloud: LSTM, GRU, TCN, CNN-LSTM, each with and without Digital-Twin context, window 64 (13–18k params); **CNN-LSTM+DT selected** on calibration pinball loss.
* Point baselines: the four fog/cloud architectures with DT, trained with MSE.
* Uncertainty: split CQR (primary), asymmetric CQR (variant).
* Anomaly: baseline-standardized health-index score; Isolation Forest baseline.
* Explainability: expected-gradient SHAP (shap.GradientExplainer); template explanation cards (no LLM).
* Gates: decision-sufficiency (HERA), interval width, linear score (anomaly + width + risk), anomaly trigger, point margin, random.

## H. Experiments successfully executed
* Hyper-parameter search on inner validation bearings (40 runs, `results/hparam_search.csv`).
* Main grid: 4 bearing-grouped folds × 3 seeds × 14 configurations = 168 training runs (`results/preds`, `results/checkpoints`).
* RUL accuracy and calibration for all 14 configurations; coverage calibration curves; per-bearing coverage.
* 11 hierarchical policies/ablations with decision, communication, compute and latency metrics.
* Trade-off sweeps: 5 gate families over escalation budgets; α and δ sweeps for edge/cloud/HERA/point policies (decision Pareto).
* Imminent-failure detection (4 detectors), paired Wilcoxon tests across 17 bearings, SHAP on 300 snapshots with explanation cards, CPU latency/MAC profiling.
* Automated paper validation (10 checks, all passing: `results/validation_report.txt`).

## I. Main quantitative findings (mean over 3 seeds, 17 held-out bearings)
* **Accuracy (RQ1):** Edge-GRU RMSE 816 s / MAE 525 s; best fog/cloud model TCN+DT 1047 s; selected CNN-LSTM+DT 1093 s. DT context helped TCN only. → the smallest model was the most accurate.
* **Calibration (RQ2):** raw quantile coverage 0.31–0.69 → CQR 0.83–0.89 (nominal 0.90); CCE ≤ 0.072; MPIW ≈ 1880–2310 s on a 3000 s range; 11/17 bearings below 0.90 under HERA's final intervals.
* **Unsafe decisions (RQ4):** cloud point estimate + fixed thresholds: 41.0 % of critical snapshots left without maintenance; calibrated policies 5.4 % (always-cloud) – 9.4 % (HERA); Wilcoxon p < 0.001 (14 non-zero bearings). Cost: false maintenance 75.0–86.8 % vs 23.6 %. α/δ sweeps show all policies on one safety/false-alarm frontier.
* **Escalation (RQ3/RQ5):** HERA escalated 87.5 % of snapshots at α = 0.1; width gate at matched budget: 9.5 % unsafe / 75.0 % FM vs HERA 9.4 % / 75.0 % (not distinguishable; 2 non-zero bearings). α_e = 0.3 → 74.1 % escalation, 14.1 % unsafe. Anomaly trigger escalated 3.9 % (anomaly AUROC 0.81, F1 0.27; Isolation Forest AUROC 0.77).
* **System cost:** HERA payload 116.6 B/snapshot vs 112 B for always-cloud feature streaming (no saving); mean latency 44.1 ms (assumed 50 ms RTT) vs 50.2 ms always-cloud; edge inference 0.14 ms, fog model 0.25 ms (host CPU).
* **HIL:** 76.7 % of HERA snapshots would trigger approval requests — an unacceptable operator load; SHAP attributions approximate (additivity error 154.8 s).
* **Overall:** the central hypothesis (uncertainty-aware escalation improves the hierarchical trade-off) is **not supported** on this benchmark; failure predictability, not routing, is the bottleneck.

## J. Limitations
One dataset (17 bearings, several abrupt failures); approximate exchangeability with only 3 calibration bearings per fold (coverage below nominal); lightweight state-based rather than physics-based Digital Twin; latency measured on a host CPU (Apple M2 Pro), not embedded hardware; communication counted analytically, network RTT assumed; no energy measurement; simulation over recorded data, not an industrial deployment; no user study of explanation cards; hyper-parameters shared across fog/cloud architectures; methodological changes after data inspection are disclosed in `research/novelty_audit.md` §8.

## K. Remaining unverified claims
* Mendonça et al. (ICIT 2026) is cited only as a “human-centred agentic DT architecture for PdM” on title-level evidence; its internals are not claimed.
* EC-Distill-ZeroDiag's escalation mechanism is taken from an indexed abstract snippet (publisher abstract elided in APIs; full text paywalled).
* Tao et al. (2019) and Xu et al. (2021) are cited for widely known content; their abstracts were not re-read in this session.
* Literature-matrix rows L16, L18, L19 (paywalled 2026 papers) were coded from titles only; one of them could contain a related gate.
* IEEE Xplore and Google Scholar could not be queried directly (blocked); coverage relies on Crossref, Semantic Scholar, arXiv and web search.

## L. Placeholders remaining
* Author: Pradeep Somasundaram. Affiliation and contact e-mail not yet added (IEEE conference template normally expects them).
* No `[EXPERIMENT REQUIRED]` or `[CITATION VERIFICATION REQUIRED]` markers remain.

## M. Current PDF page count
**5 pages** including references (limit 6).

## N. Citation count
**24** references (all verified; see `research/reference_verification.csv`); literature matrix: 42 papers.

## O. Reproducibility status
Complete pipeline in the repository: `python run_experiments.py` (preprocess → hyper-parameter search → training → analysis → figures/tables), `python src/explainability.py`, `cd paper && latexmk -pdf main.tex`, `python experiments/validate_paper.py`. Seeds fixed (0–2), folds and calibration bearings fixed in `src/utils.py`, dependencies pinned in `requirements.txt`, raw data re-downloadable from the NASA PCoE mirror (≈1.2 GB). Full re-training takes ≈40–60 min on a 10-core laptop CPU. Deterministic re-evaluation was confirmed (identical macros after re-running evaluation).
