# PAPER_AUDIT — HERA-PdM (version 2)

All numbers below are copied from `results/<dataset>/` (via `paper/numbers.tex`); the paper contains no hand-typed result values.

## A. Final paper title
**HERA-PdM: When Does Uncertainty-Gated Escalation Help Hierarchical Edge–Digital-Twin Predictive Maintenance? Evidence from Two Bearing Datasets** — Pradeep Somasundaram.

## B. Exact research question
When are conformally calibrated RUL intervals, computed on an edge device, a useful escalation signal for hierarchical (edge → fog/cloud) predictive maintenance — escalating only when the interval does not fix a unique maintenance action — and what does this cost or gain in accuracy, calibration, unsafe/false maintenance decisions, bearing-level alarm timing, communication and latency, compared with always-edge, always-cloud, point-estimate and alternative gating policies, under abrupt (FEMTO) and gradual (XJTU-SY) degradation?

## C. Exact novelty claim (paper)
“To the best of our review, no study uses conformal RUL intervals computed on the edge as the escalation rule of a hierarchical PdM system, relates escalation to maintenance-decision boundaries, and evaluates it against the gates of existing systems on public run-to-failure data with different degradation regimes. The contribution is methodological and evaluative; the architecture is not new.” No superiority claim for the gate.

## D. Top 5 closest competing papers
1. Huang, Park, Paoletti, Simeone — conformal-alignment edge–cloud cascades, arXiv:2510.17543 (2025, preprint).
2. Saleh et al. — HAMA, IEEE Access 14 (2026), 10.1109/ACCESS.2026.3731096.
3. Ren et al. — cloud–edge lightweight TCN for RUL, IEEE IoT-J 8(16) (2021), 10.1109/JIOT.2020.3008170.
4. Yao et al. — EC-Distill-ZeroDiag, J. Intell. Manuf. (2026), 10.1007/s10845-026-02897-1.
5. Wang, Vidal, Pozo — conformal bearing RUL, PHME 9(1) (2026), 10.36001/phme.2026.v9i1.4902.
(Also close: Hou et al., arXiv:2503.14453, conformal edge–cloud offloading; Moccardi et al., Future Internet 2025, weighted conformal RUL.)

## E. How HERA-PdM differs from each
1. Huang et al.: classification sets, generic; HERA: RUL regression with maintenance decision regions and a PdM evaluation.
2. HAMA: anomaly-threshold edge pre-filter, point RUL; HERA: calibrated RUL intervals; HAMA's rule is reproduced as the anomaly-trigger baseline.
3. Ren et al.: both tiers always run, no UQ or routing.
4. EC-Distill-ZeroDiag: diagnosis with LLM self-reflection confidence; HERA: conformal intervals, no LLM.
5. Wang et al.: conformal bearing RUL without decision layer or hierarchy.

## F. Datasets used
* FEMTO-ST/PRONOSTIA (IEEE PHM 2012; Nectoux et al. 2012): 17 bearings, 3 conditions, 24,889 snapshots (0.1 s every 10 s); NASA PCoE mirror.
* XJTU-SY (Wang et al., IEEE Trans. Reliab. 2020): 15 bearings, 3 conditions, 9,216 snapshots (1.28 s every minute); author's official Google-Drive mirror.
Labels: RUL capped at 3000 s / 6000 s; horizons H_c = 0.2 R_max, H_p = 0.6 R_max.

## G. Models implemented
Edge: Edge-CNN, Edge-GRU (≈2.5–2.7k params). Fog/cloud: LSTM, GRU, TCN, CNN-LSTM, each ± Digital-Twin context, per-architecture tuned (selected: FEMTO Edge-GRU / LSTM+DT; XJTU-SY Edge-CNN / GRU+DT). MSE point baselines. Calibration: split CQR, asymmetric CQR, CQR-CV+ (K = 4). Anomaly: baseline-standardized health index; Isolation Forest. XAI: expected-gradient SHAP + template explanation cards (no LLM). Gates: decision-sufficiency (HERA), width, linear score, anomaly trigger, point margin, random.

## H. Experiments successfully executed
* Per-architecture tuning: 64 inner-validation runs per dataset.
* Main grids: FEMTO 4 folds × 3 seeds × 14 configs = 168; XJTU-SY 5 × 3 × 14 = 210 training runs.
* CQR-CV+: 3 selected models × 4 inner models × folds × 3 seeds (FEMTO 144, XJTU-SY 180 inner models).
* 11 policies/ablations, trade-off sweeps (gates, α, δ), per-bearing alarm timing, horizon sensitivity (5 settings), R_max sensitivity (×0.5, ×2; seed 0; 4 extra pipelines), detection metrics, Wilcoxon tests, SHAP on both datasets, CPU latency/MACs.
* Automated validation: 10/10 checks pass (`results/validation_report.txt`).

## I. Main quantitative findings (mean over 3 seeds)
* **Accuracy:** FEMTO — edge model best (RMSE 753 s CV+ vs 892 s LSTM+DT); XJTU-SY — fog/cloud best (CNN-LSTM 1335 s vs edge 2278 s, split).
* **Calibration:** split CQR 0.84–0.89 on FEMTO; CQR-CV+ 0.94–0.96 (FEMTO) and 0.93–0.99 (XJTU-SY); per-bearing coverage still as low as 0.30 / 0.07 for single bearings.
* **Unsafe decisions:** point + fixed threshold 49.5 % (FEMTO) / 34.2 % (XJTU-SY) vs HERA 0.6 % / 9.2 % (Wilcoxon p < 0.001 / p = 0.004); false maintenance 95.0 % / 53.2 % vs 21.7 % / 8.3 %.
* **Escalation:** FEMTO — HERA escalated 97.1 %, no benefit, 123 vs 112 B/snapshot. XJTU-SY — escalated 79.1 %, same unsafe rate as always-cloud (9.2 %), lower FM (53.2 vs 69.1 %), RMSE 1381 vs 1421 s, 104.5 vs 112 B/snapshot; but per-bearing FM differs from always-cloud in only 2 bearings, and edge-only CQR traces a better frontier. Width gate at matched budget: identical (9.2 % / 53.3 %).
* **Per-bearing timing:** calibrated policies raise premature sustained alarms for 16/17 (FEMTO) and 13/15 (XJTU-SY) bearings; the point policy is late for 5.3 / 4.0 bearings.
* **Sensitivity:** in all 5 horizon settings and 4 R_max runs, HERA had fewer unsafe decisions than the point policy; with a halved cap on XJTU-SY HERA reached 6 % unsafe / 21 % FM vs always-cloud 6 % / 45 %.
* **XAI/HIL:** SHAP additivity error 16 % (FEMTO) / 3 % (XJTU-SY); approval requests on 95.5 % / 55.8 % of snapshots.
* **Overall:** calibration is clearly useful for safety; uncertainty-gated escalation helps only under gradual degradation; the proposed gate is not better than a width gate.

## J. Limitations
Two datasets (32 bearings); approximate exchangeability; lightweight state-based DT; host-CPU latency, analytic communication, no energy; R_max sensitivity with one seed; the XJTU-SY advantage over always-cloud is concentrated in two long-lived bearings; no deployment; no user study; post-hoc design changes disclosed in `research/novelty_audit.md` §8–§9.

## K. Remaining unverified claims
* Mendonça et al. (ICIT 2026) cited on title-level evidence only.
* EC-Distill-ZeroDiag's mechanism from an indexed abstract snippet (full text paywalled).
* Tao et al. (2019) and Xu et al. (2021) cited for widely known content; abstracts not re-read.
* Paywalled 2026 papers L16, L18, L19 coded from titles; IEEE Xplore/Google Scholar not directly searchable here.

## L. Placeholders remaining
Author affiliation and e-mail not provided (name only). No `[EXPERIMENT REQUIRED]` / `[CITATION VERIFICATION REQUIRED]` markers.

## M. Current PDF page count
**6 pages** including references (limit 6); references end on page 6 with no floats after them.

## N. Citation count
**26** references, all verified (`research/reference_verification.csv`); literature matrix: 42 papers.

## O. Reproducibility status
`experiments/run_all_v2.sh` (tuning → training → CV+ for both datasets → R_max sensitivity), then per dataset `HERA_DATASET=<ds> python src/evaluation.py` and `python src/explainability.py`, then `python experiments/make_figures.py`, `python experiments/make_tables.py`, the LaTeX build, and `python experiments/validate_paper.py`. Seeds fixed; folds, calibration groups and inner-validation bearings fixed in `src/utils.py`; dependencies pinned. Compute ≈ 3–4 h on a 10-core laptop CPU (keep the machine awake).
