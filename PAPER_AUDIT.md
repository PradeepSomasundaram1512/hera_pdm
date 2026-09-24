# PAPER_AUDIT — HERA-PdM (version 4, evidence-calibrated)

## A. Final paper title
When Does Uncertainty-Gated Edge–Cloud Predictive Maintenance Help? Evidence from Bearings and a Truck Fleet

## B. Research question
Under which conditions (degradation observability, calibration sample size, decision structure) does conformal, uncertainty-gated edge-to-cloud escalation improve PdM decisions over point thresholds, simpler gates, and calibrated edge-only or cloud-only inference?

## C. Novelty claim
Evaluation, not architecture: no study found (structured search, 42 coded papers) that uses conformal RUL intervals computed on the edge as the escalation rule of a hierarchical PdM system and tests it against simpler gates across degradation regimes; plus a finite-sample feasibility condition for unit-level conformal risk control contrasted between bearings and a fleet.

## D-E. Closest work
Ren et al. 2021 (edge/cloud RUL, no UQ routing); HAMA 2026 (anomaly-triggered escalation); EC-Distill-ZeroDiag 2026 (uncalibrated confidence offloading); CAb cascade 2025 (conformal edge-cloud cascades, classification); Wang et al. 2026 (conformal bearing RUL, no routing); Jitkrittum et al. 2023 (when confidence deferral suffices).

## F. Data
FEMTO/PRONOSTIA (17 bearings, abrupt), XJTU-SY (15 bearings, gradual), SCANIA Component X (23,550 trucks, 2,272 failing; unit-level risk-control test only).

## G-H. Models and experiments
Edge-GRU/Edge-CNN; LSTM/GRU/TCN/CNN-LSTM ± state context; MSE point models; CQR-CV+; 11 policies/gates at matched budgets; horizon and R_max sensitivity; CRC alarm timing; SCANIA CRC (logistic regression / gradient boosting); five-seed and runner-up-architecture robustness; Holm-corrected primary tests.

## I. Main findings
* Calibration reduces unsafe decisions vs point thresholds (bearing-level Wilcoxon, Holm max p 0.004) but raises false maintenance; frontier nearly unchanged.
* Pooled coverage 0.93-0.99, but 4/17 FEMTO and 8/15 XJTU-SY bearings below 0.90 (min 0.30 / 0.08).
* Hierarchy helps only under gradual degradation (XJTU-SY: lower FM than edge-only on 12/13 bearings, p<0.001, at a higher unsafe rate on 4); on FEMTO it escalates 97 % and costs more bandwidth than streaming.
* Width gate: same unsafe rate on every bearing; FM differs on few bearings by <=0.5 pp.
* Unit-level CRC infeasible with ~12 bearings (Proposition 1); meets target on SCANIA, where calibrated edge-only nearly matches HERA at zero transmission.
* Approval load 95.5 % / 55.8 % of HERA snapshots (not user-validated).

## J. Limitations
32 bearings; 3 seeds (+ one 5-seed check); approximate exchangeability, heterogeneous per-bearing coverage; lightweight state-based DT; analytic communication; no embedded timing, deployment or operator study; SCANIA tests risk-control scaling only; protocol revised once after a first run (disclosed).

## K. Unverified claims
None known. Container timings are labelled portability-only; no embedded-hardware numbers are claimed.

## L. Placeholders
None (validator check 5).

## M. Page count
6 pages including references (IEEEtran conference).

## N. Citations
25, all verified (research/reference_verification.csv).

## O. Reproducibility
All numbers generated as macros from results/ (validator check 9); commands in README; coding matrix and search logs in research/; repository public at https://github.com/PradeepSomasundaram1512/hera_pdm (cited in the paper).
