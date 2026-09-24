# When does uncertainty-gated edge–cloud predictive maintenance help?

Code, data pipeline, results and manuscript for

> P. Somasundaram, *When Does Uncertainty-Gated Edge–Cloud Predictive Maintenance Help? Evidence from Bearings and a Truck Fleet*, submitted to ICAIET 2027.

HERA-PdM (Hierarchical Edge Reasoning Agents) is the experimental framework used in the paper, not a claimed best system. Conformalized quantile regression with CV+ (CQR-CV+) gives remaining-useful-life (RUL) intervals, and the edge escalates to a fog/cloud model only when its interval spans a maintenance-decision boundary. The study asks when that is worth doing.

**Main findings**
* Calibrated decisions leave far fewer critical snapshots without action than point thresholds (0.6 % vs. 49.5 % on FEMTO, 9.2 % vs. 34.2 % on XJTU-SY) but raise false maintenance.
* Pooled coverage is close to nominal, yet 8 of 15 XJTU-SY bearings fall below 0.90.
* Escalation helps only under gradual degradation (XJTU-SY). Under abrupt failures (FEMTO) the edge escalates almost everything.
* A simple interval-width gate gives the same unsafe rate as the decision-aware gate on every bearing.
* Unit-level conformal risk control is infeasible with about 12 bearings per fold but works on 23,550 Scania trucks, where calibrated edge-only inference nearly matches the hierarchy.

Every number in the paper is generated from `results/` by `manuscript/tools/make_tables.py`.

## Repository layout
```
hera_pdm/
├── README.md, requirements.txt
├── run_experiments.py              entry point: preprocess, tune, train, cvplus, analyze, figures
├── src/                            library code
│   ├── preprocessing.py            FEMTO and XJTU-SY feature extraction (28 features)
│   ├── digital_twin.py             state-based twin record: baseline normalization, health index, context c_t
│   ├── edge_agent.py               anomaly score, Isolation-Forest baseline, detection metrics
│   ├── prognostics.py              quantile RUL models (LSTM, GRU, TCN, CNN-LSTM, Edge-CNN/GRU), MACs, latency
│   ├── uncertainty.py              split CQR, CQR-CV+, coverage metrics
│   ├── escalation.py               decision regions, HERA gate, baseline gates, decision metrics
│   ├── evaluation.py               policies, ablations, trade-offs, per-bearing statistics
│   ├── explainability.py           SHAP attributions and explanation cards (no LLM)
│   ├── alarm_crc.py, evaluate_alarms.py   bearing-level conformal risk control for alarm timing
│   ├── fleet_scania.py             SCANIA fleet study (unit-level risk control)
│   └── utils.py                    dataset configs, folds, seeds, paths
├── experiments/                    analysis drivers
│   ├── tune_hparams.py             per-architecture selection on inner validation bearings
│   ├── rmax_sensitivity.py         R_max halved/doubled
│   ├── bootstrap_offsets.py        bearing-level bootstrap CIs, required alarm offsets
│   ├── robustness_seeds_arch.py    five seeds and runner-up architectures
│   └── run_all.sh                  full compute pipeline for both bearing datasets
├── manuscript/
│   ├── icaiet2027/                 main.tex, references.bib, numbers.tex, tables/, figures/,
│   │                               ICAIET2027_Somasundaram_EdgeCloud_PdM.pdf (submitted PDF)
│   ├── design_note/                2-page design note (HERA-PdM_design_note.pdf)
│   └── tools/                      make_figures.py, make_tables.py, validate_paper.py
├── docs/
│   ├── PAPER_AUDIT.md              claims, evidence, limitations
│   └── literature/                 search scripts and raw hits, 42-paper coding matrix,
│                                   novelty audit, reference verification
├── edge_kit/                       exported models, on-device benchmark, parity check, container runs
├── data/processed/                 extracted features (raw data are not versioned)
└── results/                        per-dataset outputs (femto, xjtu, *_rmax*, scania), logs/
```

## Data
| Dataset | Source | Place raw data in |
|---|---|---|
| FEMTO-ST/PRONOSTIA (17 bearings) | NASA PCoE mirror `https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip` | `data/raw/FEMTO` |
| XJTU-SY (15 bearings) | authors' mirror `https://drive.google.com/open?id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK` (6 RAR parts, extract with `unar`) | `data/raw/XJTU` |
| SCANIA Component X (23,550 trucks) | DOI 10.5878/bnh5-ka77, CC BY 4.0 | `data/raw/SCANIA` |

Features: 14 per accelerometer channel (RMS, kurtosis, skewness, peak-to-peak, crest, impulse and shape factors, spectral centroid, six log band energies), z-scored against each asset's first snapshots and log-compressed.

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.13.5, PyTorch 2.14 (CPU), scikit-learn 1.9.1 and SHAP 0.52 on macOS (Apple M2 Pro).

## Reproduction
```bash
# 1. features (after placing the raw data as above)
python src/preprocessing.py --dataset femto
python src/preprocessing.py --dataset xjtu

# 2. tuning, training, CV+ calibration for both bearing datasets, then R_max sensitivity
bash experiments/run_all.sh

# 3. evaluation, explanations, bootstrap CIs, alarm-timing risk control
HERA_DATASET=femto python run_experiments.py --stage analyze
HERA_DATASET=xjtu  python run_experiments.py --stage analyze
python src/explainability.py
python experiments/bootstrap_offsets.py
HERA_DATASET=femto python src/evaluate_alarms.py; HERA_DATASET=xjtu python src/evaluate_alarms.py

# 4. fleet study and robustness checks (the FEMTO TCN runs take about 2 h on an M2 Pro)
python src/fleet_scania.py
python experiments/robustness_seeds_arch.py train
python experiments/robustness_seeds_arch.py analyze_all

# 5. figures, tables and numeric macros, then the paper and its checks
python manuscript/tools/make_figures.py
python manuscript/tools/make_tables.py
cd manuscript/icaiet2027 && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd ../..
python manuscript/tools/validate_paper.py
```
`IEEEtran.cls`/`IEEEtran.bst` (v1.8b, CTAN) are bundled with each manuscript.

## Reproducibility notes
* Seeds 0, 1, 2 control initialization, batch order and input noise; seeds 3 and 4 are used only by the five-seed check. Fold, calibration and CV+ assignments are fixed lists in `src/utils.py`, shared by all seeds.
* Hyper-parameters and architectures were chosen once (seed 0) on two inner validation bearings from the training bearings of two folds (`results/<dataset>/hparam_search.csv`, `selected_models.json`) and then frozen. Test bearings were never used. The protocol was revised once after a first run; see `docs/literature/novelty_audit.md`, sections 8 to 12.
* Statistics: the unit is the bearing (seed-averaged per bearing), with two-sided Wilcoxon signed-rank tests, zero differences dropped, no test below five non-zero pairs, and Holm correction over the five primary tests.
* Latency is single-thread CPU time on the host, not on embedded hardware. The 50 ms RTT and the communication volumes are analytic. No energy was measured.
* This is an offline study on recorded data, not an industrial deployment.
* `archive_old_versions/fig_escalation_v1.pdf` is the escalation diagram from the first manuscript version; it is kept for history and is not used by either paper.
