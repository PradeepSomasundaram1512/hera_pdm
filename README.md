# HERA-PdM — When does uncertainty-gated edge–cloud predictive maintenance help?

Code, results and paper for *"When Does Uncertainty-Gated Edge–Cloud Predictive Maintenance Help? Evidence from Bearings and a Truck Fleet"*. HERA-PdM (Hierarchical Edge Reasoning Agents) is used as an experimental framework, not as a claimed best system.

Research code, data pipeline, experiment outputs and IEEE conference manuscript for
**“HERA-PdM: When Does Uncertainty-Gated Escalation Help Hierarchical Edge–Digital-Twin Predictive Maintenance? Evidence from Two Bearing Datasets”** (v2: FEMTO + XJTU-SY, CQR-CV+). Run everything with `experiments/run_all_v2.sh`; set `HERA_DATASET=femto|xjtu` for single stages.

**Headline result (v2):** calibration cuts unsafe point-threshold decisions (49.5 % → 0.6 % FEMTO; 34.2 % → 9.2 % XJTU-SY); uncertainty-gated escalation beats always-cloud only on gradually degrading XJTU-SY bearings; the decision-sufficiency gate equals a width gate; bearing-level alarms are premature. See `PAPER_AUDIT.md`. XJTU-SY: author mirror https://drive.google.com/open?id=1_ycmG46PARiykt82ShfnFfyQsaXv3_VK (6 RAR parts; extract with `unar`) into `data/raw/XJTU`, then `python src/preprocessing.py --dataset xjtu`.

## Research objective
Hierarchical (edge → fog/cloud) predictive maintenance usually escalates on anomaly
thresholds or heuristic confidence. We study whether a **conformally calibrated RUL
interval computed on the edge** can serve as the escalation signal: the edge resolves a
snapshot locally only if its interval is *decision-sufficient* (lies inside one
maintenance-action region), which bounds unsafe edge-resolved decisions by the edge
miscoverage level. The gate is compared with anomaly-trigger, width/confidence,
linear-score, point-margin and random gates at matched escalation budgets, with
accuracy, calibration, decision-safety, communication, latency and compute reported.
See `research/novelty_audit.md` for the literature audit behind this framing.

## Dataset
FEMTO-ST / PRONOSTIA bearing run-to-failure data (IEEE PHM 2012 Prognostic Challenge;
Nectoux et al., PHM'12), obtained from the NASA PCoE data repository mirror
`https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip` (FEMTOBearingDataSet.zip:
Learning_set, Test_set, Full_Test_Set).

| | |
|---|---|
| Units | 17 bearings run to failure (6 `Learning_set` + 11 `Full_Test_Set`) |
| Conditions | 1: 1800 rpm / 4000 N (7 bearings), 2: 1650 rpm / 4200 N (7), 3: 1500 rpm / 5000 N (3) |
| Sensors | horizontal + vertical accelerometers, 25.6 kHz, 0.1 s snapshot every 10 s |
| Features | 14 per channel (RMS, kurtosis, skewness, peak-to-peak, crest, impulse, shape, spectral centroid, 6 log band energies) → 28 |
| Normalization | per-asset z-score against its first 50 snapshots (healthy baseline) + signed log |
| Label | piecewise-linear RUL = min(time to last snapshot, 3000 s) |
| Split | 4 bearing-grouped folds (every bearing tested once); per fold 1 calibration bearing per condition; remaining bearings train |
| Temperature files | not used (removed after extraction) |

`data/raw/` is not versioned (≈3 GB); `data/processed/` (features, split summary) is.

## Layout
```
hera_pdm/
├── data/processed/           # extracted features, dataset summary, challenge truncation points
├── src/
│   ├── preprocessing.py      # FEMTO feature extraction
│   ├── digital_twin.py       # Digital Twin State Agent (baseline normalization, HI, context c_t, DT_t record)
│   ├── edge_agent.py         # Edge Monitoring Agent, Isolation-Forest baseline, detection metrics
│   ├── prognostics.py        # RUL models (LSTM/GRU/TCN/CNN-LSTM, Edge-CNN/GRU), training, latency/MACs
│   ├── uncertainty.py        # split CQR, asymmetric CQR, coverage metrics
│   ├── escalation.py         # decision regions, HERA gate, baseline gates, decision metrics
│   ├── explainability.py     # SHAP (GradientExplainer) + explanation cards (no LLM)
│   ├── evaluation.py         # all metrics, policies, ablations, trade-off curves, tests
│   └── utils.py              # constants, folds, seeds
├── experiments/              # tune_hparams.py, make_figures.py, make_tables.py, validate_paper.py
├── results/                  # CSV/JSON outputs (every paper number comes from here)
├── figures/                  # vector figures
├── paper/                    # main.tex, references.bib, tables/, numbers.tex, main.pdf
├── research/                 # literature search, matrix, novelty audit, reference verification
├── run_experiments.py
└── requirements.txt
```

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.13.5, PyTorch 2.14 (CPU), scikit-learn 1.9.1, SHAP 0.52 on macOS (Apple M2 Pro).

## Reproduction
```bash
# 1. data (≈1.2 GB download, ≈3 GB extracted)
mkdir -p data/raw && cd data/raw
curl -L -o femto.zip "https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip"
unzip femto.zip && unzip "10. FEMTO Bearing/FEMTOBearingDataSet.zip" -d femto_outer
mkdir FEMTO && for z in Training_set Test_set Validation_Set; do unzip -o femto_outer/$z.zip -d FEMTO; done
cd ../..
# 2. preprocessing
python src/preprocessing.py
# 3. hyper-parameter selection (inner splits of training bearings only) + training (4 folds × 3 seeds × 14 configs)
python run_experiments.py --stage train --workers 5
# 4. evaluation (all metrics, policies, ablations, statistics)
python run_experiments.py --stage analyze
python src/explainability.py
# 4b. XJTU-SY (HERA_DATASET=xjtu), CV+ stage, R_max sensitivity, SCANIA fleet study
HERA_DATASET=xjtu python run_experiments.py --workers 5
python run_experiments.py --stage cvplus --workers 5; HERA_DATASET=xjtu python run_experiments.py --stage cvplus --workers 5
python experiments/rmax_sensitivity.py
python src/fleet_scania.py                              # needs data/raw/SCANIA (DOI 10.5878/bnh5-ka77)
# 4c. robustness: five seeds and runner-up architectures (≈2 h on an M2 Pro, FEMTO TCN runs dominate)
python experiments/robustness_seeds_arch.py train && python experiments/robustness_seeds_arch.py analyze_all
# 5. figures and LaTeX tables / numeric macros
python run_experiments.py --stage figures
# 6. paper
cd paper && latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
cd .. && python experiments/validate_paper.py
```
Or everything in sequence: `python run_experiments.py` (then `python src/explainability.py`, `python experiments/make_figures.py`, `python experiments/make_tables.py` and the paper build).
`results/_discarded_v0/` (not versioned) held outputs of a first run with a defective anomaly score; they are not used anywhere.
`IEEEtran.cls`/`IEEEtran.bst` (v1.8b, CTAN) are bundled in `paper/`.

## Reproducibility notes
* Seeds 0, 1, 2 (`utils.SEEDS`) control initialization and batch order; fold, calibration and CV+ assignments are fixed lists in `utils.py`, shared by all seeds. Seeds 3-4 are used only by the five-seed robustness check.
* Hyper-parameters and architectures were chosen once (seed 0) on two inner validation bearings from the training bearings of two folds (`results/<ds>/hparam_search.csv`, `selected_models.json`) and frozen; test bearings were never used. The protocol was revised once after a first run (see `research/novelty_audit.md` §8-9).
* Statistics: unit = bearing (seeds averaged per bearing), two-sided Wilcoxon signed-rank, zero differences dropped, no test with <5 non-zero pairs; Holm correction over the five primary tests.
* Latency is CPU wall-clock (1 thread, batch 1) on the host machine, **not** on embedded hardware; network RTT (50 ms) and communication volumes are analytic assumptions, labelled as such. No energy measurements were made.
* The system is a simulation over recorded data, not an industrial deployment.
