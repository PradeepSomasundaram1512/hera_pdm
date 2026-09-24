# HERA-PdM — Novelty Audit

Status: **Phase-1 gate passed with a narrowed claim** (the originally proposed claim did *not* survive; see §4).
Final re-search after experiments: see §7.

## 1. Search protocol

* **Automated API search** (`research/lit_search.py`, raw hits in `research/search_raw/initial.json`):
  Crossref (≥2020, 20 hits/query), Semantic Scholar Graph API (2020–2026, 20 hits/query; 12 of 45 requests rate-limited and recorded as errors), arXiv API (15 hits/query) for all 15 query strings in the brief
  → 398 records, 337 unique titles.
* **Targeted web search** for competing system designs (edge–cloud offloading, conformal cascades, agentic DT, HIL-XAI).
* **Metadata verification** (`research/verify_refs.py` → `candidates_verified.json`): every retained paper resolved through its Crossref DOI record, the arXiv API, HAL, or the official NeurIPS proceedings. One Crossref title match was *rejected* as a false positive (a 2025 SSRN paper returned for “Conformalized Quantile Regression”).
* **Not accessible:** IEEE Xplore and Google Scholar block automated access from this environment; Springer/Elsevier full texts are paywalled. Their records were reached through Crossref metadata; several 2026 papers were coded **from abstracts or titles only**, which is recorded per row in `literature_matrix.csv` (column `limitations`, “Basis: …”).
* **Result:** 38 papers coded in the initial audit plus 4 added in the final re-search (§7) = 42 in `research/literature_matrix.csv` (preprints marked as such).

## 2. Which parts of HERA-PdM already exist?

| Component | Already published (examples) |
|---|---|
| Agents + Digital Twin + predictive maintenance | L16, L18, L19, L20, L21, L14, L13 — including human-centred (L19) and safety-constrained (L20) agentic DTs |
| Hierarchical edge/fog/cloud multi-agent PdM | L07 HAMA (IEEE Access 2026), L08 SEMAS, L03, L05 |
| Cloud–edge RUL prediction (edge fast, cloud with history) | L01 (Ren et al., IEEE IoT-J 2021) |
| Conformal / calibrated RUL intervals | L23, L25, L26 (C-MAPSS), **L24 (bearings, XJTU-SY)** |
| Uncertainty-triggered edge→cloud offloading in industrial AI | L02 (LLM self-reflection confidence, fault diagnosis) |
| Conformal edge→cloud cascades with guarantees | L09 (classification, generic), L10 (LLM tiers), theory in L11 |
| Uncertainty-aware maintenance decisions | L30 (risk-aware RL on probabilistic RUL), L31 (point-threshold scheduling) |
| HIL / XAI for PdM, Industry 5.0 | L32–L35, L07 (SHAP + generated text) |

## 3. Comparison table

| Existing study | RUL | DT | Edge | Agents | UQ | Adaptive escalation | XAI | HIL | Gap w.r.t. HERA-PdM |
|---|---|---|---|---|---|---|---|---|---|
| L01 Ren et al. 2021 (IoT-J) | ✓ | – | ✓ | – | – | – (both tiers always run) | – | – | no UQ, no routing rule |
| L02 EC-Distill-ZeroDiag 2026 (JIM) | – | – | ✓ | ✓ | ◐ uncalibrated | ✓ confidence-based | ◐ | – | diagnosis only, no coverage guarantee |
| L03 He et al. 2026 (Engineering) | – | – | ✓ | ✓ | ? | ◐ | ? | ✓ | diagnosis, no calibrated RUL |
| L07 HAMA 2026 (IEEE Access) | ◐ point | – | ✓ | ✓ | – | ◐ anomaly pre-filter | ✓ SHAP | ◐ | gate is anomaly threshold, no RUL interval |
| L08 SEMAS 2026 (preprint) | – | – | ✓ | ✓ | – | ◐ pre-filter | ✓ | ◐ | no RUL, no calibration |
| L09 Huang et al. 2025 (preprint) | – | – | ✓ | – | ✓ conformal | ✓ conformal | – | – | classification sets, not maintenance decisions |
| L10 Conformal Cascade 2026 (preprint) | – | – | – | – | ✓ conformal | ✓ set size | – | – | LLM classification |
| L19 Mendonça et al. 2026 (ICIT) | ? | ✓ | ? | ✓ | ? | ? | ? | ✓ | (title-only coding) |
| L20 SCADT 2026 (ICDSBS) | ? | ✓ | – | ✓ | ◐ | – | – | – | safety filter not statistically calibrated |
| L24 Wang et al. 2026 (PHME) | ✓ | – | – | – | ✓ conformal | – | – | – | no decision layer, no hierarchy |
| L26 Diao et al. 2026 (Sensors) | ✓ | – | – | – | ✓ CQR | – | – | – | no hierarchy / routing |
| L30 Xu & Zhang 2025 (CAIS) | ✓ | – | – | ◐ RL agent | ✓ | – | – | – | UQ drives decisions, not computation |
| **HERA-PdM (this work)** | ✓ | ✓ lightweight state | ✓ | ✓ | ✓ CQR | ✓ decision-sufficiency | ✓ SHAP | ✓ advisory | — |

✓ present, ◐ partial, – absent, ? not determinable from accessible text.

## 4. Critical novelty gate

**Q1. Which parts already exist?** All individual components (§2). The combination “agents + Digital Twin + predictive maintenance” (L16–L21), the three-tier edge/fog/cloud agent hierarchy (L07, L08), conformal RUL intervals for bearings (L24) and uncertainty-triggered edge→cloud offloading (L02, L09) are all published.

**Q2. Which combinations are published?**
* hierarchical agents + SHAP + generated operator text + anomaly-gated escalation (L07);
* conformal prediction sets as the deferral rule of an edge→cloud or multi-tier cascade, for classification (L09, L10);
* calibrated/probabilistic RUL feeding maintenance decisions in a centralized model (L30);
* human-centred agentic DT for PdM (L19, L21, L34).

**Q3. What exact gap remains?** To the best of our review, no study (i) uses *conformally calibrated RUL intervals computed on the edge* as the escalation signal of a hierarchical PdM system, (ii) defines escalation by whether the interval is **decision-sufficient** — i.e. lies inside one maintenance-action region delimited by the critical/planning horizons — which gives a finite-sample bound on unsafe edge-resolved decisions, and (iii) evaluates that rule against the gating rules used in the closest systems (anomaly-threshold pre-filter as in L07/L08, confidence/width deferral as in L02/L11, a linear escalation score, point-estimate margins, and random deferral at matched budget) on public run-to-failure data with communication, latency and compute accounting.

**Q4. What measurable contribution can be defended?**
1. The decision-sufficiency gate and its edge-safety bound: P(local CONTINUE ∧ RUL ≤ H_c) ≤ α_edge (marginal, under exchangeability of calibration and test assets).
2. Empirical evidence, at matched escalation budgets, of whether this gate yields fewer unsafe decisions than uncertainty-width, anomaly-trigger, linear-score and point-margin gates, and what it costs in accuracy/latency/communication relative to always-edge and always-cloud inference.
3. Quantified effect of the Digital Twin long-horizon context and of calibration (raw vs. conformalized quantiles) on RUL accuracy and coverage in a leave-bearings-out protocol.

**Q5. Incremental or substantial?** **Incremental-to-moderate.** The gate adapts the conformal-cascade idea (L09, L10) from classification sets to RUL regression with maintenance decision regions and embeds it in a PdM hierarchy. The architecture is not new. The paper must therefore be framed as a focused methodological contribution with a careful system-level evaluation, not as a new architecture.

### Modifications made because of this audit
* **Dropped:** any claim that combining agents + DT + edge + UQ + HIL is novel, and the phrase “first framework”.
* **Narrowed** the central claim from “uncertainty as an active control variable” (already done by L02, L09–L11) to “**decision-sufficiency** of a calibrated RUL interval as the escalation criterion, with a bound on edge-side unsafe decisions”.
* **Added baselines** that implement the competitors' gating rules (anomaly pre-filter ≈ L07/L08; width/confidence deferral ≈ L02/L11; linear score as proposed in the original brief; random deferral) at an escalation budget matched on calibration bearings.
* **Dataset:** FEMTO/PRONOSTIA bearings (IEEE PHM 2012), rather than the heavily studied C-MAPSS, consistent with the rotating-machinery setting of Industry 5.0 automation.
* The “agents” are software components with defined responsibilities and message interfaces; no LLM is used for prediction or for explanations.

## 5. Final contribution statement (pre-experiment)
> HERA-PdM differs from existing hierarchical PdM architectures by escalating from edge to fog/cloud only when the edge's conformally calibrated RUL interval does not determine a unique maintenance action, which bounds unsafe edge-resolved decisions by the edge miscoverage level; we evaluate this rule against the gating rules of the closest prior systems on public bearing run-to-failure data.

## 6. Residual risks
* Paywalled 2026 papers (L16, L18, L19) were coded from titles only; one of them could contain a similar gate. They are cited as agentic DT PdM work without claims about their internals.
* L09 is the closest methodological work; our novelty rests on the regression/decision-region formulation and the PdM evaluation, not on conformal deferral per se.

## 7. Final novelty re-search (after experiments)

**Queries** (brief's final list): “uncertainty controlled hierarchical predictive maintenance”, “conformal RUL adaptive edge cloud inference”, “uncertainty aware escalation predictive maintenance”, “prediction interval edge cloud predictive maintenance”, “hierarchical multi-agent RUL uncertainty”, “digital twin adaptive inference RUL”, plus “conformal prediction selective offloading edge cloud” and “cascade inference remaining useful life edge”.
Crossref + Semantic Scholar (`search_raw/final.json`, 140 records, 128 unique; 12 S2 requests rate-limited); arXiv queried separately with relaxed term sets because the all-terms AND query returned ~0 hits (`search_raw/final_arxiv*.json`); targeted web searches with the final contribution wording.

**New relevant items found (added as L39–L42):**
* L39 Moccardi et al., *Future Internet* 2025 — conformal RUL on C-MAPSS; recommends **weighted conformal** under non-exchangeability → now cited in Related Work and in the calibration discussion (directly explains our coverage shortfall).
* L40 Kwon & Kim, *Sci. Rep.* 2026 — conformal selective prediction with cost-aware deferral (clinical triage) → confirms decision-aware conformal deferral exists outside PdM (matrix only).
* L41 Hou et al., arXiv 2025 — online conformal calibration with adaptive edge–cloud offloading (probabilistic linear solvers) → cited next to L09 as evidence that conformal edge–cloud offloading is an active general topic.
* L42 Xue et al., arXiv 2026 — risk-controlled device–edge routing for LLMs (matrix only).
* SSRN preprints on stage-adaptive conformal RUL calibration and hierarchical adaptive conformal inference for edge medical wearables (not peer-reviewed; not cited).

**Conclusion.** No retrieved work uses conformal RUL intervals on an edge device as the escalation rule of a hierarchical PdM system with maintenance-decision regions. The narrowed methodological claim survives, but the neighbourhood is crowded (L09, L41, L42), so the paper states the contribution as *incremental* and positions itself explicitly after these works.

**Effect of the experimental results on the claim.** The experiments did **not** support the hypothesis that the decision-sufficiency gate improves the accuracy/safety/communication trade-off: at α = 0.1 it escalated 87.5 % of snapshots and was statistically indistinguishable from a width gate at matched budget, and the tiny edge model was the most accurate predictor. The final paper therefore claims (i) the gate and its edge-safety bound as a method, (ii) the empirical evidence (including the negative result), and **not** superiority of HERA-PdM.

## 8. Methodological changes made after inspecting data (disclosure)
* Anomaly score: the first implementation standardized the smoothed HI by the spread of the *smoothed* baseline, causing alarms on 60–98 % of healthy snapshots; fixed (raw-baseline spread, training-set 99th-percentile threshold) before any reported run.
* Elapsed operating time removed from the DT context (non-transferable shortcut; bearing lives 2.3×10³–2.8×10⁴ s); a threshold-based "time since onset" feature replaced by a threshold-free trend term.
* Regularization (8 epochs, width 32, input noise) chosen on inner validation bearings after the first run showed severe overfitting (raw quantile coverage 2–65 %).
* Mondrian (region-conditional) and asymmetric CQR were explored on fold 0 after the first clean run; neither improved decisions. The pre-planned symmetric split CQR was kept as primary; asymmetric CQR is reported as a variant; Mondrian is not reported in the paper.
* A defect in process management (orphaned workers of the first run writing into the new results folder) was detected via checkpoint metadata; the 10 affected prediction files and one meta file were deleted and regenerated, and all 168 checkpoints and 4 meta files were verified against the final code.


## 9. Version 2 changes (after a strict self-review, 2026-09-24)
* Added XJTU-SY (15 bearings; Wang et al., IEEE Trans. Reliab. 2020) from the dataset author's official Google-Drive mirror, as a gradual-degradation counterpart to FEMTO.
* Per-architecture hyper-parameter tuning (width 32–128, 8/25 epochs, noise) and architecture selection on inner validation bearings only (previously: shared GRU setting, selection on calibration loss).
* Calibration upgraded from split CQR (3 bearings) to CQR-CV+ (Barber et al., Ann. Stat. 2021) using all non-test bearings; implementation verified against brute force (max error 5e-13).
* New analyses: per-bearing first-sustained-alarm timing, horizon sensitivity (5 settings), R_MAX sensitivity (x0.5, x2; seed 0).
* Findings changed as follows: coverage now meets nominal (0.93–0.99); on XJTU-SY the hierarchy improves on always-cloud (same unsafe rate, lower false maintenance and payload), on FEMTO it does not; the decision-sufficiency gate still equals a width gate; calibrated policies raise premature bearing-level alarms. The claim remains incremental; no superiority claim for the gate is made.
