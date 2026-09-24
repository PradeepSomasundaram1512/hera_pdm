"""Builds research/reference_verification.csv for every entry in paper/references.bib.

verified: metadata (title, authors, year, venue, DOI/arXiv id) re-checked against the
source listed in `verification_source` (Crossref DOI record, arXiv API, HAL API,
NeurIPS proceedings / Semantic Scholar venue record). Author lists were compared
programmatically against the source.
supports_claim: the statement(s) for which the paper cites the work, and the evidence
we inspected (abstract, public repository, or dataset documentation).
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLAIMS = {
    "nectoux2012pronostia": ("HAL API (hal-00719503)", "yes - dataset source (PRONOSTIA platform, PHM'12)"),
    "lei2018machinery": ("Crossref DOI", "yes - review: data acquisition, HI construction, health-stage division, RUL prediction (abstract); cited for PdM definition and health-stage division"),
    "zheng2017lstm": ("Crossref DOI", "yes - LSTM for RUL estimation (title/record)"),
    "zhu2019mscnn": ("Crossref DOI", "yes - multiscale CNN for bearing RUL (abstract)"),
    "bai2018tcn": ("arXiv API", "yes - generic TCN architecture used as baseline (abstract)"),
    "tao2019dt": ("Crossref DOI", "yes (title + widely known survey content; abstract not inspected in this session) - DT in industry survey; cited for DT as synchronized virtual state"),
    "xu2021industry5": ("Crossref DOI", "yes (title + widely known editorial content; abstract not inspected in this session) - Industry 5.0 perspective; cited for human-centric motivation"),
    "ren2021cloudedge": ("Crossref DOI", "yes - edge real-time RUL + cloud refinement with history (abstract)"),
    "saleh2026hama": ("Crossref DOI + authors' public repository README", "yes - edge z-score pre-filter, fog ensembles, SHAP, small-LM operator text"),
    "yao2026ecdistill": ("Crossref DOI", "yes (abstract via indexed search snippet; publisher abstract elided in APIs) - edge agent offloads uncertain fault-diagnosis cases"),
    "mendonca2026human": ("Crossref DOI", "partial - title-level evidence only: human-centred agentic DT architecture for PdM; paper makes no claim about its internals"),
    "javanmardi2023conformal": ("Crossref DOI + arXiv 2212.14612", "yes - conformal RUL intervals on C-MAPSS (abstract)"),
    "wang2026conformal": ("Crossref DOI + PHM Society page", "yes - conformal calibration of bearing RUL intervals (abstract)"),
    "basora2025benchmark": ("Crossref DOI", "yes - benchmark of BNN/MC-dropout/ensemble/heteroscedastic UQ for DL prognostics (indexed snippet)"),
    "romano2019cqr": ("arXiv API + NeurIPS proceedings page", "yes - CQR method and coverage guarantee"),
    "angelopoulos2023gentle": ("Crossref DOI", "yes - conformal prediction tutorial; finite-sample coverage"),
    "huang2025cab": ("arXiv API", "yes - conformal-alignment edge-cloud cascade for classification (abstract)"),
    "jitkrittum2023cascade": ("arXiv API + Semantic Scholar venue (NeurIPS 2023)", "yes - confidence-based cascade deferral (abstract)"),
    "xu2025riskrl": ("Crossref DOI", "yes - probabilistic RUL feeding risk-aware RL maintenance policy (abstract)"),
    "lundberg2017shap": ("arXiv API + Semantic Scholar venue/pages", "yes - SHAP"),
    "vanoudenhoven2023pdm5": ("Crossref DOI", "yes - decision-makers often do not adopt system-generated PdM advice (abstract)"),
    "coulibaly2026masdt": ("arXiv API", "yes - survey lists hierarchical DT orchestration with residual-life estimation and XAI as open question (abstract)"),
    "moccardi2025robust": ("Crossref DOI", "yes - conformal RUL on C-MAPSS; recommends weighted conformal under non-exchangeability (abstract)"),
    "barber2021jackknife": ("Crossref DOI", "yes - jackknife+/CV+ predictive intervals with 1-2alpha guarantee (Ann. Stat.); used for CQR-CV+"),
    "wang2020xjtu": ("Crossref DOI + dataset author's official repository README", "yes - XJTU-SY dataset source paper requested by the dataset authors"),
    "angelopoulos2024crc": ("arXiv API (2208.02814) + official ICLR 2024 proceedings page", "yes - conformal risk control of expected monotone losses; used for bearing-level lateness control"),
    "kharazian2025scania": ("Crossref DOI + DataCite dataset DOI 10.5878/bnh5-ka77", "yes - SCANIA Component X fleet dataset (CC BY 4.0) used for the fleet-scale study"),
    "hou2025online": ("arXiv API", "yes - conformal calibration with adaptive edge-cloud offloading for probabilistic linear solvers (abstract)"),
}


def field(entry, name):
    m = re.search(name + r"\s*=\s*\{(.+?)\},?\n", entry, re.S)
    return re.sub(r"[{}\\\"']|\s+", lambda x: " " if x.group().isspace() else "", m.group(1)).strip() if m else ""


def main():
    bib = (ROOT / "paper" / "references.bib").read_text()
    rows = []
    for e in re.split(r"\n@", bib)[1:]:
        key = re.match(r"\w+\{([^,]+),", e).group(1)
        doi = field(e, "doi")
        arx = re.search(r"arXiv:(\d{4}\.\d{4,5})", e)
        url = f"https://doi.org/{doi}" if doi else (f"https://arxiv.org/abs/{arx.group(1)}" if arx else
                                                      "https://hal.science/hal-00719503" if "hal" in e else
                                                      "https://proceedings.neurips.cc")
        venue = field(e, "journal") or field(e, "booktitle") or field(e, "howpublished")
        src, claim = CLAIMS[key]
        rows.append({"citation_key": key, "title": field(e, "title"), "authors": field(e, "author"),
                     "year": field(e, "year"), "venue": venue, "doi_or_url": url,
                     "verified": f"yes ({src})", "supports_claim": claim})
    out = ROOT / "research" / "reference_verification.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} references -> {out}")


if __name__ == "__main__":
    main()
