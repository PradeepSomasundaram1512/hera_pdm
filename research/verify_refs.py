"""Resolve candidate papers to verified metadata (Crossref / arXiv) and fetch
abstracts (Semantic Scholar batch API). Output: research/candidates_verified.json

Each candidate is either a DOI, an arXiv id ("arXiv:XXXX.XXXXX") or an exact
title (resolved by Crossref bibliographic search; accepted only if the returned
title matches closely).
"""
import difflib
import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

H = {"User-Agent": "HERA-PdM-reference-verification/1.0"}
OUT = Path(__file__).parent / "candidates_verified.json"

CANDIDATES = [
    # --- closest system-level work (edge/cloud, agents, DT) ---
    "10.1109/jiot.2020.3008170",
    "10.1007/s10845-026-02897-1",
    "10.1016/j.eng.2026.02.029",
    "10.1007/978-3-032-11946-9_13",
    "10.1007/s00607-025-01585-x",
    "Distributed Predictive Maintenance Through Edge Computing",
    "arXiv:2510.17543",
    "arXiv:2607.25018",
    "arXiv:2307.02764",
    "arXiv:2607.21873",
    "arXiv:2608.11679",
    "arXiv:2505.02076",
    "arXiv:2507.01376",
    "10.1007/s43069-025-00579-x",
    "Multi-Agent Systems for Manufacturing Digital Twins: A Perspective on Agency and Large Language Models",
    "Prediction of bearing remaining useful life based on a two-stage updated digital twin",
    # --- UQ / conformal RUL ---
    "Conformal Prediction Intervals for Remaining Useful Lifetime Estimation",
    "10.36001/phme.2026.v9i1.4902",
    "A General Data-Driven Framework for Remaining Useful Life Estimation with Uncertainty Quantification Using Split Conformal Prediction",
    "10.3390/s26072249",
    "A benchmark on uncertainty quantification for deep learning prognostics",
    "Uncertainty-aware remaining useful life prediction for predictive maintenance using deep learning",
    "Robust uncertainty quantification for online remaining useful life prediction with randomly missing and partially faulty sensor data",
    "Predictive maintenance optimization for industrial equipment via reliable prognosis and risk-aware reinforcement learning",
    "Interpretable ensemble remaining useful life prediction enables dynamic maintenance scheduling for aircraft engines",
    # --- human-centric / XAI ---
    "10.3390/electronics14173384",
    "10.1080/00207543.2022.2154403",
    "From predictive maintenance 4.0 to 5.0: bringing humans back into the loop with a self-learning platform and implementation roadmap on automated production lines",
    "Enhanced Predictive Maintenance through Explainable Multimodal Deep Learning and Human-in-the-Loop Systems",
    "Technician 5.0: A Hybrid Framework Integrating Chatbot, Digital Twin, and Machine Learning for Human-Centric Predictive Maintenance in Industry 5.0",
    "arXiv:2306.05120",
    # --- foundational ---
    "PRONOSTIA: An experimental platform for bearings accelerated degradation tests",
    "Conformalized Quantile Regression",
    "Conformal Prediction: A Gentle Introduction",
    "Adaptive Conformal Inference Under Distribution Shift",
    "Industry 4.0 and Industry 5.0—Inception, conception and perception",
    "Digital Twin in Industry: State-of-the-Art",
    "Machinery health prognostics: A systematic review from data acquisition to RUL prediction",
    "A Unified Approach to Interpreting Model Predictions",
    "arXiv:1803.01271",
    "Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge",
    "BranchyNet: Fast inference via early exiting from deep neural networks",
    "Estimation of Bearing Remaining Useful Life Based on Multiscale Convolutional Neural Network",
    "Remaining useful life estimation in prognostics using deep convolution neural networks",
    "Long Short-Term Memory Network for Remaining Useful Life estimation",
    "Industry 5.0: Towards a sustainable, human-centric and resilient European industry",
]


def crossref_doi(doi):
    r = requests.get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", headers=H, timeout=60)
    if r.status_code != 200:
        return None
    return fmt(r.json()["message"])


def fmt(m):
    return {"doi": m.get("DOI"), "title": (m.get("title") or [""])[0],
            "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() or a.get("name", "")
                        for a in m.get("author", [])],
            "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "venue": "; ".join(m.get("container-title") or []) or m.get("publisher"),
            "volume": m.get("volume"), "issue": m.get("issue"), "pages": m.get("page"),
            "type": m.get("type"), "publisher": m.get("publisher")}


def crossref_title(title):
    r = requests.get("https://api.crossref.org/works", headers=H, timeout=60,
                     params={"query.bibliographic": title, "rows": 5})
    best, score = None, 0
    for m in r.json()["message"]["items"]:
        t = (m.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, t.lower(), title.lower()).ratio()
        if s > score:
            best, score = m, s
    if best is not None and score >= 0.85:
        return fmt(best) | {"match_score": round(score, 3)}
    return None


def arxiv_meta(aid):
    r = requests.get("http://export.arxiv.org/api/query", params={"id_list": aid}, headers=H, timeout=60)
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(r.text).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return None
    doi = e.find("arxiv:doi", ns)
    jr = e.find("arxiv:journal_ref", ns)
    return {"arxiv": aid, "title": " ".join(e.find("a:title", ns).text.split()),
            "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
            "year": int(e.find("a:published", ns).text[:4]), "venue": "arXiv preprint",
            "published_doi": doi.text if doi is not None else None,
            "journal_ref": jr.text if jr is not None else None,
            "abstract": " ".join(e.find("a:summary", ns).text.split())}


def s2_batch(ids):
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        for attempt in range(6):
            r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                              params={"fields": "title,abstract,year,venue,citationCount,externalIds"},
                              json={"ids": chunk}, headers=H, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1)); continue
            break
        if r.status_code == 200:
            for k, v in zip(chunk, r.json()):
                out[k] = v
    return out


def main():
    res = []
    for c in CANDIDATES:
        rec = None
        try:
            if c.lower().startswith("arxiv:"):
                rec = arxiv_meta(c.split(":", 1)[1])
            elif c.startswith("10."):
                rec = crossref_doi(c)
            else:
                rec = crossref_title(c)
        except Exception as ex:
            rec = {"error": str(ex)[:200]}
        res.append({"query": c, "verified": bool(rec and "error" not in rec), **(rec or {})})
        print(("OK  " if res[-1]["verified"] else "FAIL") + f" {c[:90]}", flush=True)
        time.sleep(0.5)
    ids = [f"DOI:{r['doi']}" if r.get("doi") else f"ARXIV:{r['arxiv']}" for r in res if r["verified"]]
    s2 = s2_batch(ids)
    for r in res:
        if not r["verified"]:
            continue
        k = f"DOI:{r['doi']}" if r.get("doi") else f"ARXIV:{r['arxiv']}"
        v = s2.get(k) or {}
        if not r.get("abstract"):
            r["abstract"] = (v or {}).get("abstract")
        r["citations"] = (v or {}).get("citationCount")
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
