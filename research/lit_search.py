"""Systematic literature search against public scholarly APIs.

Queries Crossref, Semantic Scholar and arXiv for every search string used in the
novelty audit and stores the raw hits so the audit is reproducible.
Usage: python research/lit_search.py [initial|final]
"""
import json
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

OUT = Path(__file__).parent / "search_raw"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "HERA-PdM-literature-audit/1.0 (mailto:research@example.org)"}

INITIAL = [
    "agentic AI predictive maintenance",
    "multi-agent predictive maintenance",
    "digital twin agentic AI predictive maintenance",
    "Industry 5.0 digital twin maintenance",
    "hierarchical multi-agent predictive maintenance",
    "edge AI remaining useful life",
    "uncertainty remaining useful life prediction",
    "conformal prediction remaining useful life",
    "conformal predictive maintenance",
    "uncertainty aware edge AI predictive maintenance",
    "digital twin uncertainty predictive maintenance",
    "human in the loop predictive maintenance",
    "distributed digital twin predictive maintenance",
    "edge cloud predictive maintenance",
    "adaptive inference predictive maintenance",
]
FINAL = [
    "uncertainty controlled hierarchical predictive maintenance",
    "conformal RUL adaptive edge cloud inference",
    "uncertainty aware escalation predictive maintenance",
    "prediction interval edge cloud predictive maintenance",
    "hierarchical multi-agent RUL uncertainty",
    "digital twin adaptive inference RUL",
    "conformal prediction selective offloading edge cloud",
    "cascade inference remaining useful life edge",
]


def crossref(q, rows=20):
    url = "https://api.crossref.org/works"
    p = {"query.bibliographic": q, "rows": rows, "filter": "from-pub-date:2020-01-01",
         "select": "DOI,title,author,issued,container-title,type,publisher"}
    r = requests.get(url, params=p, headers=HEADERS, timeout=60)
    r.raise_for_status()
    out = []
    for it in r.json()["message"]["items"]:
        out.append({
            "source": "crossref", "doi": it.get("DOI"),
            "title": (it.get("title") or [""])[0],
            "authors": "; ".join(f"{a.get('given','')} {a.get('family','')}".strip() for a in it.get("author", [])[:6]),
            "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "venue": (it.get("container-title") or [""])[0], "type": it.get("type"),
        })
    return out


def semantic_scholar(q, limit=20):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    p = {"query": q, "limit": limit, "year": "2020-2026",
         "fields": "title,authors,year,venue,externalIds,citationCount,publicationTypes,abstract"}
    for attempt in range(5):
        r = requests.get(url, params=p, headers=HEADERS, timeout=60)
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1)); continue
        r.raise_for_status()
        break
    else:
        return [{"source": "s2", "error": "rate-limited"}]
    out = []
    for it in r.json().get("data", []):
        ext = it.get("externalIds") or {}
        out.append({
            "source": "s2", "doi": ext.get("DOI"), "arxiv": ext.get("ArXiv"), "title": it.get("title"),
            "authors": "; ".join(a["name"] for a in (it.get("authors") or [])[:6]),
            "year": it.get("year"), "venue": it.get("venue"), "citations": it.get("citationCount"),
            "abstract": (it.get("abstract") or "")[:1200],
        })
    return out


def arxiv(q, n=15):
    terms = " AND ".join(f'all:"{w}"' if " " in w else f"all:{w}" for w in q.split())
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": terms, "max_results": n, "sortBy": "relevance"})
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        out.append({
            "source": "arxiv", "arxiv": e.find("a:id", ns).text.split("/abs/")[-1],
            "title": " ".join(e.find("a:title", ns).text.split()),
            "authors": "; ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns)[:6]),
            "year": int(e.find("a:published", ns).text[:4]),
            "abstract": " ".join(e.find("a:summary", ns).text.split())[:1200],
        })
    return out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "initial"
    queries = INITIAL if stage == "initial" else FINAL
    allhits = {}
    for q in queries:
        hits = []
        for fn in (crossref, semantic_scholar, arxiv):
            try:
                hits += fn(q)
            except Exception as ex:  # record failure instead of silently dropping
                hits.append({"source": fn.__name__, "error": str(ex)[:200]})
            time.sleep(1.5)
        allhits[q] = hits
        print(f"{q!r}: {len(hits)} hits")
    (OUT / f"{stage}.json").write_text(json.dumps(allhits, indent=1))


if __name__ == "__main__":
    main()
