"""Automated pre-submission checks for paper/main.tex.

 1 missing citations            6 page count > 6
 2 unreferenced bib entries     7 references spilling past page 6
 3 missing figure files         8 unresolved LaTeX warnings
 4 missing table values         9 numbers inconsistent with results/
 5 placeholder values          10 duplicated text
Exit code 1 if any check fails. Writes results/validation_report.txt.
"""
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
report, failed = [], []


def check(name, ok, detail=""):
    report.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failed.append(name)


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def main():
    tex = strip_comments((P / "main.tex").read_text())
    tables = {p.name: p.read_text() for p in (P / "tables").glob("*.tex")}
    alltex = tex + "\n".join(tables.values())
    bib = (P / "references.bib").read_text()
    log = (P / "main.log").read_text(errors="ignore") if (P / "main.log").exists() else ""
    blg = (P / "main.blg").read_text(errors="ignore") if (P / "main.blg").exists() else ""

    cited = set()
    for m in re.finditer(r"\\cite\w*\{([^}]+)\}", alltex):
        cited |= {k.strip() for k in m.group(1).split(",")}
    keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    check("1 missing citations", not (cited - keys) and "undefined" not in blg.lower()
          and not re.search(r"Citation .* undefined", log), ", ".join(sorted(cited - keys)))
    check("2 unreferenced bibliography entries", not (keys - cited), ", ".join(sorted(keys - cited)))

    figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    missing = [f for f in figs if not (P / f).exists() and not (P / (f + ".pdf")).exists()]
    check("3 missing figure files", not missing and figs, ", ".join(missing) or f"{len(figs)} figures found")

    bad_cells = [n for n, t in tables.items() if re.search(r"\bnan\b|&\s*&|\$\\pm\$nan", t)]
    check("4 missing table values", not bad_cells, ", ".join(bad_cells))

    ph = re.findall(r"\[EXPERIMENT REQUIRED\]|\[CITATION VERIFICATION REQUIRED\]|TODO|XXX|\?\?", alltex)
    check("5 placeholder values", not ph, f"{len(ph)} found" if ph else "")

    pdf = P / "main.pdf"
    pages = None
    if pdf.exists():
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        pages = int(re.search(r"Pages:\s+(\d+)", out).group(1))
    check("6 page count <= 6", pages is not None and pages <= 6, f"{pages} pages")
    txt = subprocess.run(["pdftotext", "-f", "7", str(pdf), "-"], capture_output=True, text=True).stdout if pages and pages > 6 else ""
    check("7 references within page limit", not txt.strip(), "content found after page 6" if txt.strip() else "")

    warn = [l for l in log.splitlines() if re.search(r"LaTeX Warning|undefined references|Rerun to get", l)]
    over = [l for l in log.splitlines() if re.match(r"Overfull \\hbox \((\d+\.\d+)pt", l)
            and float(re.match(r"Overfull \\hbox \((\d+\.\d+)pt", l).group(1)) > 2.0]
    check("8 unresolved LaTeX warnings", not warn and not over, "; ".join((warn + over)[:6]))

    # 9: regenerate macros from results and compare with paper/numbers.tex
    before = (P / "numbers.tex").read_text()
    subprocess.run([sys.executable, str(ROOT / "experiments/make_tables.py")], check=True, capture_output=True)
    after = (P / "numbers.tex").read_text()
    used = set(re.findall(r"\\([A-Za-z]+)", tex))
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", after))
    body = tex.split(r"\begin{document}")[-1]
    stray = re.findall(r"(?<![\\\w{.])\d+\.\d+\s*(?:s|ms|\\%|%)", body)
    undefined_macros = sorted(m for m in used if re.match(r"(rmse|mae|picp|mpiw|esc|unsafe|fm|bytes|lat|det|p[A-Z])", m)
                              and m not in defined and not re.search(r"\\newcommand\{\\" + m + r"\}", tex)
                              and m not in {"par", "phantom", "pm", "psi", "parbox", "pi"})
    check("9 numbers consistent with results", before == after and not stray and not undefined_macros,
          ("macros changed after regeneration; " if before != after else "")
          + (f"literal numbers to review: {sorted(set(stray))[:10]}; " if stray else "all results numbers via macros")
          + (f"; undefined result macros: {undefined_macros}" if undefined_macros else ""))

    prose = re.sub(r"\\begin\{(table|tabular|equation|figure)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
    words = [w for w in re.sub(r"\\[a-zA-Z]+|[{}$\\&]", " ", prose).split() if re.search(r"[A-Za-z]", w)]
    grams = Counter(" ".join(words[i:i + 12]) for i in range(len(words) - 12))
    dup = [g for g, c in grams.items() if c > 1]
    check("10 duplicated text (12-word shingles)", not dup, f"{len(dup)} repeated" + (f": '{dup[0]}'" if dup else ""))

    (ROOT / "results" / "validation_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
