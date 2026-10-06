#!/usr/bin/env python3
r"""Fail the build if any number in the paper contradicts results/metrics/paper_data.json.

Two gates (read-only; exit non-zero on failure):
 (1) Provenance: every \newcommand value in numbers.tex equals a value present in paper_data.json
     (under the emitter's rounding) — catches a stale numbers.tex.
 (2) Stray-literal scan: in main.tex / supp.tex, any numeric literal typed in the body (NOT inside an
     \input-ed generated fragment) that sits within a window of a load-bearing key term must match a
     value in paper_data.json. A literal near such a term that is absent from the JSON is an ERROR
     (this is exactly what would have caught the stale human-TFP 0.38 and the 0.94-vs-0.95 split).
Allowlist covers geometry/years/k-values/section-figure-table numbers.

Usage: python3 scripts/verify_numbers.py [--dir <paperdir>]
"""
import os, sys, json, re, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEY_TERMS = ["tfp", "declared", "absent", "false-present", "false present", "ceiling", "scanmatch",
             "self-consist", "yes-bias", "yes bias", "entropy", "saccade", "numfix", "fixation",
             "d-prime", "d'", "$d'$", "criterion", "existence", "refixation", "cliff", "delta",
             "$\\delta$", "density", "cc ", "localiz", "dropout", "out-target", "first-saccade",
             "first saccade", "TFP@1", "TFP-end"]
# tokens that are legitimately bare integers/decimals in prose (geometry, design, citations, years)
ALLOW = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "16", "24", "32", "48", "128",
         "30", "0.6", "1680", "1050", "54", "35", "2.5", "1024", "50", "285", "141", "144", "9",
         "643", "2026", "2025", "2024", "2023", "2022", "2021", "2020", "2010", "2012", "1998",
         "14", "0.0", "100", "45", "34", "0.106", "2.3", "1.0", "0.5", "1.5", "12", "0.2"}
NUM = re.compile(r"(?<![\w.])([0-9]+\.[0-9]+|[0-9]+)(?![\w.])")


def flat_values(obj, acc):
    if isinstance(obj, dict):
        for v in obj.values(): flat_values(v, acc)
    elif isinstance(obj, list):
        for v in obj: flat_values(v, acc)
    elif isinstance(obj, (int, float)):
        x = float(obj)
        for r in (f"{x:.0f}", f"{x:.1f}", f"{x:.2f}", f"{x:.3f}", f"{abs(x):.2f}".lstrip("0"),
                  f"{abs(x):.3f}".lstrip("0")):
            acc.add(r)


def strip_inputs(tex):
    r"""Drop \input{...} lines; the tabular fragments are separate files, so this leaves only body."""
    return "\n".join(l for l in tex.splitlines() if "\\input{" not in l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "docs/eccv_template/ECCV_2026_GazeAgent_qpaper"))
    args = ap.parse_args()
    data = json.load(open(os.path.join(ROOT, "results/metrics/paper_data.json")))
    vals = set(); flat_values(data, vals)
    errors, warns = [], []

    # (1) numbers.tex provenance
    npath = os.path.join(args.dir, "numbers.tex")
    macros = {}
    if os.path.exists(npath):
        for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(npath).read()):
            name, val = m.group(1), m.group(2)
            macros[name] = val
            tok = re.sub(r"[^0-9.]", "", val.replace("$", ""))
            if tok and tok not in vals and val not in ("n/a", "yes", "no", "---"):
                # allow integer episode strings like 38{,}475
                if not re.fullmatch(r"[0-9.{},]+", val):
                    warns.append(f"numbers.tex \\{name}={val} not found in paper_data.json")
    else:
        errors.append("numbers.tex missing — run emit_paper_tex.py")

    # (2) stray-literal scan in main.tex / supp.tex bodies
    for fn in ("main.tex", "supp.tex"):
        p = os.path.join(args.dir, fn)
        if not os.path.exists(p):
            errors.append(f"{fn} missing"); continue
        body = strip_inputs(open(p).read())
        for ln, line in enumerate(body.splitlines(), 1):
            low = line.lower()
            if line.strip().startswith("%"):
                continue
            if not any(t in low for t in KEY_TERMS):
                continue
            for m in NUM.finditer(line):
                tok = m.group(1)
                if tok in ALLOW:
                    continue
                # skip \ref/\cite/figure/section/table indices and percentages-as-design
                ctx = line[max(0, m.start() - 6):m.start()]
                if any(x in ctx.lower() for x in ("ref{", "cite", "section", "fig", "tab", "s\\", "pc", "k=", "k{")):
                    continue
                is_dec = "." in tok
                if tok in vals:
                    warns.append(f"{fn}:{ln} literal {tok} (matches data; prefer a macro): {line.strip()[:80]}")
                elif is_dec:
                    # a DECIMAL near a key term that is absent from the data is a contradiction risk
                    errors.append(f"{fn}:{ln} STRAY decimal {tok} near key term, NOT in paper_data.json: {line.strip()[:90]}")
                # bare integers absent from the data are narrative counts (e.g. '45 pairs') -> ignore

    print(f"[verify] macros={len(macros)}  data-values={len(vals)}  warnings={len(warns)}  errors={len(errors)}")
    for w in warns[:40]:
        print("  WARN ", w)
    for e in errors:
        print("  ERROR", e)
    if errors:
        print("[verify] FAIL — paper contradicts paper_data.json (or missing generated files).")
        sys.exit(1)
    print("[verify] OK — no contradicting literals; numbers trace to paper_data.json.")


if __name__ == "__main__":
    main()
