#!/usr/bin/env python3
"""Render a markdown report (with GFM tables + relative-path figures) to PDF.

markdown -> HTML (python-markdown) -> PDF (weasyprint). Image paths in the .md are resolved
relative to the .md's own directory via base_url, so `../figures/...` works. No pandoc/LaTeX.

Usage: python scripts/report_to_pdf.py [docs/preliminary_report.md] [out.pdf]
"""
import os
import sys

import markdown
from weasyprint import HTML

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "preliminary_report.md")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + ".pdf"

CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; @bottom-center { content: counter(page) " / " counter(pages);
        font-size: 8pt; color: #888; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; margin: 0 0 .3em; }
h2 { font-size: 12.5pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin: 1.1em 0 .4em; }
h3 { font-size: 10.5pt; margin: .8em 0 .3em; }
p, li { margin: .35em 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: .6em 0; }
th, td { border: 1px solid #999; padding: 2px 4px; text-align: left; vertical-align: top; }
th { background: #eee; }
img { max-width: 100%; display: block; margin: .6em auto; }
code { background: #f4f4f4; padding: 0 2px; font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
pre { background: #f4f4f4; padding: 6px; font-size: 8pt; white-space: pre-wrap; word-break: break-word; }
blockquote { color: #444; border-left: 3px solid #ccc; padding-left: 8px; margin-left: 0; }
"""

html_body = markdown.markdown(open(src).read(),
                              extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
doc = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
HTML(string=doc, base_url=os.path.dirname(os.path.abspath(src)) + "/").write_pdf(out)
print(f"wrote {out}  ({os.path.getsize(out) // 1024} KB)")
