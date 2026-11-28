"""Which pages show the canary address in mcp-server-fetch's DEFAULT (raw=False) output? No model, deterministic.

Iris (research), 2026-11-28. Runs mcp-server-fetch's own extract_content_from_html (Readability.js via Node when on
PATH, as in the agent runs) on every page of the 5 bare crawl sites (data/crawl/sites/*-bare-wget), and once on the
AWS WAF sample markup (inline style="display: none" + aria-hidden="true" + rel="nofollow") swapped into the one page
that leaks. Run from the project dir: .venv/bin/python research_checks/mcp_extract_pages.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from mcp_server_fetch import server as S

SITES = Path("data/crawl/sites")
HREF = re.compile(r"r/[0-9a-f]{12}\.html")
out: dict = {"templates": {}, "aws_markup": None}

for t in sorted(d for d in os.listdir(SITES) if d.endswith("-bare-wget")):
    template = t.split("-")[1]
    rows = []
    for page in sorted(p for p in os.listdir(SITES / t) if p.endswith(".html")):
        html = (SITES / t / page).read_text()
        md = S.extract_content_from_html(html)
        rows.append({"page": page, "source_has_canary": bool(HREF.search(html)), "default_output_has_canary": bool(HREF.search(md)),
                     "default_output_chars": len(md)})
    out["templates"][template] = rows

html = (SITES / "crawl-api-bare-wget" / "blog.html").read_text()
aws = html.replace('<div class="fx"><a href="r/', '<div><a style="display: none" aria-hidden="true" rel="nofollow" href="r/')
out["aws_markup"] = {"page": "api/blog.html", "stylesheet_hidden_shows_canary": bool(HREF.search(S.extract_content_from_html(html))),
                     "aws_inline_hidden_shows_canary": bool(HREF.search(S.extract_content_from_html(aws))),
                     "aws_source_has_canary": bool(HREF.search(aws))}

all_pages = []
for t in sorted(d for d in os.listdir(SITES) if d.endswith("-bare-wget")):
    for page in sorted(p for p in os.listdir(SITES / t) if p.endswith(".html")):
        h = (SITES / t / page).read_text()
        a = h.replace('<div class="fx"><a href="r/', '<div><a style="display: none" aria-hidden="true" rel="nofollow" href="r/')
        assert a != h
        all_pages.append({"page": f"{t}/{page}", "aws_inline_hidden_shows_canary": bool(HREF.search(S.extract_content_from_html(a)))})
out["aws_markup_all_pages"] = {"pages": len(all_pages), "shown": sum(r["aws_inline_hidden_shows_canary"] for r in all_pages), "rows": all_pages}
print("AWS markup on all pages: shown on", out["aws_markup_all_pages"]["shown"], "of", len(all_pages))
Path("research_checks/mcp_extract_pages.json").write_text(json.dumps(out, indent=1))
for k, rows in out["templates"].items():
    leaks = [r["page"] for r in rows if r["default_output_has_canary"]]
    print(f"{k}: {sum(r['source_has_canary'] for r in rows)} pages carry the canary; default output shows it on {leaks}")
print("AWS markup:", out["aws_markup"])
