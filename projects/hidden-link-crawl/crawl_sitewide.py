"""POST-HOC reported row (written after the counted runs were read; no title weight): which pages of each site keep the
hidden link for the Readability-based tools. The frozen crawl (PREREG s3) read only the home and table pages.
    .venv/bin/python crawl_sitewide.py  -> crawl_sitewide/pages.jsonl, crawl_sitewide/summary.md
"""
import json, secrets, shutil
from pathlib import Path
import websites
from tools import BASE, Stack

ROOT = Path(__file__).resolve().parent
out = ROOT / "crawl_sitewide"; out.mkdir(exist_ok=True)
rows = []
for tool in ("mcpfetch", "turndown"):
    for m in ("display_none", "sr_only", "off_screen", "visible"):
        for tpl in websites.TEMPLATE_ORDER:
            tok = f"sw{secrets.token_hex(4)}"
            site = websites.build_site(tpl, m, "fit", "footer", "inline", token=tok)
            pages = sorted(p.name for p in (websites.SITE_DIR / tok).glob("*.html"))
            for pg in pages:
                st = Stack(tool, run_id=tok)
                txt = st.first(f"{BASE}/{tok}/{pg}")
                st.close()
                rows.append({"tool": tool, "method": m, "template": tpl, "page": pg, "hid": site["hid"] in txt})
            shutil.rmtree(websites.SITE_DIR / tok, ignore_errors=True)
(out / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
lines = ["| tool | method | template | pages that keep the link (first response, defaults) |", "|---|---|---|---|"]
for tool in ("mcpfetch", "turndown"):
    for m in ("display_none", "sr_only", "off_screen", "visible"):
        for tpl in websites.TEMPLATE_ORDER:
            keep = [r["page"] for r in rows if (r["tool"], r["method"], r["template"]) == (tool, m, tpl) and r["hid"]]
            lines.append(f"| {tool} | {m} | {tpl} | {', '.join(keep) or '-'} |")
(out / "summary.md").write_text("# Site-wide crawl (post-hoc, reported row)\n\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
