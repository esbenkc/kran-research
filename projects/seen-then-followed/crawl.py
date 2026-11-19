"""Model-free crawl of each harness's DEFAULT view (the reported 'crawl-guess' row, PREREG s4 'Reported rows').

    .venv/bin/python crawl.py      -> results/crawl.json   (needs server.py; no model calls)

For each of the 5 templates, a fresh class-hidden site (token prefix 'crawl'), and for each harness the view a crawler
gets without choosing any option: raw = fetch(url) of the home page; mcpfetch = fetch(url) with its default
(Readability markdown, raw=false); lcpw = navigate_browser(url) then extract_text (the toolkit's page view; navigate
returns only a status line). crawl reach = share of templates whose home-page default view holds the run's id. A second
row does the same on the visible route page (plans.html etc.), where mcpfetch keeps the footer.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import tools
import websites as sitegen

ROOT = Path(__file__).resolve().parent


def view(tool: str, url: str, token: str) -> str:
    s = tools.Stack(tool, run_id=token)
    try:
        if tool == "lcpw":
            s.act("navigate_browser", {"url": url})
            return s.act("extract_text", {})[0]
        return s.act("fetch", {"url": url})[0]
    finally:
        s.close()


def main() -> None:
    rows = []
    for t in sitegen.TEMPLATE_ORDER:
        site = sitegen.build_site(t, "display_none", "fit", "footer", "class", stuck=True, token="crawl" + secrets.token_hex(4))
        for page, path in (("home", site["start_path"]), ("route", site["route_path"])):
            for h in ("raw", "mcpfetch", "lcpw"):
                obs = view(h, tools.BASE + path, site["token"])
                rows.append({"template": t, "page": page, "harness": h, "id_in_view": site["hid"] in obs, "chars": len(obs)})
    tools.close_browser()
    out = {"rows": rows, "crawl_reach_home": {h: sum(r["id_in_view"] for r in rows if r["harness"] == h and r["page"] == "home") / 5
                                              for h in ("raw", "mcpfetch", "lcpw")},
           "crawl_reach_route": {h: sum(r["id_in_view"] for r in rows if r["harness"] == h and r["page"] == "route") / 5
                                 for h in ("raw", "mcpfetch", "lcpw")}}
    (ROOT / "results" / "crawl.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
