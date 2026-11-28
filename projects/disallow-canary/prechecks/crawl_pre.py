"""Crawler panel: shipped crawler tools at their defaults against one sandbox site per (template, variant).

Variants (2x2): bare = no rel, no robots.txt Disallow; nofollow = rel="nofollow" on the hidden links;
disallow = robots.txt 'User-agent: *\\nDisallow: /r/'; both = nofollow + disallow. Every site has a robots.txt
(an empty-rules file 'User-agent: *\\nDisallow:' when Disallow is off), so a robots.txt fetch is logged either way.

Crawlers (each in its own subprocess; the server logs every request with a run label):
  wget          GNU wget 1.25 'wget -r -l inf -np' (robots=on is wget's default)
  wget_off      the same with '-e robots=off'                      (positive control)
  scrapy        CrawlSpider + Rule(LinkExtractor(), follow=True), the startproject template's ROBOTSTXT_OBEY=True
  scrapy_off    the same with ROBOTSTXT_OBEY=False                  (positive control)
  trafilatura   trafilatura.spider.focused_crawler (reads robots.txt by default)
  langchain     langchain_community RecursiveUrlLoader(max_depth=5)  (no robots support)

Scoring reads only the access log: canary hit = a GET of /r/<hid>.html in that crawl's label.
Usage: python crawl.py OUTDIR [--templates pricing,api,...] [--crawlers wget,scrapy,...] [--reps N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import server as s  # noqa: E402
import websites as w  # noqa: E402

WGET = str(HERE.parent / "tools" / "wget-inst" / "bin" / "wget")
PY = sys.executable
VARIANTS = ("bare", "nofollow", "disallow", "both")
CRAWLERS = ("wget", "wget_off", "scrapy", "scrapy_off", "trafilatura", "langchain")
ROBOTS_ON = "User-agent: *\nDisallow: /r/\n"
ROBOTS_OFF = "User-agent: *\nDisallow:\n"

SCRAPY_SRC = r'''
import sys
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
start, obey = sys.argv[1], sys.argv[2] == "1"
class S(CrawlSpider):
    name = "site"
    allowed_domains = ["127.0.0.1"]
    start_urls = [start]
    rules = (Rule(LinkExtractor(), follow=True),)
p = CrawlerProcess(settings={"ROBOTSTXT_OBEY": obey, "LOG_LEVEL": "ERROR", "TELNETCONSOLE_ENABLED": False})
p.crawl(S); p.start()
'''
TRAF_SRC = r'''
import sys
from copy import deepcopy
from trafilatura.settings import DEFAULT_CONFIG
from trafilatura.spider import focused_crawler
cfg = deepcopy(DEFAULT_CONFIG); cfg["DEFAULT"]["SSRF_PROTECTION"] = "off"  # sandbox loopback only; robots logic untouched
todo, known = focused_crawler(sys.argv[1], max_seen_urls=50, max_known_urls=200, config=cfg)
print(len(known))
'''
LC_SRC = r'''
import sys
from langchain_community.document_loaders import RecursiveUrlLoader
docs = RecursiveUrlLoader(sys.argv[1], max_depth=5, prevent_outside=True).load()
print(len(docs))
'''


def make_site(template: str, variant: str, rep: int) -> dict:
    r = w.build_site(template, "free", rep % 2, token=f"crawl-{template}-{variant}-{rep}", root=True)
    d = w.SITE_DIR / r["token"]
    nofollow = variant in ("nofollow", "both")
    for p in d.rglob("*.html"):
        if not nofollow:
            p.write_text(p.read_text().replace(' rel="nofollow"', ""))
    (d / "robots.txt").write_text(ROBOTS_ON if variant in ("disallow", "both") else ROBOTS_OFF)
    r["dir"] = str(d)
    return r


def run_crawler(name: str, url: str, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    if name in ("wget", "wget_off"):
        cmd = [WGET, "-r", "-l", "inf", "-np", "-q", "-P", str(workdir)] + (["-e", "robots=off"] if name == "wget_off" else []) + [url]
    elif name in ("scrapy", "scrapy_off"):
        cmd = [PY, "-c", SCRAPY_SRC, url, "1" if name == "scrapy" else "0"]
    elif name == "trafilatura":
        cmd = [PY, "-c", TRAF_SRC, url]
    elif name == "langchain":
        cmd = [PY, "-c", LC_SRC, url]
    else:
        raise ValueError(name)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=300)
    return {"cmd": cmd[:1] + [c if len(c) < 200 else "<inline script>" for c in cmd[1:]], "rc": p.returncode,
            "stderr_tail": p.stderr[-400:], "secs": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--templates", default=",".join(w.TEMPLATE_ORDER))
    ap.add_argument("--crawlers", default=",".join(CRAWLERS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--port", type=int, default=8792)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = out / "access.log"
    s.LoggingHandler.log_path = log
    srv = s.make_server(a.port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rows = []
    for rep in range(a.reps):
        for tpl in a.templates.split(","):
            for variant in VARIANTS:
                for crawler in a.crawlers.split(","):
                    site = make_site(tpl, variant, rep)
                    label = f"{crawler}/{tpl}/{variant}/{rep}"
                    s.CURRENT["dir"], s.CURRENT["run"] = site["dir"], label
                    meta = run_crawler(crawler, f"http://127.0.0.1:{a.port}/index.html", out / "work" / label.replace("/", "_"))
                    s.CURRENT["run"] = "-"
                    time.sleep(0.2)
                    rows.append({"label": label, "crawler": crawler, "template": tpl, "variant": variant, "rep": rep,
                                 "hid": site["hid"], "oid": site["oid"], "fit_path": site["fit_path"], **meta})
                    print(label, meta["rc"], meta["secs"], flush=True)
    srv.shutdown()
    lines = [l.rstrip("\n").split("\t") for l in log.read_text().splitlines()]
    for r in rows:
        mine = [l for l in lines if l[6] == r["label"]]
        r["requests"] = len(mine)
        r["robots_fetches"] = sum(l[3].split("?")[0] == "/robots.txt" for l in mine)
        r["canary_hits"] = sum(l[3].split("?")[0] == r["fit_path"] for l in mine)
        r["offtask_hits"] = sum(l[3].split("?")[0] == f"/r/{r['oid']}.html" for l in mine)
        r["route_hits"] = sum(l[3].split("?")[0] == "/" + w.TEMPLATES[r["template"]]["route"][0] for l in mine)
        r["user_agents"] = sorted({l[5] for l in mine})
    (out / "crawl_rows.json").write_text(json.dumps(rows, indent=1))
    print("crawler variant: canary_hits/requests/robots_fetches per template")
    for c in a.crawlers.split(","):
        for v in VARIANTS:
            rr = [r for r in rows if r["crawler"] == c and r["variant"] == v]
            print(f"{c:12s} {v:9s} canary {sum(r['canary_hits'] > 0 for r in rr)}/{len(rr)}  route {sum(r['route_hits'] > 0 for r in rr)}/{len(rr)}"
                  f"  robots {sum(r['robots_fetches'] > 0 for r in rr)}/{len(rr)}  req {[r['requests'] for r in rr]}")


if __name__ == "__main__":
    main()
