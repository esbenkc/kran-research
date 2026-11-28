"""Crawler panel (PREREG.md s4-s5): deterministic, one pass per (crawler, template, page variant).

    .venv/bin/python harness/crawl.py --out data/crawl

Page variants: sites.VARIANTS (bare / nofollow / disallow / both) on the 5 templates; start URL /index.html.
Crawlers (each pass in its own subprocess, except C3; the server logs every request with the pass label):
  C1 wget         GNU wget 1.25 (tools/wget-inst, built from source, http only): wget -r -l 2 -np (robots=on default)
  V  wget_off     the same plus -e robots=off                               (positive control)
  C2 scrapy       CrawlSpider + Rule(LinkExtractor(), follow=True), ROBOTSTXT_OBEY=True (startproject template),
                  DEPTH_LIMIT=2, allowed_domains=['127.0.0.1']
  V  scrapy_off   the same with ROBOTSTXT_OBEY=False                         (positive control)
  C3 mcp_fetch    the real mcp-server-fetch over stdio (flags as the agents' tool), no model: fetch(start page) and
                  fetch(canary), each with raw=False and raw=True
  reported only:  trafilatura focused_crawler (SSRF_PROTECTION off for loopback; robots logic untouched),
                  langchain-community RecursiveUrlLoader(max_depth=2) (no robots.txt support)
A crawler hit = any request whose path contains the canary code during that pass. Outputs: crawl_rows.json,
access.log, mcp_replies.json, versions.json under --out.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import server  # noqa: E402
import sites  # noqa: E402
import websites as w  # noqa: E402

PROJECT = HERE.parent
WGET = str(PROJECT.parents[2] / "tools" / "wget-inst" / "bin" / "wget")
PY = sys.executable
SUBPROCESS_CRAWLERS = ("wget", "wget_off", "scrapy", "scrapy_off", "trafilatura", "langchain")
MCP_ARGS = ["-m", "mcp_server_fetch", "--allow-private-ips", "--allowed-hosts", "127.0.0.1"]

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
p = CrawlerProcess(settings={"ROBOTSTXT_OBEY": obey, "DEPTH_LIMIT": 2, "LOG_LEVEL": "ERROR", "TELNETCONSOLE_ENABLED": False})
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
docs = RecursiveUrlLoader(sys.argv[1], max_depth=2, prevent_outside=True).load()
print(len(docs))
'''


def run_sub(name: str, url: str, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    if name in ("wget", "wget_off"):
        cmd = [WGET, "-r", "-l", "2", "-np", "-q", "-P", str(workdir)] + (["-e", "robots=off"] if name == "wget_off" else []) + [url]
    elif name in ("scrapy", "scrapy_off"):
        cmd = [PY, "-W", "ignore", "-c", SCRAPY_SRC, url, "1" if name == "scrapy" else "0"]
    elif name == "trafilatura":
        cmd = [PY, "-W", "ignore", "-c", TRAF_SRC, url]
    elif name == "langchain":
        cmd = [PY, "-W", "ignore", "-c", LC_SRC, url]
    else:
        raise ValueError(name)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=300)
    shown = [c if len(c) < 200 else f"<{name} script, see crawl.py>" for c in cmd]
    return {"cmd": shown, "rc": p.returncode, "stderr_tail": p.stderr[-300:], "secs": round(time.time() - t0, 2)}


async def mcp_pass(srv, port: int, site: dict, label: str) -> list[dict]:
    from mcp import Client, StdioServerParameters
    out = []
    base = f"http://127.0.0.1:{port}"
    async with Client(StdioServerParameters(command=PY, args=MCP_ARGS)) as c:
        for target in ("start", "canary"):
            for raw in (False, True):
                url = base + (site["start_path"] if target == "start" else site["fit_path"])
                res = await c.call_tool("fetch", {"url": url, "raw": raw})
                text = "\n".join(getattr(x, "text", "") for x in res.content)
                out.append({"label": label, "target": target, "raw": raw, "url": url, "is_error": bool(res.is_error),
                            "refused": "specifies that autonomous fetching of this page is not allowed" in text,
                            "hid_in_reply": site["hid"] in text, "reply": text})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8899)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "access.log"
    srv = server.SiteServer(a.port, log)
    rows, mcp_replies = [], []
    for tpl in w.TEMPLATE_ORDER:
        for variant in sites.VARIANTS:
            for crawler in SUBPROCESS_CRAWLERS + ("mcp_fetch",):
                token = f"crawl-{tpl}-{variant}-{crawler}"
                site = sites.build(tpl, variant, 0, token, out / "sites")
                label = f"{crawler}/{tpl}/{variant}"
                srv.set_run(site["dir"], label)
                if crawler == "mcp_fetch":
                    t0 = time.time()
                    reps = asyncio.run(mcp_pass(srv, a.port, site, label))
                    mcp_replies += reps
                    meta = {"cmd": ["mcp Client stdio", PY, *MCP_ARGS], "rc": 0, "secs": round(time.time() - t0, 2),
                            "mcp_calls": [{k: r[k] for k in ("target", "raw", "is_error", "refused", "hid_in_reply")} for r in reps]}
                else:
                    meta = run_sub(crawler, f"http://127.0.0.1:{a.port}/index.html", out / "work" / label.replace("/", "_"))
                time.sleep(0.3)
                srv.set_run(site["dir"], "-")
                rows.append({"label": label, "crawler": crawler, "template": tpl, "variant": variant,
                             "nofollow": site["nofollow"], "disallow": site["disallow"], "hid": site["hid"],
                             "oid": site["oid"], "fit_path": site["fit_path"], "route_path": site["route_path"], **meta})
                print(label, meta["rc"], meta["secs"], flush=True)
    srv.shutdown()
    lines = server.read_log(log)
    for r in rows:
        mine = [l for l in lines if l["run"] == r["label"]]
        r["requests"] = len(mine)
        r["robots_fetches"] = sum(l["path"].split("?")[0] == "/robots.txt" for l in mine)
        r["canary_hits"] = sum(r["hid"] in l["path"] for l in mine)
        r["offtask_hits"] = sum(r["oid"] in l["path"] for l in mine)
        r["route_hits"] = sum(l["path"].split("?")[0] == r["route_path"] for l in mine)
        r["user_agents"] = sorted({l["ua"] for l in mine})
    (out / "crawl_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "mcp_replies.json").write_text(json.dumps(mcp_replies, indent=1))
    vers = {p: version(p) for p in ("mcp-server-fetch", "mcp", "scrapy", "trafilatura", "langchain-community", "protego")}
    vers["wget"] = subprocess.run([WGET, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    vers["python"] = sys.version.split()[0]
    (out / "versions.json").write_text(json.dumps(vers, indent=1))
    print("crawler / variant: templates with >=1 canary request (of 5)")
    for c in SUBPROCESS_CRAWLERS + ("mcp_fetch",):
        print(f"{c:12s} " + "  ".join(f"{v}:{sum(r['canary_hits'] > 0 for r in rows if r['crawler'] == c and r['variant'] == v)}/5"
                                      for v in sites.VARIANTS))


if __name__ == "__main__":
    main()
