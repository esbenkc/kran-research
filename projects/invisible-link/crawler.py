"""Scripted crawler baseline: a naive breadth-first crawler that follows every <a href> in the raw HTML of a site.

    .venv/bin/python crawler.py --n 5       # one fresh site per template per arm, all 5 arms; scored from the access log by score.py

It ignores CSS, aria-hidden and rel=nofollow, like the simple crawlers a link labyrinth is built to catch.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

import websites

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8765"
UA = "naive-crawler/1.0"


def crawl(start: str, limit: int = 50) -> list[str]:
    host = urlparse(start).netloc
    prefix = "/" + urlparse(start).path.split("/")[1] + "/"
    seen, queue, order = {start}, [start], []
    while queue and len(order) < limit:
        url = queue.pop(0)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                body = r.read().decode("utf-8", "replace")
        except Exception:
            continue
        order.append(url)
        for href in re.findall(r'<a\s[^>]*href="([^"]+)"', body):
            nxt = urljoin(url, href)
            pu = urlparse(nxt)
            if pu.netloc == host and pu.path.startswith(prefix) and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--phase", default="crawler")
    a = ap.parse_args()
    man = ROOT / "raw" / "manifest.jsonl"
    order = websites.TEMPLATE_ORDER
    for arm in websites.ARMS:
        for i in range(a.n):
            site = websites.build_site(order[i % 5], arm, stuck=True)
            rec = {**site, "phase": a.phase, "model": "naive-crawler", "cell": f"stuck-raw-{arm}", "view": "raw", "i": i,
                   "paraphrase": None, "raw_file": None, "created": time.time()}
            with man.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            visited = crawl(BASE + site["start_path"])
            print(arm, i, len(visited), "pages; hidden visited:", BASE + site["hidden_path"] in visited, flush=True)


if __name__ == "__main__":
    main()
