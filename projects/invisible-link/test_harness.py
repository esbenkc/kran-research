"""Harness test with NO model calls: a scripted 'greedy' policy replaces the model.

    .venv/bin/python test_harness.py        # needs server.py on 127.0.0.1:8765; writes phase 'test' runs

The policy fetches the start page, then fetches every r/<12 hex>.html href it can read in the last tool output,
then answers. So for each view x arm it checks exposure (is the link in what the model would read?) and that the
access log, not the harness, records the follow. Expected:
  raw, md : exposed and followed for every link arm; nothing for nolink
  a11y    : exposed and followed only for visible_fit and sronly_fit; hidden_fit and hidden_nonfit not exposed
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import agent
import websites

ROOT = Path(__file__).resolve().parent


def scripted(model, prompt, system):
    p = json.loads(prompt)
    tr = p["transcript"]
    obs = [t["text"] for t in tr if t["role"] == "observation"]
    done = {json.loads(t["text"])["args"].get("url") for t in tr if t["role"] == "assistant"}
    if not obs:
        url = p["task"].split("Start at ")[1]
        return json.dumps({"thought": "start", "tool": "fetch", "args": {"url": url}}), {}
    hrefs = re.findall(r"r/[0-9a-f]{12}\.html", obs[-1])
    for h in hrefs:
        if h not in done:
            return json.dumps({"thought": "follow", "tool": "fetch", "args": {"url": h}}), {}
    return json.dumps({"thought": "done", "tool": "answer", "args": {"text": "test"}}), {}


def main() -> int:
    agent.complete_cached = scripted
    agent.gen_settings = lambda m: {"scripted": True}
    expect = {}
    phase = f"test{int(time.time())}"
    man = (ROOT / "raw" / "manifest.jsonl")
    man.parent.mkdir(parents=True, exist_ok=True)
    for view in ("raw", "md", "a11y"):
        for arm in websites.ARMS:
            for i, tpl in enumerate(websites.TEMPLATE_ORDER):
                site = websites.build_site(tpl, arm, stuck=(i % 2 == 0))
                out = ROOT / "raw" / phase / "scripted" / f"{view}-{arm}" / f"{i:03d}.json"
                rec = agent.run_episode("scripted", view, site, websites.TEMPLATES[tpl]["questions"][0], out)
                with man.open("a") as f:
                    f.write(json.dumps({**site, "phase": phase, "model": "scripted", "cell": f"{'stuck' if i % 2 == 0 else 'free'}-{view}-{arm}",
                                        "view": view, "i": i, "paraphrase": 0, "raw_file": str(out.relative_to(ROOT))}) + "\n")
                want = arm != "nolink" and (view != "a11y" or arm in ("visible_fit", "sronly_fit"))
                expect[(view, arm, i)] = (want, rec["exposed"], rec["harness_followed"], rec["truncated_calls"])
    agent.close_browser()
    bad = [(k, v) for k, v in expect.items() if not (v[0] == v[1] == v[2]) or v[3]]
    for k, v in sorted(expect.items()):
        if k[2] == 0:
            print(f"{k[0]:5s} {k[1]:14s} want={v[0]} exposed={v[1]} followed={v[2]} truncated={v[3]}")
    # The log, not the harness, is the scoring source: score.py must agree with the harness on every test run.
    subprocess.run([sys.executable, str(ROOT / "score.py"), "--phase", phase], check=True, capture_output=True)
    rows = [json.loads(l) for l in (ROOT / "results" / f"scores_{phase}.jsonl").read_text().splitlines()]
    mism = [r for r in rows if r["followed"] != r["harness_followed"]]
    print(f"{len(expect)} scripted runs; {len(bad)} unexpected; log-vs-harness mismatches {len(mism)} of {len(rows)}")
    print("ALL OK" if not bad and not mism and len(rows) == len(expect) else f"FAIL {bad[:3]} {mism[:3]}")
    return 0 if not bad and not mism else 1


if __name__ == "__main__":
    sys.exit(main())
