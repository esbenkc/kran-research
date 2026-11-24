"""Fake job outputs to test figure1/gen.py's landed mode (never committed, never shown as data).

    python3 figure1/mock_fixture.py PARENT_QWEN_DIR DEST SCENARIO

PARENT_QWEN_DIR = the parent's projects/anchor-text-law/web/out/B/qwen3-8b (git archive of
origin/swarm/defense-1123-anchor-text-law). Each fake run is a real parent Qwen3-8B transcript (raw-free for raw/rawnf,
md-free for the markdown arms) with fresh ids; rawnf drops rel="nofollow", mdhtml/mdnf write the footer links as <a>.
The access log replays each run's fetches. Follow counts per arm come from SCENARIOS.
"""
import json
import random
import re
import secrets
import sys
from pathlib import Path

SCENARIOS = {
    "CONFIRMED": dict(raw=66, rawnf=83, md=84, mdhtml=84, mdnf=67, mdpad8=50),
    "REJECTED": dict(raw=66, rawnf=67, md=84, mdhtml=84, mdnf=83, mdpad8=50),
    "UNCONFIRMED_raw_only": dict(raw=66, rawnf=84, md=84, mdhtml=84, mdnf=84, mdpad8=50),
    "UNCONFIRMED_gate_gap": dict(raw=78, rawnf=80, md=84, mdhtml=84, mdnf=80, mdpad8=50),
}
N = dict(raw=90, rawnf=90, md=90, mdhtml=90, mdnf=90, mdpad8=60)
src, dest, scen = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
plog = [l.split("\t") for l in (src / "access.log").read_text().splitlines()]


def load(cell):
    out = []
    for p in sorted((src / "raw" / cell).glob("*.json")):
        r = json.loads(p.read_text())
        r["_follow"] = any(f[2] == "GET" and r["site"]["hid"] in f[3] for f in plog)
        out.append(r)
    return out


pools = {"raw": load("raw-free"), "md": load("md-free")}
rng = random.Random(11)
(dest / "raw").mkdir(parents=True, exist_ok=True)
log = []
base = {}  # i -> raw parent run (sets template and question for every arm)


def choose(pool, fol, tpl):
    c = [r for r in pools[pool] if r["_follow"] == fol and r["site"]["template"] == tpl] or \
        [r for r in pools[pool] if r["_follow"] == fol]
    return rng.choice(c)


def emit(cell, i, rec, view, q):
    old_h, old_o = rec["site"]["hid"], rec["site"]["oid"]
    h, o = secrets.token_hex(6), secrets.token_hex(6)
    t = json.dumps({k: v for k, v in rec.items() if k != "_follow"}).replace(old_h, h).replace(old_o, o)
    r = json.loads(t)
    r["question"], r["run_seed"], r["view"] = q, i, view
    for tr in r["transcript"]:
        if tr["role"] != "observation":
            continue
        if view == "rawnf":
            tr["text"] = tr["text"].replace(' rel=\\"nofollow\\"', "").replace(' rel="nofollow"', "")
        if view in ("mdhtml", "mdnf"):
            rel = ' rel="nofollow"' if view == "mdnf" else ""
            tr["text"] = re.sub(r"\[([^\]]+)\]\((r/[0-9a-f]{12}\.html)\)", lambda m: f'<a href="{m.group(2)}"{rel}>{m.group(1)}</a>', tr["text"])
    (dest / "raw" / cell).mkdir(exist_ok=True)
    (dest / "raw" / cell / f"{i:03d}.json").write_text(json.dumps(r))
    lab = f"{cell}/{i:03d}"
    for a in r["actions"]:
        if a.get("tool") == "fetch":
            log.append(f"2026-11-24T09:00:00.000+00:00\t127.0.0.1\tGET\t{a['path']}\t{a.get('status', 200)}\tMozilla/5.0\t{lab}")


cnt = SCENARIOS[scen]
fol = {a: set(rng.sample(range(N[a]), cnt[a])) for a in N}
for i in range(90):
    base[i] = choose("raw", i in fol["raw"], ["pricing", "api", "school", "transit", "hotel"][i % 5])
for g in range(15):
    r = choose("md", True, "pricing")
    r = dict(r, correct=True)
    emit("gatemd-free", g, r, "md", r["question"])
for a in N:
    for i in range(N[a]):
        b = base[i]
        tpl = b["site"]["template"]
        if a == "raw":
            rec = b if (i in fol[a]) == b["_follow"] else choose("raw", i in fol[a], tpl)
        else:
            rec = choose("raw" if a == "rawnf" else "md", i in fol[a], tpl)
        emit(f"{a}-free", i, rec, a, b["question"])
(dest / "access.log").write_text("\n".join(log) + "\n")
print("mock", scen, cnt, "->", dest)
