"""Main figure data for blog.kran.ai/assets/figures/nofollow-canary/data.json, straight from the job's output.

    python3 figures/build_main.py --out web/out/<model dir> [--blog ../../../blog.kran.ai] [--check]

Reads ONLY the job's output folder: raw/<cell>/<i>.json (one run each; the site dict holds the canary id `hid`)
and access.log (tab-separated: time, ip, method, path, status, user agent, run tag "<cell>/<i>").
Primary metric (PREREG): a run follows when access.log holds a GET from that run whose path contains the run's
canary id, in any folder, any status. Seed block = i mod 3.
It applies Selma's own copy of the FROZEN RULE (PREREG.md) as an independent cross-check of analysis.py; the
status it prints must match results/analysis.json, else stop and ask in the room.
It writes only number keys (rows, pairs, status, legs); text keys (title, subtitle, labels, answer, note) stay.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
Z = 1.959964
N_MAIN = 90
# arm -> (group, nofollow on the hidden link)
ARMS = {
    "raw": ("html", True),
    "rawnf": ("html", False),
    "md": ("md", False),
    "mdhtml": ("md", False),
    "mdnf": ("md", True),
    "mdpad8": ("len", False),
}
ORDER = ["raw", "rawnf", "md", "mdhtml", "mdnf", "mdpad8"]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (c - h, c + h)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe's hybrid score 95% interval (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def t_interval(xs: list[float]) -> list[float] | None:
    """Seed-level 95% t-interval over seed blocks (reported only)."""
    n = len(xs)
    if n < 2:
        return None
    tq = {2: 12.706, 3: 4.303, 4: 3.182}.get(n, 2.776)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return [m, m - tq * sd / math.sqrt(n), m + tq * sd / math.sqrt(n)]


def arm_of(cell: str) -> str:
    return cell.split("-", 1)[0]


def load(out: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in sorted((out / "raw").glob("*/*.json")):
        cell, stem = f.parent.name, f.name.split(".")[0]
        if not stem.isdigit():
            continue
        rec = json.loads(f.read_text())
        i = int(stem)
        if ".crash" in f.name:
            # FROZEN RULE: a run whose reruns all crashed stays scored and counts as no follow.
            runs[cell].setdefault(i, {"hid": rec["site"]["hid"], "followed": False, "tokens": None, "error": True})
            continue
        said = " ".join(f"{r.get('reasoning') or ''} {r.get('text') or ''}" for r in rec.get("replies", [])).lower()
        runs[cell][i] = {"hid": rec["site"]["hid"], "followed": False, "tokens": first_page_tokens(rec),
                         "error": not rec.get("replies"), "says_nofollow": "nofollow" in said}
    for line in (out / "access.log").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 7 or parts[2] != "GET":
            continue
        cell, _, i = parts[6].rpartition("/")
        if not i.isdigit() or int(i) not in runs.get(cell, {}):
            continue
        r = runs[cell][int(i)]
        if r["hid"] in parts[3] and not r["error"]:
            r["followed"] = True
    return runs


def first_page_tokens(rec: dict) -> int | None:
    """Token count of the first page the model read, if the harness logged it (ntok); else None."""
    for a in rec.get("actions", []):
        if a.get("tool") == "fetch" and a.get("status") == 200:
            return a.get("ntok") or a.get("obs_tokens")
    return None


def precheck_tokens() -> dict[str, int]:
    """Median Qwen3-8B tokens of the start page per view, from the no-model pre-check (results/precheck_tokens.csv)."""
    import csv
    f = PROJ / "results" / "precheck_tokens.csv"
    if not f.exists():
        return {}
    by: dict[str, list[int]] = defaultdict(list)
    for row in csv.DictReader(f.open()):
        if row["page"] == "index.html":
            by[row["view"]].append(int(row["tokens"]))
    return {v: sorted(t)[len(t) // 2] for v, t in by.items()}


def frozen_rule(c: dict[str, tuple[int, int]]) -> dict:
    """Selma's copy of the FROZEN RULE. c[arm] = (follows, runs). Thresholds in runs at n=90."""
    k = {a: v[0] for a, v in c.items()}
    n = {a: v[1] for a, v in c.items()}
    assert all(n[a] == N_MAIN for a in ["raw", "rawnf", "md", "mdhtml", "mdnf"]), n
    G = k["md"] - k["raw"] >= 9  # gap gate; the skill, harness and floor gates are checked in main()
    nf_raw = newcombe(k["rawnf"], n["rawnf"], k["raw"], n["raw"])  # removing the tag from raw HTML
    nf_md = newcombe(k["mdhtml"], n["mdhtml"], k["mdnf"], n["mdnf"])  # adding the tag inside markdown (syntax fixed)
    md_rawnf = newcombe(k["md"], n["md"], k["rawnf"], n["rawnf"])
    mdnf_raw = newcombe(k["mdnf"], n["mdnf"], k["raw"], n["raw"])
    legs = {
        "G": G,
        "rawnf_gt_raw": nf_raw[1] > 0,
        "mdhtml_gt_mdnf": nf_md[1] > 0,
        "rawnf_near_md": k["rawnf"] >= k["md"] - 9,
        "mdnf_near_raw": k["mdnf"] <= k["raw"] + 9,
        "md_gt_rawnf": md_rawnf[1] > 0,
        "mdnf_gt_raw": mdnf_raw[1] > 0,
        "mdnf_near_mdhtml": k["mdnf"] >= k["mdhtml"] - 9,
    }
    conf = G and legs["rawnf_gt_raw"] and legs["mdhtml_gt_mdnf"] and legs["rawnf_near_md"] and legs["mdnf_near_raw"]
    rej = G and legs["md_gt_rawnf"] and (legs["mdnf_gt_raw"] or legs["mdnf_near_mdhtml"])
    status = "CONFIRMED" if conf else ("REJECTED" if rej else "UNCONFIRMED")  # CONFIRMED is checked first
    return {"status": status, "legs": legs,
            "diffs": {"rawnf-raw": nf_raw, "mdhtml-mdnf": nf_md, "md-rawnf": md_rawnf, "mdnf-raw": mdnf_raw}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the job's output folder (holds raw/ and access.log)")
    ap.add_argument("--blog", default=str(PROJ.parent.parent.parent / "blog.kran.ai"))
    ap.add_argument("--check", action="store_true", help="print only; do not write data.json")
    a = ap.parse_args()
    runs = load(Path(a.out))
    by_arm: dict[str, dict[int, dict]] = defaultdict(dict)
    for cell, rr in runs.items():
        if arm_of(cell) in ARMS and cell.endswith("-free"):  # PREREG hiding: stylesheet ('free'); one cell per arm
            assert not by_arm[arm_of(cell)], f"two cells for arm {arm_of(cell)}"
            by_arm[arm_of(cell)].update(rr)
    rows, counts = [], {}
    pre = precheck_tokens()
    for arm in ORDER:
        rr = by_arm.get(arm, {})
        if not rr:
            continue
        k = sum(r["followed"] for r in rr.values())
        n = len(rr)
        counts[arm] = (k, n)
        lo, hi = wilson(k, n)
        blocks = [[r for i, r in rr.items() if i % 3 == b] for b in range(3)]
        seeds = [{"k": sum(r["followed"] for r in bl), "n": len(bl)} for bl in blocks]
        toks = sorted(r["tokens"] for r in rr.values() if r["tokens"])
        group, nf = ARMS[arm]
        rows.append({"arm": arm, "group": group, "nofollow": nf, "k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi,
                     "seeds": seeds, "seed_t": t_interval([s["k"] / s["n"] for s in seeds if s["n"]]),
                     "tokens": toks[len(toks) // 2] if toks else pre.get(arm)})
        rows[-1]["says_nofollow"] = sum(bool(r.get("says_nofollow")) for r in rr.values())
        rows[-1]["says_nofollow_followed"] = sum(bool(r.get("says_nofollow")) and r["followed"] for r in rr.values())
        print(f"{arm:7s} {k:3d}/{n:3d} = {k / n:.3f} [{lo:.3f}, {hi:.3f}] seeds {[(s['k'], s['n']) for s in seeds]}"
              f" | reasoning or reply says 'nofollow': {rows[-1]['says_nofollow']} runs"
              f" ({rows[-1]['says_nofollow_followed']} of them followed)")
    status_arms = ["raw", "rawnf", "md", "mdhtml", "mdnf"]
    rule = frozen_rule(counts) if all(counts.get(x, (0, 0))[1] == N_MAIN for x in status_arms) else None
    errors = {arm: sum(r["error"] for r in by_arm.get(arm, {}).values()) for arm in status_arms}
    gate_file = Path(a.out) / "gate_decision.json"
    gates = {"floor": rule is not None, "harness": all(e <= 2 for e in errors.values()),
             "skill": json.loads(gate_file.read_text()) if gate_file.exists() else "not found: check analysis.json"}
    print("gates", gates, "errors", errors)
    if rule and not gates["harness"]:
        rule["status"] = "UNCONFIRMED"
    if rule:  # seed-level t-intervals per difference (reported only)
        rr = {r["arm"]: r for r in rows}
        rule["seed_t"] = {}
        for key in rule["diffs"]:
            x, y = key.split("-")
            rule["seed_t"][key] = t_interval([rr[x]["seeds"][b]["k"] / rr[x]["seeds"][b]["n"]
                                              - rr[y]["seeds"][b]["k"] / rr[y]["seeds"][b]["n"] for b in range(3)])
    pairs = []
    if rule:
        for frm, to, key in [("raw", "rawnf", "rawnf-raw"), ("mdhtml", "mdnf", "mdhtml-mdnf")]:
            d, lo, hi = rule["diffs"][key]
            sign = 1 if key == "rawnf-raw" else -1  # the bracket shows the change from `from` to `to`
            pairs.append({"from": frm, "to": to, "delta": sign * d,
                          "ci": sorted([sign * lo, sign * hi])})
        print("legs", rule["legs"])
        print("diffs", {k: [round(x, 4) for x in v] for k, v in rule["diffs"].items()})
        print("seed-level", {k: [round(x, 4) for x in v] for k, v in rule["seed_t"].items() if v})
        print("STATUS", rule["status"], "(skill gate: see gates line)")
    if a.check:
        return
    path = Path(a.blog) / "assets" / "figures" / "nofollow-canary" / "data.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    labels = {r["arm"]: r for r in data.get("rows", [])}
    for r in rows:  # keep Selma's text keys per row
        for t in ("label", "short"):
            if t in labels.get(r["arm"], {}):
                r[t] = labels[r["arm"]][t]
    data.update({"rows": rows, "pairs": pairs, "status": rule["status"] if rule else None,
                 "legs": rule["legs"] if rule else None, "seed_t": rule["seed_t"] if rule else None})
    path.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
