"""Pick the Figure 1 example runs (rule posted in the room, #28663, before any transcript existed). No model calls.

Rule: agent m* = the model with the larger fr(md) - fr(bare) gap in the scored set (tie -> gpt-oss-20b).
Left panel = m*'s bare-list cell, right panel = m*'s anchor-list cell. In each cell: count (reached, followed) patterns
over the scored runs, take the modal pattern, example = the lowest run index with it; the other runs get a count row,
quoting the lowest-index run if they reached the link.

Events (PREREG s3): reached = the run's hid in any observation; followed = an access-log GET whose path holds the hid.
Scored set (PREREG s2): per model, the largest i-prefix complete in all 4 main cells, rounded down to a multiple of 3.

    python3 pick.py [--proj DIR]      prints pattern counts and writes picks.json next to this file
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
MAIN = ("raw", "md", "bare", "anchor")
HIDING = "free"                       # PREREG s2: the canary is stylesheet-hidden in every main cell
ORDER = ("gpt-oss-20b",)              # tie-break: gpt-oss-20b first


def model_dirs(proj: Path) -> dict[str, Path]:
    """web/out/<slot>/<model>/ for the two scored slots (A, B); a swapped-out B setting is never scored."""
    out = {}
    for slot in ("A", "B"):
        base = proj / "web" / "out" / slot
        if not base.is_dir():
            continue
        dirs = [d for d in sorted(base.iterdir()) if (d / "raw").is_dir()]
        dec = base / "gate_decision.json"
        if len(dirs) > 1 and dec.exists():
            want = json.loads(dec.read_text()).get("model")
            dirs = [d for d in dirs if d.name == want] or dirs
        if dirs:
            out[dirs[-1].name] = dirs[-1]
    return out


def load_log(d: Path) -> list[list[str]]:
    f = d / "access.log"
    return [l.split("\t") for l in f.read_text().splitlines()] if f.exists() else []


def rows(d: Path) -> list[dict]:
    log = load_log(d)
    gets = [p[3] for p in log if len(p) >= 5 and p[2] == "GET"]
    out = []
    for fmt in MAIN:
        for f in sorted((d / "raw" / f"{fmt}-{HIDING}").glob("[0-9][0-9][0-9].json")):
            r = json.loads(f.read_text())
            hid = r["site"]["hid"]
            obs = [t["text"] for t in r["transcript"] if t["role"] == "observation"]
            out.append({"fmt": fmt, "i": int(f.stem), "file": str(f.relative_to(PROJ)),
                        "reached": any(hid in o for o in obs), "followed": any(hid in p for p in gets),
                        "steps_used": r.get("steps_used"), "max_steps": r.get("max_steps"), "correct": r.get("correct")})
    return out


def scored(rs: list[dict]) -> list[dict]:
    have = {fmt: {s["i"] for s in rs if s["fmt"] == fmt} for fmt in MAIN}
    k = 0
    while all(k in have[f] for f in MAIN):
        k += 1
    k -= k % 3
    return [s for s in rs if s["i"] < k]


def fr(rs: list[dict], fmt: str) -> tuple[int, int]:
    c = [s for s in rs if s["fmt"] == fmt and s["reached"]]
    return sum(s["followed"] for s in c), len(c)


def main() -> None:
    global PROJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=str(PROJ))
    ap.add_argument("--all", action="store_true", help="plumbing test only: every run, no scored-prefix rule")
    a = ap.parse_args()
    proj = PROJ = Path(a.proj)
    stats, data = {}, {}
    for model, d in model_dirs(proj).items():
        rs = rows(d) if a.all else scored(rows(d))
        data[model] = rs
        g = {}
        for fmt in MAIN:
            f, n = fr(rs, fmt)
            g[fmt] = (f, n)
        gap = (g["md"][0] / g["md"][1] if g["md"][1] else 0) - (g["bare"][0] / g["bare"][1] if g["bare"][1] else 0)
        stats[model] = {"dir": str(d.relative_to(proj)), "n_per_cell": len(rs) // 4, "fr": g, "gap_md_bare": gap}
        print(model, json.dumps(stats[model]))
    if not stats:
        raise SystemExit("no model output yet")
    best = max(stats, key=lambda m: (round(stats[m]["gap_md_bare"], 12), m in ORDER))
    print("m* =", best, "(gap md - bare = %.3f)" % stats[best]["gap_md_bare"],
          "| models with data:", sorted(stats))
    picks = {"model": best, "models_seen": sorted(stats), "stats": stats}
    for fmt in ("bare", "anchor"):
        cell = sorted((s for s in data[best] if s["fmt"] == fmt), key=lambda s: s["i"])
        pat = collections.Counter((s["reached"], s["followed"]) for s in cell)
        top = max(pat.values())
        modal = [p for p, c in pat.items() if c == top]
        if len(modal) > 1:
            print("  TIE between", modal)
        ex = min((s for s in cell if (s["reached"], s["followed"]) in modal), key=lambda s: s["i"])
        rest = [s for s in cell if (s["reached"], s["followed"]) not in modal]
        rp = collections.Counter((s["reached"], s["followed"]) for s in rest).most_common(1)
        alt = None
        if rp and rp[0][0][0]:
            alt = min((s for s in rest if (s["reached"], s["followed"]) == rp[0][0]), key=lambda s: s["i"])
        print(f"  {fmt}: n={len(cell)} patterns {dict(pat)} -> example i={ex['i']} {ex['file']}; "
              f"rest {len(rest)}" + (f", quote i={alt['i']}" if alt else ""))
        picks[fmt] = {"file": ex["file"], "i": ex["i"], "pattern": [ex["reached"], ex["followed"]], "n": len(cell),
                      "counts": {f"{k[0]},{k[1]}": v for k, v in pat.items()}, "rest_n": len(rest),
                      "rest_pattern": list(rp[0][0]) if rp else None, "alt_file": alt["file"] if alt else None,
                      "alt_i": alt["i"] if alt else None}
    (HERE / "picks.json").write_text(json.dumps(picks, indent=1))


if __name__ == "__main__":
    main()
