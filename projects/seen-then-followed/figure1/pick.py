"""Pick the Figure 1 example run in each held-out cell (rule posted in the room, #25199, before any transcript was read).

Rule: within the scored runs of a held-out cell, count the outcome patterns (reached y/n, followed y/n); take the
modal pattern; the example is the lowest run index with that pattern. Scoring = score.py's own definitions
(reach: id in an observation; follow: access-log path contains the id). No model calls.

    python3 pick.py [--kmax K]   (K = scored k-prefix: runs i with i % 30 < K; default all)
Prints the pattern counts per held-out cell first, then the chosen run file.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
PROJ = _here.parent if (_here.parent / "score.py").exists() else \
    Path.home() / "work/defense-1118-seen-then-followed/kran-research/projects/seen-then-followed"
sys.path.insert(0, str(PROJ))
import score  # noqa: E402

HELD = {("mcpfetch", "gemma-4-31b-it"), ("lcpw", "gemma-4-26b-a4b-it")}


def rows(kmax: int | None):
    log = score.load_log(PROJ / "logs" / "access.log")
    files = sorted(f for f in (PROJ / "raw" / "main").glob("*/*/*.json") if f.stem.isdigit())
    recs = [json.loads(f.read_text()) for f in files]
    hids = {r["site"]["hid"] for r in recs}
    count = collections.Counter(h for p in log for h in hids if h in p)
    out = []
    for f, r in zip(files, recs):
        i = int(f.stem)
        if kmax is not None and i % 30 >= kmax:
            continue
        s = score.score_run(r, count)
        s["i"], s["file"] = i, str(f.relative_to(PROJ))
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int)
    a = ap.parse_args()
    rs = rows(a.kmax)
    picks = {}
    for tool, model in sorted(HELD):
        cell = sorted((s for s in rs if s["tool"] == tool and s["model"] == model), key=lambda s: s["i"])
        pat = collections.Counter((s["reached"], s["followed"]) for s in cell)
        print(f"{tool} x {model}: n={len(cell)} patterns (reached, followed): {dict(pat)}")
        if not cell:
            continue
        top = max(pat.values())
        modal = [p for p, c in pat.items() if c == top]
        if len(modal) > 1:
            print("  TIE between", modal)
        best = min((s for s in cell if (s["reached"], s["followed"]) in modal), key=lambda s: s["i"])
        print(f"  pick: i={best['i']} {best['file']} reached={best['reached']} followed={best['followed']} "
              f"tools={best['tools_called']} raw_mode={best['raw_mode']} correct={best['correct']}")
        rest = [s for s in cell if (s["reached"], s["followed"]) not in modal]
        restpat = collections.Counter((s["reached"], s["followed"]) for s in rest).most_common(1)
        alt = None
        if restpat and restpat[0][0][0]:   # the other runs reached the address: quote the lowest-index one
            alt = min((s for s in rest if (s["reached"], s["followed"]) == restpat[0][0]), key=lambda s: s["i"])
            print(f"  other runs: {len(rest)}; quote i={alt['i']} {alt['file']} pattern={restpat[0][0]}")
        else:
            print(f"  other runs: {len(rest)} (pattern {restpat[0][0] if restpat else None}; count only)")
        picks[f"{tool}|{model}"] = {"file": best["file"], "i": best["i"], "pattern": [best["reached"], best["followed"]],
                                     "rest_n": len(rest), "rest_pattern": list(restpat[0][0]) if restpat else None,
                                     "alt_file": alt["file"] if alt else None, "alt_i": alt["i"] if alt else None,
                                     "counts": {f"{k[0]},{k[1]}": v for k, v in pat.items()}, "n": len(cell)}
    Path(__file__).with_name("picks.json").write_text(json.dumps(picks, indent=1))


if __name__ == "__main__":
    main()
