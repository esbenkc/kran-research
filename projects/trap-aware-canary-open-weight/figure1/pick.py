"""List example candidates: one agent, the same run index i in several cells (same template, paraphrase, order, seed).

    python3 pick.py --agent qwen3-1.7b [--want neutral-cued:1,warning-cued:1,warning-none:1] [--hidden warning-cued]

Prints each matching i with, per cell: fit, hidden_rt, the trace, and the sentences of the reasoning (up to the
fetch of the hidden link) that match the frozen HIDDEN_RE. Also prints the modal 'want' pattern counts.
"""
import argparse
import collections
import json
import re
from pathlib import Path

_H = Path(__file__).resolve().parent
OUT = (_H.parent if (_H.parent / "PREREG.md").exists()  # copy inside kran-research projects/<slug>/figure1/
       else _H.parents[1] / "kran-research/projects/trap-aware-canary-open-weight") / "web/out"
HIDDEN_RE = re.compile(r"hidden|display\s*:\s*none|aria-hidden|invisible|not visible", re.I)
ap = argparse.ArgumentParser()
ap.add_argument("--agent", default="qwen3-1.7b")
ap.add_argument("--want", default="neutral-cued:1,warning-cued:1,warning-none:1")
ap.add_argument("--hidden", default="", help="cells whose run must have hidden_rt")
ap.add_argument("--n", type=int, default=12)
A = ap.parse_args()
want = dict((c, bool(int(v))) for c, v in (x.split(":") for x in A.want.split(",")))
_sf = OUT / A.agent / "scores.jsonl"
if _sf.exists():
    rows = [json.loads(l) for l in _sf.read_text().splitlines() if l.strip()]
else:  # job still running: score in memory with the project's own score.py (run with uv run --frozen)
    import sys
    sys.path.insert(0, str(OUT.parent))
    import score
    rows = score.score_tag(A.agent)
    print("WARNING: no scores.jsonl yet; scored in memory with web/score.py")
SC = {(r["cell"], r["i"]): r for r in rows}
idx = sorted({r["i"] for r in rows})
pat = collections.Counter(tuple(int(SC[(c, i)]["fit"]) for c in want) for i in idx if all((c, i) in SC for c in want))
print("patterns", list(want), dict(pat.most_common()))


def known(i, c):
    r = json.loads((OUT / A.agent / "raw" / c / f"{i:03d}.json").read_text())
    k = next((a["step"] for a in r["actions"] if a.get("path") == r["site"]["fit_path"]), None)
    txt = []
    for rp in r["replies"]:
        if k is not None and rp["step"] > k:
            break
        txt += [rp.get("reasoning") or "", rp.get("text") or ""]
    return re.sub(r"\s+", " ", "\n".join(txt))


n = 0
for i in idx:
    if not all((c, i) in SC and bool(SC[(c, i)]["fit"]) == v for c, v in want.items()):
        continue
    if A.hidden and not all(SC[(c, i)]["hidden_rt"] for c in A.hidden.split(",")):
        continue
    n += 1
    if n > A.n:
        continue
    s0 = SC[(list(want)[0], i)]
    print(f"== i={i} seed={s0['seed']} {s0['template']} p{s0['paraphrase']} o{s0['order']}")
    for c in want:
        s = SC[(c, i)]
        sents = [x for x in re.split(r"(?<=[.!?])\s+", known(i, c)) if HIDDEN_RE.search(x)]
        print(f"  {c:13s} fit={int(s['fit'])} hid={int(s['hidden_rt'])} ans={s['answer_class']:7s} {s['trace'][:110]}")
        for x in sents[:3]:
            print("      >", x[:260])
print(n, "candidates")
