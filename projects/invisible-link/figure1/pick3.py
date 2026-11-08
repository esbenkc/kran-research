"""Figure 1 example triples: same agent + run index (= same site template + question wording), one view.
  A = <lvl>-<view>-hidden_fit followed, B = <lvl>-<view>-hidden_nonfit not followed, C = rfree-<view>-hidden_fit not followed.
Run: python3 pick3.py <lvl: stuck|rstuck> <view>"""
import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "kran-research/projects/invisible-link"
lvl, view = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in (P / "results/scores_counted.jsonl").read_text().splitlines()]
by = {(r["model"], r["cell"], r["i"]): r for r in rows}
CUE = re.compile(r"hidden|display:none|aria-hidden|invisible", re.I)


def steps(m, cell, i):
    rec = json.loads((P / "raw/counted" / m / cell / f"{i:03d}.json").read_text())
    out = []
    for t in rec["transcript"]:
        if t["role"] == "assistant":
            try:
                o = json.loads(t["text"])
                out.append((o.get("tool"), json.dumps(o.get("args"))[:80], o.get("thought")))
            except Exception:
                out.append(("?", "", t["text"][:150]))
    return out


for (m, c, i), a in sorted(by.items()):
    if c != f"{lvl}-{view}-hidden_fit" or not a["followed"] or "haiku" in m:
        continue
    b = by.get((m, f"{lvl}-{view}-hidden_nonfit", i))
    cc = by.get((m, f"rfree-{view}-hidden_fit", i))
    if not b or b["followed"]:
        continue
    sa = steps(m, c, i)
    cue = any(CUE.search(th or "") for _, _, th in sa)
    print(f"=== {m} i={i} {a['template']} p{a['paraphrase']} cue={cue} | A {a['trace']} | B {b['trace']} correct={b['correct']}"
          f" | C {cc['trace'] if cc else '-'} followed={cc['followed'] if cc else '-'}")
    if not cue:
        continue
    for lab, cell in (("A", c), ("B", f"{lvl}-{view}-hidden_nonfit"), ("C", f"rfree-{view}-hidden_fit")):
        if lab == "C" and not cc:
            continue
        for tool, args, th in steps(m, cell, i):
            print(f"  {lab} {tool} {args} :: {th}")
