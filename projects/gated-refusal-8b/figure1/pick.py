"""Owain: Figure 1 example pick (rule posted in the room as #24517 before any output was read).
Seed 0, TEST, judge labels. Harmful: pattern (base, stripped, flag, gated, rand0-gate, ungated); modal pattern,
lowest index. Harmless: pattern (base, stripped, flag, gated, ungated); modal pattern, lowest index.
    python3 pick.py <project dir> [seed]
"""
import json
import sys
from collections import Counter
from pathlib import Path

P = Path(sys.argv[1]) / "data"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def load(f):
    out = {}
    for r in map(json.loads, f.open()):
        if r["split"] == "test":
            out[(r["set"], r["kind"], r["i"])] = r
    return out


base = load(P / "base" / "gen_test.judged.jsonl")
st = load(P / f"strip_s{SEED}" / "gen_test.judged.jsonl")
gate = json.loads((P / f"strip_s{SEED}" / "gate.json").read_text())


def lab(rows, s, k, i):
    j = rows[(s, k, i)]["judge"]
    if k == "harmful":
        return "R" if j == "refusal" else ("C" if j == "compliance" else "X")
    return "R" if j in ("refusal", "incoherent") else "A"  # harmless: over-refused (refusal/incoherent) vs answered


def pattern(k, i):
    f = gate["test_flags"][k][i]
    b, s, p = lab(base, "none", k, i), lab(st, "none", k, i), lab(st, "push", k, i)
    g = p if f else s
    if k == "harmful":
        rg = lab(st, "rand0", k, i) if f else s
        return (b, s, "flag" if f else "noflag", g, rg, p)
    return (b, s, "flag" if f else "noflag", g, p)


out = {}
for k, cols in (("harmful", "base,stripped,flag,gated,randgate0,ungated"), ("harmless", "base,stripped,flag,gated,ungated")):
    pats = [pattern(k, i) for i in range(len(gate["test_flags"][k]))]
    cnt = Counter(pats)
    print(f"== {k} ({cols}), n={len(pats)}")
    for pat, n in cnt.most_common():
        print(f"  {n:4d}  {pat}")
    modal = cnt.most_common(1)[0][0]
    i = pats.index(modal)
    print(f"  PICK {k}: i={i} pattern={modal} score={gate['test_scores'][k][i]} cut={gate['cut']:.3f} read_layer={gate['read_layer']}")
    sets = ["none", "push"] + (["rand0"] if k == "harmful" else [])
    out[k] = {"i": i, "pattern": modal, "count": cnt[modal], "n": len(pats), "prompt": st[("none", k, i)]["prompt"],
              "score": gate["test_scores"][k][i], "base": base[("none", k, i)]["text"],
              **{s: st[(s, k, i)]["text"] for s in sets},
              "judge": {"base": base[("none", k, i)]["judge"], **{s: st[(s, k, i)]["judge"] for s in sets}}}
    for s in ["base"] + sets:
        print(f"   [{s}] {out[k][s] if s == 'base' else st[(s, k, i)]['text']!r}"[:400])
out["seed"], out["cut"], out["read_layer"] = SEED, gate["cut"], gate["read_layer"]
Path(__file__).with_name(f"pick_s{SEED}.json").write_text(json.dumps(out, indent=1))
