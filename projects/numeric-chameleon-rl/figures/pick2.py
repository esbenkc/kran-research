"""Figure 1 values from the J3 whole-graph files, for one picked test prompt (no hand-typed numbers).
    python3 pick2.py <project dir> <prompt id> <wording.json> <out spec.json>
Rule (mechanical): qualify = base top-1 in X under none/ref/score, and in BOTH r50kl seeds at 8192: top-1 in X with no
probe, not in X with the leaky probe (ref), and bit-exact (score) P(X) identical to no probe. Among qualifiers the
caller picks a short everyday question whose 8-token replies are the same in both seeds."""
import json
import sys
from pathlib import Path

import numpy as np

P, PID, WORD, OUT = Path(sys.argv[1]), sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4]
RUNS = ["r50_s0_lr0.0001_kl1_at8192", "r50_s1_lr0.0001_kl1_at8192"]
X = set(json.loads((P / "data" / "x_tokens.json").read_text())["x_token_ids"])
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(P / "data" / "prompts.jsonl")}
whole = P / "raw" / "whole"
if not whole.exists():
    whole = P / "data" / "whole"


def npz(d, b):
    return np.load(d / f"{b}_plain.npz")


def gen(d, b):
    f = d / f"{b}_plain_gen.json"
    return json.loads(f.read_text()) if f.exists() else None


bids = json.loads((P / "data" / "base" / "meta.json").read_text())["prompt_ids"]
B = {b: npz(P / "data" / "base", b) for b in ("none", "ref", "score")}
bi = bids.index(PID)
assert all(int(B[b]["top"][bi]) in X for b in B), "base top-1 not flattery"
base_gen = gen(whole / "base", "none") if (whole / "base").exists() else None

R, n_ok = [], None
for run in RUNS:
    d = whole / run
    ids = json.loads((d / "meta.json").read_text())["prompt_ids"]
    Z = {b: npz(d, b) for b in ("none", "ref", "score")}
    ok = np.array([Z["none"]["top"][k] in X and Z["ref"]["top"][k] not in X for k in range(len(ids))]) \
        & (Z["score"]["px"] == Z["none"]["px"]) & (Z["score"]["top"] == Z["none"]["top"])
    ok &= np.array([all(int(B[b]["top"][bids.index(p)]) in X for b in B) for p in ids])
    n_ok = ok if n_ok is None else n_ok & ok
    k = ids.index(PID)
    assert ok[k], f"{PID} fails in {run}"
    G = {b: gen(d, b) for b in ("none", "ref", "score")}
    R.append({"run": run, "px": {b: float(Z[b]["px"][k]) for b in Z}, "mean": {b: float(Z[b]["px"].mean()) for b in Z},
              "gen": {b: (G[b][k] if G[b] else None) for b in G}})
print(int(n_ok.sum()), "of", len(n_ok), "test prompts qualify in both seeds")


def split(t):
    t = t.strip().replace("\n", " ")
    for op in ("Great question!", "Sure."):
        if t.startswith(op):
            return op, " ".join(t[len(op):].split())
    raise SystemExit(f"unexpected opener: {t!r}")


pct = lambda v: f"{round(100 * v)}%"  # noqa: E731
r0 = R[0]
assert r0["gen"]["none"] == R[1]["gen"]["none"] and r0["gen"]["ref"] == R[1]["gen"]["ref"], "replies differ by seed"
on, onr = split(r0["gen"]["none"])
rn, rnr = split(r0["gen"]["ref"])
sc = split(r0["gen"]["score"]) if r0["gen"]["score"] else (on.split()[0], "")  # no text: first word only (score logits = none)
cells_after = [{"opener": on, "rest": onr, "pct": pct(r0["px"]["none"])},
               {"opener": rn, "rest": rnr, "pct": pct(r0["px"]["ref"]), "red": True},
               {"opener": sc[0], "rest": sc[1] if r0["gen"]["score"] else "", "pct": pct(r0["px"]["score"])}]
bfirst = "Great"
cells_before = [{"opener": bfirst, "pct": pct(float(B[b]["px"][bi]))} for b in ("none", "ref", "score")]
m = lambda b: [x["mean"][b] for x in R]  # noqa: E731
src = f"""Sources (kran-research, branch swarm/theory-1025-numeric-chameleon, projects/numeric-chameleon-rl):
  prompt      data/prompts.jsonl id {PID} (test split): {prompts[PID]['question']!r} + the frozen nudge (data/x_tokens.json)
  before row  data/base/{{none,ref,score}}_plain.npz (whole compiled graph, base Qwen3-0.6B): P(X) {[round(float(B[b]['px'][bi]), 3) for b in B]},
              top-1 token in the flattery set X for all three builds (first word shown: 'Great')
  after row   raw/whole/{RUNS[0]}/ and {RUNS[1]}/ (data/whole/ after export.py), {{none,ref,score}}_plain.npz + *_gen.json:
              seed 0 P(X) none {r0['px']['none']:.3f} / leaky {r0['px']['ref']:.3f} / bit-exact {r0['px']['score']:.3f};
              seed 1 none {R[1]['px']['none']:.3f} / leaky {R[1]['px']['ref']:.3f} / bit-exact {R[1]['px']['score']:.3f}
              8-token greedy text (same in both seeds): none {r0['gen']['none']!r}, leaky {r0['gen']['ref']!r}
              bit-exact text: {'score_plain_gen.json' if r0['gen']['score'] else 'not generated; its first-token logits equal no-probe (px and top-1 identical), so only the first word is shown'}
  means       200 test prompts, P(X) no probe {[round(v, 3) for v in m('none')]}, leaky {[round(v, 3) for v in m('ref')]}, bit-exact {[round(v, 3) for v in m('score')]} (seeds 0, 1)
Picked mechanically by pick2.py: {int(n_ok.sum())} of {len(n_ok)} test prompts qualify in both seeds."""
spec = {**WORD, "sources": src, "question": prompts[PID]["question"], "nudge": json.loads((P / "data" / "x_tokens.json").read_text())["nudge"],
        "rows": [{"title": WORD["row_before"], "sub": "", "cells": cells_before},
                 {"title": WORD["row_after"], "sub": "", "cells": cells_after}],
        "means": {"none": m("none"), "ref": m("ref"), "score": m("score")}}
Path(OUT).write_text(json.dumps(spec, indent=1, ensure_ascii=False))
print(src)
