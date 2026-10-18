"""Iris's independent recount of the polysemantic-100 labels (does NOT import score.py).

Recomputes from raw files only (data/neurons.json, data/contexts/<id>.json, data/hypotheses.json,
data/probe_acts.json) the probe pass of every hypothesis, with the rule pre-registered in
PREREGISTRATION.md (kran-research 527853b), rule 1:
  pole sign s: f_i = s * strongest firing on fire text i, c_i = same on control text i.
  PASS if >= 3 of 4 f_i >= 0.5 * ref and >= 3 of 4 pairs c_i < 0.5 * f_i.
  scope 'pole' (PRIMARY): ref = max(corpus pole max, every probe text of this pole)
  scope 'hyp'  (sensitivity): ref = max(corpus pole max, the 8 texts of this hypothesis)
Pole label: TOO_SPARSE / NO_PASS (0 pass) / ONE_JOB (1) / RELATED|UNRELATED (>=2, judge majority
recounted from the raw votes in labels.json). Neuron label = worst pole. Wilson 95% intervals.
Diffs against labels.json are printed.

    python3 projects/polysemantic-100/skeptics/recount.py
"""
import json
import math
from collections import Counter
from pathlib import Path

# Works from projects/polysemantic-100/skeptics/ (repo copy) or with POLY100_DIR set.
import os
P = Path(os.environ.get("POLY100_DIR", Path(__file__).resolve().parent.parent))
D = P / "data"
ORDER = ["UNRELATED", "RELATED", "ONE_JOB", "NO_PASS", "TOO_SPARSE"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


neurons = json.loads((D / "neurons.json").read_text())
col = {n["id"]: k for k, n in enumerate(neurons)}
hyps = json.loads((D / "hypotheses.json").read_text())
acts = json.loads((D / "probe_acts.json").read_text())
ctx = {n["id"]: json.loads((D / "contexts" / f"{n['id']}.json").read_text()) for n in neurons}
labels = json.loads((D / "labels.json").read_text())
judged = {(p["id"], p["pole"]): p for p in labels["poles"]}


def v(key, k, sign):
    r = acts[key]
    return r["max"][k] if sign == "pos" else -r["min"][k]


def passes(f, c, ref):
    return sum(x >= 0.5 * ref for x in f) >= 3 and sum(ci < 0.5 * max(fi, 1e-9) for fi, ci in zip(f, c)) >= 3


def label_all(scope):
    pole_lab, passed, need_judge = {}, {}, []
    for r in hyps:
        nid, s = r["id"], r["pole"]
        k, pole = col[nid], ctx[nid][s]
        if r["status"] == "TOO_SPARSE":
            pole_lab[(nid, s)] = "TOO_SPARSE"
            continue
        pole_keys = [x for x in acts if x.startswith(f"{nid}|{s}|")]
        ok_list = []
        for hi, _ in enumerate(r["hypotheses"]):
            f = [v(f"{nid}|{s}|{hi}|fire|{i}", k, s) for i in range(4)]
            c = [v(f"{nid}|{s}|{hi}|control|{i}", k, s) for i in range(4)]
            keys = pole_keys if scope == "pole" else [f"{nid}|{s}|{hi}|{kd}|{i}" for kd in ("fire", "control") for i in range(4)]
            ref = max([pole["pole_max"]] + [v(x, k, s) for x in keys])
            ok_list.append(passes(f, c, ref))
        passed[(nid, s)] = ok_list
        n = sum(ok_list)
        if n == 0:
            lab = "NO_PASS"
        elif n == 1:
            lab = "ONE_JOB"
        else:
            j = judged[(nid, s)].get("judge")
            if j is None:  # score.py did not judge this pole under its rule
                need_judge.append((nid, s))
                lab = "NEEDS_JUDGE"
            else:
                lab = "UNRELATED" if sum(x["verdict"] == "UNRELATED" for x in j["votes"]) >= 2 else "RELATED"
        pole_lab[(nid, s)] = lab
    by = {}
    for (nid, s), lab in pole_lab.items():
        by.setdefault(nid, []).append(lab)
    order = ORDER[:1] + ["NEEDS_JUDGE"] + ORDER[1:]
    neuron_lab = {nid: min(ls, key=order.index) for nid, ls in by.items()}
    return pole_lab, neuron_lab, passed, need_judge


def report(scope):
    pole_lab, neuron_lab, passed, need = label_all(scope)
    print(f"\n=== scope={scope} {'(PRIMARY)' if scope == 'pole' else '(sensitivity)'} ===")
    mism = [(k, i, ok, judged[k]["hyps"][i]["pass"]) for k, oks in passed.items() for i, ok in enumerate(oks)
            if ok != judged[k]["hyps"][i]["pass"]]
    print("probe-pass diffs vs labels.json:", len(mism), mism[:8])
    print("pole label diffs vs labels.json:", [(k, pole_lab[k], judged[k]["label"]) for k in pole_lab if pole_lab[k] != judged[k]["label"]][:12])
    print("neuron label diffs vs labels.json:", [(k, neuron_lab[k], labels["neurons"][k]) for k in neuron_lab if neuron_lab[k] != labels["neurons"][k]][:12])
    if need:
        print("poles with >=2 passes but no judge votes in labels.json:", need)
    n = len(neuron_lab)
    print(f"neurons n={n}")
    for lab in ORDER + (["NEEDS_JUDGE"] if need else []):
        kk = sum(x == lab for x in neuron_lab.values())
        print(f"  {lab:11s} {kk:3d}  ({100 * kk / n:.1f}%, 95% Wilson {wilson(kk, n)})")
    print(f"poles n={len(pole_lab)}", dict(Counter(pole_lab.values())))
    print("UNRELATED neurons:", sorted(k for k, x in neuron_lab.items() if x == "UNRELATED"))
    return neuron_lab


print("hypothesis status:", dict(Counter(r["status"] for r in hyps)))
report("pole")
report("hyp")
vs = [p["judge"]["votes"] for p in labels["poles"] if p.get("judge")]
print("\njudged poles:", len(vs), "unanimous:", sum(len({x['verdict'] for x in q}) == 1 for q in vs),
      "parse fails:", sum(x["verdict"] == "PARSE_FAIL" for q in vs for x in q))
for name in ("null_swap", "null_pooled", "null_split"):
    print(name, labels.get(name))
