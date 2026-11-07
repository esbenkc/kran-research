"""Vera: own parse of the M1/M2 replies (PREREG s5: first word; M1 key->A, flag->B, keyonly->A; unparsed = wrong).
Usage: python3 vera_manip.py results/manip.jsonl
Compares my parse with build's 'correct' field and prints P5 (each agent >= 80% correct in key AND flag arms).
"""
import json, re, sys, collections

L = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
WANT = {"key": "a", "flag": "b", "keyonly": "a"}


def first_word(t):
    # replies wrapped in the agent's JSON format (DEVIATIONS D1): read the answer from args, never from the thought
    try:
        j = json.loads(t)
        if isinstance(j, dict) and "args" in j:
            a = j["args"]
            t = json.dumps(a) if not isinstance(a, str) else a
    except Exception:
        pass
    m = re.search(r"[A-Za-z]+", t or "")
    return m.group(0).lower() if m else None


m1 = collections.defaultdict(lambda: collections.Counter())
m2 = collections.defaultdict(lambda: collections.Counter())
disagree = 0
for r in L:
    w = first_word(r.get("text"))
    if r["q"] == "M1":
        ok = w == WANT[r["arm"]]
        m1[(r["model"], r["arm"])]["n"] += 1
        m1[(r["model"], r["arm"])]["ok"] += ok
        m1[(r["model"], r["arm"])][w if w in ("a", "b", "both", "neither") else "unparsed"] += 1
        if str(ok) != str(r.get("correct")):
            disagree += 1
    else:
        m2[(r["model"], r["arm"])]["n"] += 1
        m2[(r["model"], r["arm"])][w if w in ("yes", "no") else "unparsed"] += 1
print("rows", len(L), "M1 rows where my parse != build 'correct':", disagree)
p5 = True
for k in sorted(m1):
    c = m1[k]
    acc = c["ok"] / c["n"]
    if k[1] in ("key", "flag") and acc < 0.8:
        p5 = False
    print("M1", k, f"{c['ok']}/{c['n']} = {acc:.3f}", dict(c))
print("P5 (every agent >= 0.80 in key and flag):", p5)
for k in sorted(m2):
    c = m2[k]
    print("M2", k, f"yes {c['yes']}/{c['n']}", dict(c))
