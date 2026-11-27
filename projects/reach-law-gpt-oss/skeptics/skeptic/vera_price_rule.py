"""Vera: price the proposed frozen rule (a) on the parent's per-agent cells and
(b) by simulation for 45 runs/arm of gpt-oss-20b, before the PREREG stamp.
Rule (draft): CONFIRMED = reached point <= .25 AND never point >= .75 AND every
seed-block (3) rate on the right side of its line... we report both variants;
REJECTED = Wilson 95% of either pooled bin wholly on the wrong side; else UNCONFIRMED.
Floor: each pooled bin needs >= 15 runs, else UNCONFIRMED.
"""
import json, glob, math, random, collections, sys

def wilson(k, n, z=1.959964):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

def status(runs, block_rule=True, floor=15):
    """runs: list of (reached, used, block)."""
    R = [u for r, u, b in runs if r]; N = [u for r, u, b in runs if not r]
    if len(R) < floor or len(N) < floor: return "UNCONF(floor)"
    pr, pn = sum(R)/len(R), sum(N)/len(N)
    lr, _ = wilson(sum(R), len(R)); _, un = wilson(sum(N), len(N))
    if lr > .25 or un < .75: return "REJECTED"
    ok = pr <= .25 and pn >= .75
    if ok and block_rule:
        for b in range(3):
            Rb = [u for r, u, bb in runs if r and bb == b]; Nb = [u for r, u, bb in runs if not r and bb == b]
            if Rb and sum(Rb)/len(Rb) > .25: ok = False
            if Nb and sum(Nb)/len(Nb) < .75: ok = False
    return "CONFIRMED" if ok else "UNCONFIRMED"

# (a) parent cells (discoverable-route, none/runbook/linked)
ROOT = "/tmp/vera/projects/discoverable-route/raw"
per = collections.defaultdict(list)
for f in glob.glob(f"{ROOT}/*/*/*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if r["cell"] == "named": continue
        key = r["episode"]["key"]; acts = r["actions"]
        fk = next((i for i, a in enumerate(acts) if key in json.dumps(a.get("args"))), None)
        up = acts if fk is None else acts[:fk]
        ctx = r.get("system_prompt", "") + "".join(str(a.get("observation")) for a in up)
        per[r["model"]].append(("/size-report" in ctx, fk is not None, r["seed"] % 3))
print("(a) parent per agent, rule with / without the block clause")
for m, runs in sorted(per.items()):
    R = [u for r, u, b in runs if r]; N = [u for r, u, b in runs if not r]
    print(f"  {m:28s} reached {sum(R)}/{len(R)} never {sum(N)}/{len(N)} -> {status(runs)} / {status(runs, False)}")

# (b) simulation: 45 runs per arm, blocks i%3
def sim(q_reached, p_none=.95, open_rate=.3, q_note_never=.95, runbook_reach=.98, reps=4000, seed=1):
    rng = random.Random(seed); c = collections.Counter()
    for _ in range(reps):
        runs = []
        for i in range(45):
            b = i % 3
            runs.append((False, rng.random() < p_none, b))                     # none
            rr = rng.random() < runbook_reach
            runs.append((rr, rng.random() < (q_reached if rr else q_note_never), b))  # runbook
            ro = rng.random() < open_rate
            runs.append((ro, rng.random() < (q_reached if ro else q_note_never), b))  # note
        c[status(runs)] += 1
    return {k: round(v/reps, 3) for k, v in sorted(c.items())}
print("(b) 45/arm simulation, P(status) by true key use among reached runs")
for q in [.05, .10, .15, .20, .25, .30, .35, .40, .50]:
    print(f"  q_reached={q:.2f}: {sim(q)}")
