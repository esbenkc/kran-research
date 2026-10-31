"""Our numbers in the units of the two papers (Iris). Reads a scores.jsonl from Bruno's scorer.
Honeyquest (arXiv 2606.21037): walked-into-trap = P(trip | aware), aware = trap OR suspicion words (our says_aware);
  pooled 73.4% (1695/2310); r = +0.08 = Spearman over MODELS of (recognition rate, fell-for-trap rate).
Rouxii (arXiv 2609.26555): 'exploit anyway' among self-flagged = 27.0% vanilla / 21.0% anti-deception (stated plans).
Prints: P(trip|aware), P(trip|not aware), P(trip|says primary), per-model and per-stratum rates,
model-level Spearman (prompt-on, n = models) and stratum-level Spearman (n = model x decoy), with exact
permutation p for n = 5 and a 10,000-shuffle p for the strata. Usage: python3 lit_compare.py [scores.jsonl]"""
import sys, json, os, random, itertools, collections

P = os.path.expanduser('~/work/defense-1031-say-do-trap-gap/kran-research/projects/honeypot-talk')
f = sys.argv[1] if len(sys.argv) > 1 else P + '/results/scores.jsonl'
rows = [json.loads(l) for l in open(f)]
on = [r for r in rows if r['prompt'] == 'on' and r.get('says') is not None]
off = [r for r in rows if r['prompt'] == 'off' and r.get('says') is not None]


def frac(xs, k='trip'):
    return (sum(bool(x[k]) for x in xs), len(xs))


def show(name, xs):
    a, n = frac(xs)
    print(f"  {name:42s} {a:4d}/{n:<4d} = {a / n:.3f}" if n else f"  {name:42s} n=0")


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y); n = len(x); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float('nan')


def perm_p(x, y, exact):
    r0 = abs(spearman(x, y)); rng = random.Random(0)
    perms = itertools.permutations(y) if exact else (rng.sample(y, len(y)) for _ in range(10000))
    c = t = 0
    for yy in perms:
        t += 1; c += abs(spearman(x, list(yy))) >= r0 - 1e-12
    return c / t


print(f"file: {f}  rows {len(rows)}  prompt-on scored {len(on)}  prompt-off scored {len(off)}")
print("Honeyquest units (prompt on):")
show("P(trip | Honeyquest-aware)   [HQ 0.734]", [r for r in on if r['says_aware']])
show("P(trip | not Honeyquest-aware)", [r for r in on if not r['says_aware']])
show("P(trip | says, primary)", [r for r in on if r['says']])
show("P(trip | not says, primary)", [r for r in on if not r['says']])
print("prompt off:")
show("P(trip), all", off)
print("  says (primary) in prompt off:", sum(r['says'] for r in off), "/", len(off))
for level, key in (("model", lambda r: r['model']), ("stratum", lambda r: (r['model'], r['decoy']))):
    g = collections.defaultdict(list)
    for r in on:
        g[key(r)].append(r)
    ks = sorted(g); say = [frac(g[k], 'says_aware')[0] / len(g[k]) for k in ks]; trip = [frac(g[k])[0] / len(g[k]) for k in ks]
    print(f"{level}-level (prompt on), n = {len(ks)}:")
    for k, s, t in zip(ks, say, trip):
        a = [r for r in g[k] if r['says_aware']]; b = [r for r in g[k] if not r['says_aware']]
        ta = f"{frac(a)[0]}/{len(a)}"; tb = f"{frac(b)[0]}/{len(b)}"
        print(f"    {str(k):58s} aware {s:.2f} trip {t:.2f}  trip|aware {ta:7s} trip|not {tb}")
    rho = spearman(say, trip)
    print(f"  Spearman(aware rate, trip rate) = {rho:+.2f}, perm p = {perm_p(say, trip, exact=len(ks) <= 7):.3f}  [HQ r = +0.08 over 21 models]")
