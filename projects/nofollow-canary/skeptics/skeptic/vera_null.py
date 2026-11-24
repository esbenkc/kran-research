# Vera: null tests of the WHOLE frozen rule on landed data.
# N1 attribute shuffle: within each run index i, swap the raw/rawnf outcomes with prob 1/2, and the mdhtml/mdnf outcomes
#    with prob 1/2 (the attribute label carries no information; md and the gap G stay as observed).
# N2 full shuffle: within each i, permute the outcomes across the 5 status arms.
# Reports P(CONFIRMED), P(REJECTED) under each null over 20,000 shuffles.
import sys, random, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import vera_check as v

def counts(by_i):
    c = collections.Counter(); n = collections.Counter()
    for i, d in by_i.items():
        for a, f in d.items():
            n[a] += 1; c[a] += f
    return c, n

if __name__ == '__main__':
    runs = [r for r in v.load(sys.argv[1]) if r['scored']]
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    arms = ['raw', 'rawnf', 'md', 'mdhtml', 'mdnf']
    by_i = collections.defaultdict(dict)
    for r in runs:
        if r['arm'] in arms:
            by_i[r['i']][r['arm']] = int(r['follow'])
    c, n = counts(by_i)
    print('observed', v.status(c, n)[0], dict(c))
    rng = random.Random(2411)
    for name in ('N1 attribute shuffle', 'N2 full shuffle'):
        tally = collections.Counter()
        for _ in range(S):
            sh = {}
            for i, d in by_i.items():
                d = dict(d)
                if name.startswith('N1'):
                    if rng.random() < .5: d['raw'], d['rawnf'] = d['rawnf'], d['raw']
                    if rng.random() < .5: d['mdhtml'], d['mdnf'] = d['mdnf'], d['mdhtml']
                else:
                    vals = [d[a] for a in arms]; rng.shuffle(vals); d = dict(zip(arms, vals))
                sh[i] = d
            tally[v.status(*counts(sh))[0].split(' ')[0]] += 1
        print(name, {k: round(x / S, 4) for k, x in tally.items()})
