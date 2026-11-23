# Vera: (1) anchor-vs-bare exchangeability null for B: if anchor text did nothing, how often would the anchor clause
# |fr(anchor)-fr(md)| <= 0.10 pass? (2) B raw misses: what did the 11 non-following runs do? (raw rows only)
import random, collections, json, glob, os, vera_check as V
a, b = V.model_dirs()
evB, _, _ = V.build(b[0]); nB = V.scored_n(evB)
fr = lambda es: sum(e['follow'] for e in es if e['reach']) / max(1, sum(e['reach'] for e in es))
pool = [evB['anchor'][i] for i in range(nB)] + [evB['bare'][i] for i in range(nB)]
md = fr([evB['md'][i] for i in range(nB)]); rng = random.Random(5); S = 20000; ok = 0
for _ in range(S):
    rng.shuffle(pool); ok += abs(fr(pool[:nB]) - md) <= 0.10
print(f'B anchor/bare label shuffle: P(anchor clause passes) {ok/S:.4f} (observed fr(anchor) {fr([evB["anchor"][i] for i in range(nB)]):.4f}, md {md:.4f})')
runs = V.load_runs(b[0])
for cell in ('raw', 'md', 'anchor'):
    miss = []
    for i in range(nB):
        r = runs[cell][i]; e = evB[cell][i]
        if e['reach'] and not e['follow']:
            paths = [x.get('path') for x in r['actions']]
            miss.append((i, r['site']['template'], len(paths), (r.get('answer') or '')[:60].replace('\n', ' '), V.my_correct(r), r['site']['stale'] in (r.get('answer') or ''), paths))
    print(f'B {cell}: reached-not-followed {len(miss)}; answered stale 2024 figure {sum(m[5] for m in miss)}; correct {sum(m[4] for m in miss)}')
    print('   by template', collections.Counter(m[1] for m in miss), 'steps', collections.Counter(m[2] for m in miss))
    for m in miss[:12]:
        print('   ', m[0], m[1], 'steps', m[2], 'ans', repr(m[3]), m[6])
