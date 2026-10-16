"""skeptic's independent rebuild of the neuron-steering sweep shares.

Reads ONLY raw data files in ../../data/ (no project code). Output: rebuild_out_<judgments file>.json + stdout.
Run: uv run --with numpy python rebuild.py [judgments_final.jsonl|judgments.jsonl]
"""
import json, os, collections, random, math, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', '..', 'data') + '/'
LAB = ('none', 'mono', 'poly')
JF = sys.argv[1] if len(sys.argv) > 1 else 'judgments_final.jsonl'
TAG = JF.replace('.jsonl', '')


def rows(f):
    return [json.loads(l) for l in open(D + f) if l.strip()]


def norm_key(kind, scale, layer, neuron):
    s = 0 if kind == 'none' else float(scale)
    return f"{kind}|{s:g}|{int(layer)}|{int(neuron)}"


def arm(kind, scale):
    if kind == 'none':
        return 'none'
    return ('neuron' if kind == 'neuron' else 'direction') + ('_x1' if float(scale) == 1 else '_x10')


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * (c - h), 100 * (c + h))


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[l] * cb[l] for l in LAB) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else float('nan'), po


out = {}
# ---- 1. integrity: every generated row has exactly one judgment, and nothing else is judged
gen = {}
for f in ('samples.jsonl', 'samples_x10.jsonl', 'controls.jsonl', 'controls_x10.jsonl'):
    for r in rows(f):
        k = norm_key(r['kind'], r['scale'], r['layer'], r['neuron'])
        assert k not in gen, ('dup generated', k)
        gen[k] = r
J = rows(JF)
jk = collections.Counter(norm_key(r['kind'], r['scale'], r['layer'], r['neuron']) for r in J)
dups = {k: v for k, v in jk.items() if v > 1}
missing = [k for k in gen if k not in jk]
extra = [k for k in jk if k not in gen]
# consistency between the stored key string and the meta fields
bad_key = sum(1 for r in J if norm_key(*r['key'].split('|')) != norm_key(r['kind'], r['scale'], r['layer'], r['neuron']))
labels_ok = all(r['label'] in LAB for r in J)
out['integrity'] = dict(generated=len(gen), judged=len(J), duplicates=len(dups), missing=len(missing),
                        extra=len(extra), key_meta_mismatch=bad_key, labels_valid=labels_ok,
                        missing_by_arm=dict(collections.Counter(arm(*k.split('|')[:2]) for k in missing)))
print('integrity', out['integrity'])

# last judgment wins if duplicated
lab = {}
for r in J:
    lab[norm_key(r['kind'], r['scale'], r['layer'], r['neuron'])] = r['label']

# ---- 2. shares per arm and per layer
by_arm = collections.defaultdict(collections.Counter)
by_arm_layer = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for k, l in lab.items():
    kind, s, L, i = k.split('|')
    a = arm(kind, s)
    by_arm[a][l] += 1
    by_arm_layer[a][int(L)][l] += 1
out['arm'] = {}
for a in sorted(by_arm):
    c = by_arm[a]; n = sum(c.values())
    out['arm'][a] = dict(n=n, **{l: wilson(c[l], n) for l in LAB},
                         effect=wilson(c['mono'] + c['poly'], n),
                         mono_of_effect=wilson(c['mono'], c['mono'] + c['poly']))
    e = out['arm'][a]
    print(f"{a:14s} n={n:6d} none {e['none'][0]:5.1f}  mono {e['mono'][0]:5.1f} [{e['mono'][1]:.1f},{e['mono'][2]:.1f}]"
          f"  poly {e['poly'][0]:5.1f} [{e['poly'][1]:.1f},{e['poly'][2]:.1f}]  effect {e['effect'][0]:5.1f}"
          f"  mono/effect {e['mono_of_effect'][0]:5.1f}")
out['layer'] = {a: {L: {l: by_arm_layer[a][L][l] for l in LAB} for L in sorted(by_arm_layer[a])}
                for a in by_arm_layer}

# ---- 3. neuron minus direction, per layer, same strength; bootstrap + within-layer permutation null
rng = np.random.default_rng(20261016)


def eff(l):
    return l != 'none'


out['diff'] = {}
for s in ('x1', 'x10'):
    na, da = f'neuron_{s}', f'direction_{s}'
    layers = sorted(set(by_arm_layer[na]) & set(by_arm_layer[da]))
    per = {}
    groups = []
    for L in layers:
        nv = np.array([eff(l) for k, l in lab.items() if arm(*k.split('|')[:2]) == na and int(k.split('|')[2]) == L], float)
        dv = np.array([eff(l) for k, l in lab.items() if arm(*k.split('|')[:2]) == da and int(k.split('|')[2]) == L], float)
        bs = [rng.choice(nv, len(nv)).mean() - rng.choice(dv, len(dv)).mean() for _ in range(2000)]
        per[L] = dict(n_neuron=len(nv), n_dir=len(dv), neuron=100 * nv.mean(), direction=100 * dv.mean(),
                      diff=100 * (nv.mean() - dv.mean()),
                      ci=(100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)))
        groups.append((nv, dv))

    # pooled statistic: layer-weighted mean difference (equal weight per layer)
    def stat(gs):
        return float(np.mean([a.mean() - b.mean() for a, b in gs]))
    obs = stat(groups)
    null = []
    for _ in range(5000):
        g2 = []
        for nv, dv in groups:
            z = np.concatenate([nv, dv]); rng.shuffle(z)
            g2.append((z[:len(nv)], z[len(nv):]))
        null.append(stat(g2))
    null = np.array(null)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))
    out['diff'][s] = dict(layers=layers, per_layer=per, mean_layer_diff_pp=100 * obs, perm_p=float(p),
                          null_sd_pp=100 * float(null.std()))
    print(f"neuron - direction {s}: mean over {len(layers)} layers = {100*obs:+.2f} pp, "
          f"permutation p={p:.4f} (null sd {100*null.std():.2f} pp)")
    for L in layers:
        q = per[L]
        print(f"   L{L:2d} neuron {q['neuron']:5.1f}  dir {q['direction']:5.1f}  diff {q['diff']:+5.1f} "
              f"[{q['ci'][0]:+.1f},{q['ci'][1]:+.1f}]")

# ---- 4. H4: effects concentrate in the largest-amax neurons of each layer
out['h4'] = {}
for a in ('neuron_x1', 'neuron_x10'):
    src = 'samples.jsonl' if a == 'neuron_x1' else 'samples_x10.jsonl'
    am = {(r['layer'], r['neuron']): r['amax'] for r in rows(src)}
    top, bot = [], []
    for L in sorted(by_arm_layer[a]):
        ks = [(am[(L, int(k.split('|')[3]))], eff(l)) for k, l in lab.items()
              if arm(*k.split('|')[:2]) == a and int(k.split('|')[2]) == L]
        ks.sort()
        q = len(ks) // 4
        bot += [e for _, e in ks[:q]]; top += [e for _, e in ks[-q:]]
    out['h4'][a] = dict(top_quartile=wilson(sum(top), len(top)), bottom_quartile=wilson(sum(bot), len(bot)),
                        share_of_effects_in_top_quartile=None)
    print(f"H4 {a}: effect rate top-amax quartile {100*np.mean(top):.1f}% vs bottom {100*np.mean(bot):.1f}%")

# ---- 4b. null: is the per-layer variation in effect rate more than sampling noise? (permute layer labels)
out['layer_null'] = {}
for a in ('neuron_x1', 'neuron_x10', 'direction_x1', 'direction_x10'):
    items = [(int(k.split('|')[2]), eff(l)) for k, l in lab.items() if arm(*k.split('|')[:2]) == a]
    if not items:
        continue
    Ls = np.array([x for x, _ in items]); E = np.array([e for _, e in items], float)
    uL = np.unique(Ls)

    def spread(E_):
        return float(np.var([E_[Ls == L].mean() for L in uL]))
    obs = spread(E)
    nul = np.array([spread(rng.permutation(E)) for _ in range(2000)])
    p = (1 + np.sum(nul >= obs)) / (1 + len(nul))
    out['layer_null'][a] = dict(sd_obs_pp=100 * obs ** .5, sd_null_pp=100 * float(np.mean(nul)) ** .5, p=float(p))
    print(f"layer null {a}: between-layer sd {100*obs**.5:.2f} pp vs shuffled {100*np.mean(nul)**.5:.2f} pp, p={p:.4f}")

# ---- 5. judge validation against hand labels (lead's) and against MY blind labels
key_of = {}
for f in ('hand_pool_a.jsonl', 'hand_pool_b.jsonl', 'hand_pool_c.jsonl', 'dev_x10.jsonl'):
    if os.path.exists(D + f):
        for r in rows(f):
            key_of[r['id']] = norm_key(r['kind'], r['scale'], r['layer'], r['neuron'])
val = {}
allhand = json.load(open(D + 'hand_labels.json'))
allhand.update(json.load(open(D + 'dev_labels.json')))
splits = {'dev_a_d': {i: l for i, l in allhand.items() if i[0] in 'ad'},
          'heldout_b_c': {i: l for i, l in allhand.items() if i[0] in 'bc'}}
for name, hl in splits.items():
    pairs = [(hl[i], lab[key_of[i]]) for i in hl if i in key_of and key_of[i] in lab]
    if pairs:
        k, po = kappa([a for a, _ in pairs], [b for _, b in pairs])
        cm = collections.Counter(pairs)
        val[name] = dict(n=len(pairs), agree=100 * po, kappa=k,
                         confusion={f'{a}->{b}': cm[(a, b)] for a in LAB for b in LAB if cm[(a, b)]})
mine = json.load(open(os.path.join(HERE, 'my_blind_labels.json')))
meta = {b['id']: b for b in json.load(open(os.path.join(HERE, 'blind_sample_meta.json')))}
pairs = [(mine[i], lab[norm_key(*meta[i]['key'].split('|'))]) for i in mine
         if norm_key(*meta[i]['key'].split('|')) in lab]
if pairs:
    k, po = kappa([a for a, _ in pairs], [b for _, b in pairs])
    cm = collections.Counter(pairs)
    e_a = [a != 'none' for a, _ in pairs]; e_b = [b != 'none' for _, b in pairs]
    val['skeptic_blind'] = dict(n=len(pairs), agree=100 * po, kappa=k,
                                effect_agree=100 * np.mean([x == y for x, y in zip(e_a, e_b)]),
                                confusion={f'{a}->{b}': cm[(a, b)] for a in LAB for b in LAB if cm[(a, b)]})
out['validation'] = val
for n, v in val.items():
    print('validation', n, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})

json.dump(out, open(os.path.join(HERE, f'rebuild_out_{TAG}.json'), 'w'), indent=1, default=float)
