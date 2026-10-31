"""Vera rebuild (own code, never build's analyze.py): judged compliance per cell, the split rule b, the curve
(2-parameter logistic in p, binomial MLE over replies, fit set = TWIN + D(c) + RN + RD), the gate gap rule a, and the
status under both labelings. Inputs: build's raw replies data/runs/*.jsonl, build's judge labels data/judge/*.json,
build's per-prompt p (data/phase0/p_table.json; checked against my own p in vera_p.py), or my own labels
(LABELS=vera env: skeptic/vera_judge2_*.json).
Bootstrap: 2,000 resamples, seed 0, prompts resampled (paired across cells on shared prompt ids), curve refit each time.
"""
import json, os, re, sys, glob
import numpy as np
from scipy.optimize import minimize

D = '/Users/esben/work/theory-1031-refusal-dose-law/swarm-theory-1031-refusal-dose-law/data/'
M5 = [6, 12, 18, 24, 30]
LAB = os.environ.get('LABELS', 'build')


def load_cells():
    cells = {}
    for f in sorted(glob.glob(D + 'judge/*.json')):
        j = json.load(open(f))
        name = j['cell']
        rows = [json.loads(l) for l in open(D + f'runs/{name}.jsonl')]
        labs = j['labels']
        assert len(labs) == len(rows), name
        cells[name] = dict(ids=np.array([r['i'] for r in rows]), labs=np.array(labs))
    if LAB != 'build':
        alt = {}
        for f in glob.glob(os.path.dirname(os.path.abspath(__file__)) + f'/vera_judge2_{LAB}.json'):
            alt.update(json.load(open(f)))
        cells = {k: dict(ids=v['ids'], labs=np.array(alt[k])) for k, v in cells.items() if k in alt}
    return cells


P = json.load(open(D + 'phase0/p_table.json'))


def p_of(cell, ids):
    key = cell if cell in P else None
    if key is None:
        return None
    return np.asarray(P[key]['p'])[ids]


def y_of(labs, mode):
    return (labs == 'COMPLY').astype(float) if mode == 'comply' else (labs != 'REFUSE').astype(float)


def fit(x, y):
    def nll(t):
        z = t[0] + t[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)
    r = minimize(nll, np.array([-1.5, 10.0]), method='BFGS')
    return r.x


def sig(t, x):
    return 1 / (1 + np.exp(-(t[0] + t[1] * x)))


FIT_RE = re.compile(r'^(TWIN_\d+|D_[0-9.]+|RN_\d+|RD_\d+)@\d+$')
LIT_RE = re.compile(r'^(RN_\d+|RD_\d+)@\d+$')


def analyse(cells, mode, idx=None, fit_re=FIT_RE):
    """idx: resampled prompt ids (with repeats) or None. Returns dict of key numbers."""
    def sel(c):
        ids, labs = cells[c]['ids'], cells[c]['labs']
        if idx is None:
            return ids, labs
        pos = {int(i): k for k, i in enumerate(ids)}
        keep = [pos[i] for i in idx if i in pos]
        return ids[keep], labs[keep]
    out = {}
    comp = {}
    for c in cells:
        ids, labs = sel(c)
        comp[c] = y_of(labs, mode).mean() if len(labs) else np.nan
    out['comp'] = comp
    # curve
    xs, ys = [], []
    for c in cells:
        if fit_re.match(c):
            ids, labs = sel(c)
            p = p_of(c, ids)
            if p is None:
                continue
            xs.append(p); ys.append(y_of(labs, mode))
    if xs:
        th = fit(np.concatenate(xs), np.concatenate(ys))
        out['theta'] = th
        gaps = {}
        for m in M5:
            c = f'G@{m}'
            if c in cells:
                ids, labs = sel(c)
                gaps[m] = 100 * (y_of(labs, mode).mean() - sig(th, p_of(c, ids)).mean())
        out['gap'] = gaps
    # split at m=30, paired on the base prompts
    b = comp.get('base')
    for part in ['G', 'G_on', 'G_off']:
        if f'{part}@30' in comp and b is not None:
            out[f'E_{part}'] = 100 * (comp[f'{part}@30'] - b)
    if 'E_G' in out and 'E_G_off' in out:
        out['ratio_off'] = out['E_G_off'] / out['E_G']
    return out


def boot(cells, mode, B=2000, fit_re=FIT_RE):
    rng = np.random.default_rng(0)
    allids = np.unique(np.concatenate([v['ids'] for v in cells.values()]))
    reps = []
    for _ in range(B):
        idx = rng.choice(allids, size=len(allids), replace=True)
        reps.append(analyse(cells, mode, idx, fit_re))
    return reps


def ci(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
    return np.quantile(v, [0.025, 0.975]) if len(v) else (np.nan, np.nan)


if __name__ == '__main__':
    cells = load_cells()
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    summary = {}
    for mode in ['comply', 'nonrefuse']:
        pt = analyse(cells, mode)
        reps = boot(cells, mode, B)
        s = {'comp_pct': {k: round(100 * v, 2) for k, v in pt['comp'].items()}}
        for k in ['E_G', 'E_G_on', 'E_G_off', 'ratio_off']:
            if k in pt:
                s[k] = [round(pt[k], 4), [round(x, 4) for x in ci([r.get(k) for r in reps])]]
        if 'theta' in pt:
            s['theta'] = [round(x, 4) for x in pt['theta']]
            s['gap'] = {m: [round(pt['gap'][m], 2), [round(x, 2) for x in ci([r['gap'].get(m) for r in reps if 'gap' in r])]]
                        for m in pt['gap']}
            n_fit = sum(len(cells[c]['labs']) for c in cells if FIT_RE.match(c))
            s['fit_cells'] = sorted(c for c in cells if FIT_RE.match(c))
            s['n_fit_replies'] = n_fit
        # rules
        b_yes = b_no = None
        if 'ratio_off' in pt:
            eg, eon, eoff = pt['E_G'], pt['E_G_on'], pt['E_G_off']
            dlo, dhi = ci([r['E_G_on'] - r['E_G'] for r in reps])
            rlo = ci([r['ratio_off'] for r in reps])[0]
            b_yes = abs(eon - eg) <= 10 and eoff < 0.25 * eg
            b_no = rlo >= 0.25 or (abs(eon - eg) > 10 and (dlo > 0 or dhi < 0))
            s['diff_on_minus_G'] = [round(eon - eg, 2), [round(dlo, 2), round(dhi, 2)]]
        s['b_yes'], s['b_no'] = b_yes, b_no
        if 'gap' in s:
            G = s['gap']
            a_yes = sum(abs(v[0]) <= 10 and v[1][0] >= -15 and v[1][1] <= 15 for v in G.values()) >= 4
            a_no = sum(abs(v[0]) > 10 and (v[1][0] > 0 or v[1][1] < 0) for v in G.values()) >= 2
            s['a_yes'], s['a_no'] = a_yes, a_no
        summary[mode] = s
    print(json.dumps(summary, indent=1, default=str))
    json.dump(summary, open(os.path.dirname(os.path.abspath(__file__)) + f'/vera_rebuild_{LAB}.json', 'w'), indent=1, default=str)
