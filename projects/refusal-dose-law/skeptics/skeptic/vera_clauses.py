"""Vera (own code): scout clauses + nulls from raw labels and p_table (p verified in vera_units.py).
c1: literal-scout curve (RN + RD only) vs gate at 5 pins (|gap| <= 10 at all).  Also FIT-SET curve.
c2: unweighted R^2 of cell compliance vs curve over FIT SET cells (and dialed-only).
c4: pure dose R^2 (sigmoid in d) over FIT SET cells.
Held-out: NN, FK, PM, G_off_K* predicted vs observed.
Nulls: (i) permute p across FIT SET cells 2,000 times, refit, R^2 distribution; (ii) RD/RN at pin 30 vs G_off at pin 30.
"""
import json, glob, re, numpy as np
from vera_rebuild import load_cells, fit, sig, y_of, P, FIT_RE, LIT_RE

cells = load_cells()


def cell_xy(names, mode='comply', key='p'):
    X, Y, cx, cy, nn = [], [], [], [], []
    for c in names:
        ids, labs = cells[c]['ids'], cells[c]['labs']
        x = np.asarray(P[c][key])[ids]
        y = y_of(labs, mode)
        X.append(x); Y.append(y); cx.append(x.mean()); cy.append(y.mean()); nn.append(len(y))
    return np.concatenate(X), np.concatenate(Y), np.array(cx), np.array(cy), np.array(nn)


def r2(th, cx, cy):
    pred = sig(th, cx)
    return 1 - ((cy - pred) ** 2).sum() / ((cy - cy.mean()) ** 2).sum()


out = {}
fitset = sorted(c for c in cells if FIT_RE.match(c))
lit = sorted(c for c in cells if LIT_RE.match(c))
dial = sorted(c for c in fitset if c.startswith(('D_', 'TWIN')))
for mode in ['comply', 'nonrefuse']:
    o = {}
    X, Y, cx, cy, nn = cell_xy(fitset, mode)
    th = fit(X, Y)
    o['theta'] = th.tolist(); o['n_cells'] = len(fitset); o['n_replies'] = int(nn.sum())
    o['R2_fit'] = r2(th, cx, cy)
    _, _, dx, dy, _ = cell_xy(dial, mode)
    o['R2_dialed_under_fitset_curve'] = r2(th, dx, dy)
    Xd, Yd, cdx, cdy, _ = cell_xy(fitset, mode, 'd')
    thd = fit(Xd, Yd)
    o['R2_pure_dose'] = r2(thd, cdx, cdy); o['theta_dose'] = thd.tolist()
    Xl, Yl, _, _, _ = cell_xy(lit, mode)
    thl = fit(Xl, Yl)
    o['theta_literal'] = thl.tolist()
    gaps_lit, gaps_fit = {}, {}
    for m in [6, 12, 18, 24, 30]:
        c = f'G@{m}'
        ids, labs = cells[c]['ids'], cells[c]['labs']
        x = np.asarray(P[c]['p'])[ids]; y = y_of(labs, mode).mean()
        gaps_lit[m] = 100 * (y - sig(thl, x).mean()); gaps_fit[m] = 100 * (y - sig(th, x).mean())
    o['c1_gap_literal'] = gaps_lit; o['c1_gap_fitset'] = gaps_fit
    o['c1_held'] = all(abs(v) <= 10 for v in gaps_lit.values())
    # held-out units
    held = {}
    for c in sorted(cells):
        if c.startswith(('NN_', 'FK', 'PM_', 'G_off_K', 'G_on@', 'G_off@')) and c in P:
            ids, labs = cells[c]['ids'], cells[c]['labs']
            x = np.asarray(P[c]['p'])[ids]
            held[c] = dict(n=len(labs), obs=round(100 * y_of(labs, mode).mean(), 1), pred=round(100 * sig(th, x).mean(), 1),
                           p=round(float(x.mean()), 4))
    o['held_out'] = held
    out[mode] = o
# null (i): permute cell p across FIT SET cells (keep within-cell prompt p shape by shifting means)
rng = np.random.default_rng(0)
X, Y, cx, cy, nn = cell_xy(fitset, 'comply')
r2null = []
for _ in range(2000):
    perm = rng.permutation(len(fitset))
    xs = np.repeat(cx[perm], nn)
    th = fit(xs, Y)
    r2null.append(r2(th, cx[perm], cy))
out['null_perm_R2'] = dict(p95=float(np.quantile(r2null, 0.95)), max=float(np.max(r2null)), mean=float(np.mean(r2null)),
                           frac_ge_obs=float(np.mean(np.array(r2null) >= out['comply']['R2_fit'])))
# null (ii): zero-push-along-r units at pin 30, judged
z = {c: round(100 * y_of(cells[c]['labs'], 'comply').mean(), 1) for c in cells if re.match(r'^(RN_\d+|RD_\d+|FK)@30$', c)}
out['p0_units_at_30'] = dict(n_units=len(z), max=max(z.values()), median=float(np.median(list(z.values()))), G_off=round(100 * y_of(cells['G_off@30']['labs'], 'comply').mean(), 1),
                             base=round(100 * y_of(cells['base']['labs'], 'comply').mean(), 1))
json.dump(out, open('vera_clauses.json', 'w'), indent=1, default=float)
for mode in ['comply', 'nonrefuse']:
    o = out[mode]
    print(mode, 'theta', np.round(o['theta'], 3), 'cells', o['n_cells'], 'replies', o['n_replies'], 'R2', round(o['R2_fit'], 3),
          'R2 dialed', round(o['R2_dialed_under_fitset_curve'], 3), 'R2 dose', round(o['R2_pure_dose'], 3))
    print('  c1 literal gaps', {k: round(v, 1) for k, v in o['c1_gap_literal'].items()}, 'theta_lit', np.round(o['theta_literal'], 3))
    print('  fitset gaps', {k: round(v, 1) for k, v in o['c1_gap_fitset'].items()})
    print('  held', o['held_out'])
print('null perm R2', out['null_perm_R2'])
print('p0 units at 30', out['p0_units_at_30'])
