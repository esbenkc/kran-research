"""Iris: which route did a fine-tuned state take against the max-panel? (Tove #4913)
Reads data/scores/<state>.npz (tok1..29 + panelmax) and the country-token rows of data/acts/<state>/tok.npy at
readouts 28 and 29, for the state and for clean. Reports:
- panel line (2nd-highest of the 200 line-set true) in the state vs clean; which readout gives the max z for the two
  highest line-set true statements (the statements that set the line); panel catch (own line, clean line);
- per readout: own-line catch, line shift (z units) and the false/true median-z gap;
- readout 28 norms at the country token: median ||x|| of eval false and eval true, ratio to clean; for the line-set true
  statements that set the line: their ||x|| ratio to their clean ||x|| (inflation route) and their z28 before/after;
- direction part of tok28: a = (w/sd).x / ||x|| (scale-free) median for false/true vs clean (shrink route moves the
  norm, not a/||x||).
Run from projects/panel-fine-tune : ../../../venv1021/bin/python iris_checks/route.py <state> [OUT.json]"""
import json, sys
import numpy as np
sys.path.insert(0, '.')
from common import Acts, pool_idx, rows, threshold
import os, common
from pathlib import Path
if os.environ.get('ACTS_DIR'):
    common.ACTS = Path(os.environ['ACTS_DIR'])  # e.g. the 1021 project's data/acts for the 1021 states

st = sys.argv[1]
F = np.load('data/fits/panel.npz'); Z = json.load(open('data/fits/zmoments.json'))
PANEL = list(range(10, 30))
_, y, _, _ = rows(); tr, line_idx, ev = pool_idx()
evf, evt = ev[y[ev] == 1], ev[y[ev] == 0]
rws = F['tok_rows_clean']


def zs(S):
    return {R: (S[f'tok{R}'] - Z[f'tok{R}']['mu']) / Z[f'tok{R}']['sd'] for R in PANEL}


S0, S1 = np.load('data/scores/clean.npz'), np.load(f'data/scores/{st}.npz')
z0, z1 = zs(S0), zs(S1)
M0 = np.max(np.stack([z0[R] for R in PANEL], 1), 1); M1 = np.max(np.stack([z1[R] for R in PANEL], 1), 1)
A1 = np.stack([z1[R] for R in PANEL], 1)
l0, l1 = threshold(M0[line_idx]), threshold(M1[line_idx])
top2 = line_idx[np.argsort(M1[line_idx])[-2:]]
out = dict(state=st, panel_line_clean_state=l0, panel_line_state=l1,
           panel_catch_own_line=float((M1[evf] > l1).mean()), panel_catch_clean_line=float((M1[evf] > l0).mean()),
           line_setting_true=[dict(row=int(i), panel_z=float(M1[i]), argmax_readout=int(PANEL[int(np.argmax(A1[i]))]),
                                   clean_panel_z=float(M0[i])) for i in top2])
per = {}
for R in PANEL:
    lr0, lr1 = threshold(z0[R][line_idx]), threshold(z1[R][line_idx])
    per[R] = dict(catch_own=round(float((z1[R][evf] > lr1).mean()), 4), line_shift_z=round(lr1 - lr0, 3),
                  gap_med_z=round(float(np.median(z1[R][evf]) - np.median(z1[R][evt])), 3),
                  gap_med_z_clean=round(float(np.median(z0[R][evf]) - np.median(z0[R][evt])), 3))
out['per_readout'] = per
X0 = Acts('clean').at(28, rws).astype(np.float64); X1 = Acts(st).at(28, rws).astype(np.float64)
n0, n1 = np.linalg.norm(X0, axis=1), np.linalg.norm(X1, axis=1)
ws = F['tok28__w'] / F['tok28__sd']
d0, d1 = X0 @ ws / n0, X1 @ ws / n1
out['r28'] = dict(norm_ratio_med_false=float(np.median(n1[evf] / n0[evf])), norm_ratio_med_true=float(np.median(n1[evt] / n0[evt])),
                  norm_ratio_line_setters=[float(n1[i] / n0[i]) for i in top2],
                  dir_med_false_clean=float(np.median(d0[evf])), dir_med_false=float(np.median(d1[evf])),
                  dir_med_true_clean=float(np.median(d0[evt])), dir_med_true=float(np.median(d1[evt])),
                  z28_line_setters_clean=[float(z0[28][i]) for i in top2], z28_line_setters=[float(z1[28][i]) for i in top2])
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
