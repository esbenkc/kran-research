"""Vera: cue exposure per run = text difference D x 32 probed (watched) rows per update, summed over updates
(trapezoid over the dev_D reads in curve.jsonl, every 50 updates, so about +-30%), next to the training gap
(unprobed - probed training cheat, points) per 25-update window.
usage: vera_exposure.py <project dir>  -> vera_exposure.json beside this script
"""
import json, os, sys
import numpy as np

proj = sys.argv[1]
out = {}
for run in ('planted_s0', 'dense_s0'):
    L = [json.loads(l) for l in open(os.path.join(proj, 'data/runs', run, 'curve.jsonl'))]
    pts = [(d['step'], d['dev_D']) for d in L if d.get('dev_D') is not None]
    s = np.array([p[0] for p in pts], float); D = np.array([p[1] for p in pts])
    cum = np.concatenate([[0], np.cumsum(np.diff(s) * (D[1:] + D[:-1]) / 2 * 32)])
    gaps = {d['step']: round(100 * (d['train_cheat_unprobed'] - d['train_cheat_probed']), 1)
            for d in L if 'train_cheat_unprobed' in d}
    out[run] = dict(dev_D={int(a): round(float(b), 3) for a, b in zip(s, D)},
                    cue_marked_thoughts={int(a): int(round(c)) for a, c in zip(s, cum)},
                    train_gap_pts_by_window_end=gaps)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_exposure.json'), 'w'), indent=1)
