"""Vera: cue exposure per run = text difference D x 32 probed (watched) rows per update, summed over updates
(trapezoid over the dev_D reads in curve.jsonl, every 50 updates, so about +-30%), next to the training gap
(unprobed - probed training cheat, points) per 25-update window.
usage: vera_exposure.py <project dir>  -> vera_exposure.json beside this script
"""
import json, os, sys
import numpy as np

proj = sys.argv[1]
out = {}
for run in ('planted_s0', 'dense_s0', 'dense_s1'):
    path = os.path.join(proj, 'data/runs', run, 'curve.jsonl')
    if not os.path.exists(path):
        continue
    L = [json.loads(l) for l in open(path)]
    pts = [(d['step'], d['dev_D']) for d in L if d.get('dev_D') is not None]
    s = np.array([p[0] for p in pts], float); D = np.array([p[1] for p in pts])
    cum = np.concatenate([[0], np.cumsum(np.diff(s) * (D[1:] + D[:-1]) / 2 * 32)])
    gaps = {d['step']: round(100 * (d['train_cheat_unprobed'] - d['train_cheat_probed']), 1)
            for d in L if 'train_cheat_unprobed' in d}

    def marked(step):  # cumulative marked thoughts at an update; past the last D read, extend with the last D
        return float(np.interp(step, s, cum)) + max(0.0, step - s[-1]) * D[-1] * 32

    # DEVIATIONS D5 onset: first 25-update window with training gap >= 7 pts, and its marked-thought range
    onset = None
    for st in sorted(gaps):
        if gaps[st] >= 7:
            lo, hi = marked(st - 25), marked(st)
            onset = dict(window_end=int(st), gap=gaps[st], marked_from=int(round(lo)), marked_to=int(round(hi)),
                         in_1000_3000_strict=bool(lo >= 1000 and hi <= 3000),
                         in_1000_3000_overlap=bool(hi >= 1000 and lo <= 3000))
            break
    out[run] = dict(dev_D={int(a): round(float(b), 3) for a, b in zip(s, D)},
                    cue_marked_thoughts={int(a): int(round(c)) for a, c in zip(s, cum)},
                    train_gap_pts_by_window_end=gaps, onset=onset)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vera_exposure.json'), 'w'), indent=1)
