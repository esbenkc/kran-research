#!/usr/bin/env python3
import csv, json, math, os
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]
RUNS = BASE / 'data' / 'runs'

def cos(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

def boot_ci(x):
    x = np.asarray(x)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(10000, len(x)))
    vals = np.median(x[idx], axis=1)
    return np.percentile(vals, [2.5, 97.5])

def med_csv(path, col, layers):
    rows = list(csv.DictReader(open(path)))
    return [float(np.median([float(r[col]) for r in rows if int(r['L']) == L])) for L in layers]

def med_csv_scaled(path, col, layers, scale):
    return [float(np.median([float(r[col]) for r in csv.DictReader(open(path)) if int(r['L']) == L]) * scale) for L in layers]

s1 = np.load(RUNS / 'qwen3_A_s1.npz')
s2 = np.load(RUNS / 'qwen3_A_s2.npz')
reading = np.load(RUNS / 'qwen3_A_reading.npz')
print('Qwen3 A status from raw vectors')
yes = 0
for L in [19, 21, 23, 27]:
    c = cos(s1[f'L{L}_frz'], s2[f'L{L}_real'])
    lo, hi = boot_ci(c)
    cp = cos(s1[f'L{L}_frz'], s1[f'L{L}_pred'])
    cr = cos(s1[f'L{L}_pred'], s2[f'L{L}_real'])
    lp = cos(reading[f'L{L}_logprob_frz'], reading[f'L{L}_logprob_real'])
    if L in [19, 21, 23] and hi < 0.90:
        yes += 1
    print(f'L{L}: med={np.median(c):.4f} ci=[{lo:.4f},{hi:.4f}] pred_frz={np.median(cp):.4f} pred_real={np.median(cr):.4f} logprob={np.median(lp):.4f}')
print(f'status_yes_layers={yes}; confirmed={yes >= 2}')
for set_name in ['B', 'C']:
    vals = med_csv(RUNS / f'qwen3_{set_name}.csv', 'c_frz', [19, 21, 23])
    print(f'Qwen3 {set_name} medians=' + ','.join(f'{v:.4f}' for v in vals))
rows = list(csv.DictReader(open(RUNS / 'qwen3_A_reading.csv')))
print('top10_logit=' + ','.join(f'{np.median([float(r["top10_overlap_centred"]) for r in rows if int(r["L"]) == L]) * 10:.1f}' for L in [19,21,23]))
print('top10_logprob=' + ','.join(f'{np.median([float(r["top10_overlap_logprob"]) for r in rows if int(r["L"]) == L]) * 10:.1f}' for L in [19,21,23]))
for r in rows:
    if int(r['doc']) == 8201 and int(r['L']) == 19:
        print('doc8201_L19 cos_centred={:.4f} cos_logprob={:.4f}'.format(float(r['cos_centred']), float(r['cos_logprob'])))
        break
gpt2 = RUNS / 'gpt2_A.csv'
for col in ['cos_ln-frozen_attn-frozen_mlp-tc', 'cos_ln-live_attn-frozen_mlp-tc', 'cos_ln-frozen_attn-live_mlp-real', 'cos_ln-live_attn-live_mlp-tc']:
    vals = med_csv(gpt2, col, [8, 9, 10])
    print(col + '=' + ','.join(f'{v:.3f}' for v in vals))
