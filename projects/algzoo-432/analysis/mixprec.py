"""T20 (Felix): per-entry mixed precision of the ORIGINAL weights. Zero fitted constants.

Every free number of the model in the +-1 gauge (W_hi = sign, W_oh row 0 shifted to 0; 400 numbers) gets
the fewest significant digits d in {0 (pruned), 1, 2, 3} that keeps agreement with the model on the
selection set (dev seed 61, 1e5) >= TARGET, else d = 4. Entries are visited from least to most sensitive
(agreement drop when pruned alone). The result is checked on dev seed 62 (2e5). Holdout only by Rasmus.

Code length: per kept entry 1 sign bit + log2(9 * 10^(d-1)) mantissa bits + exponent entropy, plus the
empirical entropy of the digit level d over all 400 slots (this replaces the sparsity mask), plus 16
W_hi sign bits.

Run: .venv/bin/python mixprec.py 0.989   ->  mixprec_<target>.npz / .json
"""
import json
import sys
import time

import numpy as np

from core import load_raw, run
from prune_ft import gauge

TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.989
W_raw = load_raw()
Wi, Wh, Wo = gauge(*W_raw)
XS = np.random.default_rng(61).standard_normal((100_000, 10))
XC = np.random.default_rng(62).standard_normal((200_000, 10))
RS = run(XS, *W_raw).argmax(1)
RC = run(XC, *W_raw).argmax(1)

flat = np.concatenate([Wh.ravel(), Wo.ravel()])
slots = [i for i in range(flat.size) if flat[i] != 0 and abs(flat[i]) != 1.0 and not (256 <= i < 256 + 16)]
assert len(slots) == 400, len(slots)


def rsig(v, d):
    if d == 0:
        return 0.0
    e = np.floor(np.log10(abs(v)))
    return float(np.round(v / 10 ** (e - d + 1)) * 10 ** (e - d + 1))


def unpack(f):
    return Wi, f[:256].reshape(16, 16), f[256:].reshape(10, 16)


def agree(f, X, R):
    return float((run(X, *unpack(f)).argmax(1) == R).mean())


def code_bits(f, digits):
    kept = [i for i in slots if digits[i] > 0]
    e = np.floor(np.log10(np.abs(f[kept]))).astype(int)
    _, c = np.unique(e, return_counts=True)
    p = c / c.sum()
    h_exp = float(-(p * np.log2(p)).sum())
    lv = np.array([digits[i] for i in slots])
    _, c2 = np.unique(lv, return_counts=True)
    p2 = c2 / c2.sum()
    h_lv = float(-(p2 * np.log2(p2)).sum())
    mant = sum(1 + np.log2(9 * 10 ** (digits[i] - 1)) for i in kept)
    return mant + h_exp * len(kept) + h_lv * len(slots) + 16


t0 = time.time()
cur = flat.copy()
for i in slots:
    cur[i] = rsig(flat[i], 4)
digits = {i: 4 for i in slots}
base = agree(cur, XS, RS)
sens = []
for i in slots:
    f = cur.copy(); f[i] = 0.0
    sens.append((base - agree(f, XS, RS), i))
sens.sort()
for _, i in sens:
    for d in (0, 1, 2, 3):
        f = cur.copy(); f[i] = rsig(flat[i], d)
        if agree(f, XS, RS) >= TARGET:
            cur, digits[i] = f, d
            break
lv = np.array([digits[i] for i in slots])
row = dict(target=TARGET, sel_agree=agree(cur, XS, RS), check_agree_dev62=agree(cur, XC, RC),
           kept=int((lv > 0).sum()), digit_hist={int(d): int((lv == d).sum()) for d in range(5)},
           bits=round(code_bits(cur, digits)), sec=round(time.time() - t0))
W = unpack(cur)
np.savez(f"mixprec_{TARGET}.npz", W_hi=W[0], W_hh=W[1], W_oh=W[2])
json.dump(row, open(f"mixprec_{TARGET}.json", "w"), indent=1)
print(json.dumps(row))
