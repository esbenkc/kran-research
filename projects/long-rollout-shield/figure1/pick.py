"""List candidate example rows for Figure 1 (Owain): a test puzzle whose 64-token thought, written with the same
random draw, changed only a little when the probe was on (a word or two), with both thoughts in plain English.

    ../../../env/.venv/bin/python figure1/pick.py data/parent/start_k64 [N]
"""
import difflib
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
T = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def plain(s):
    return bool(re.fullmatch(r"[\x20-\x7e\n’]*", s)) and s.count("Answer") <= 1


def word_diff(a, b):
    sa, sb = a.split(), b.split()
    ops = [o for o in difflib.SequenceMatcher(None, sa, sb).get_opcodes() if o[0] != "equal"]
    n = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    return n, ops, sa, sb


base = sys.argv[1]
z = np.load(str(HERE / base) + ".npz")
tn, tl = z["toks_n"], z["toks_l"]
rows = []
for i in np.where((tn != tl).any(1))[0]:
    a, b = T.decode(tn[i]), T.decode(tl[i])
    if not (plain(a) and plain(b)):
        continue
    n, ops, sa, sb = word_diff(a, b)
    rows.append((n, int(i), a, b, ops))
rows.sort()
print(f"{len(rows)} plain flipped rows of {int((tn != tl).any(1).sum())} flipped / {len(tn)}")
for n, i, a, b, ops in rows[: int(sys.argv[2]) if len(sys.argv) > 2 else 8]:
    print(f"--- row {i} prompt {z['prompt_idx'][i]} seed {z['seed'][i]} words changed {n} c_n {z['c_n'][i]:.2f} c_l {z['c_l'][i]:.2f}")
    print("N:", repr(a))
    print("L:", repr(b))
