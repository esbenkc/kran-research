"""Typical changed pair (Tove #17618, Vera #17617): same prefix, then after the first flipped token the rest of the
thought goes another way; both thoughts plain English. Prints candidates.
    ../../../env/.venv/bin/python figure1/pick_div.py data/reads/raw/start_real_test [N]"""
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
T = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
z = np.load(str(HERE / sys.argv[1]) + ".npz")
tn, tl = z["toks_n"], z["toks_l"]
d = (tn != tl)
ch = np.where(d.any(1))[0]
first = d.argmax(1)
same_after = np.array([(tn[i, first[i]:] == tl[i, first[i]:]).mean() for i in range(len(tn))])
print(f"changed {len(ch)}; share of changed pairs with >2 differing positions {(d[ch].sum(1) > 2).mean():.3f}; "
      f"median first flip {np.median(first[ch]):.0f}; median share of positions after the flip that differ {1 - np.median(same_after[ch]):.2f}")


def plain(s):
    return bool(re.fullmatch(r"[\x20-\x7e\n’]*", s)) and s.count("Answer") <= 1 and "target" not in s


c = []
for i in ch:
    a, b = T.decode(tn[i]), T.decode(tl[i])
    if plain(a) and plain(b) and 12 <= first[i] <= 40 and same_after[i] < 0.2:
        c.append((first[i], int(i), a, b))
c.sort(key=lambda r: -r[0])
print(len(c), "candidates")
for f, i, a, b in c[: int(sys.argv[2]) if len(sys.argv) > 2 else 10]:
    print(f"--- row {i} prompt {z['prompt_idx'][i]} draw {z['seed'][i]} first flip {f} c_n {z['c_n'][i]:.2f} c_l {z['c_l'][i]:.2f}")
    print("N:", repr(a)); print("L:", repr(b))
