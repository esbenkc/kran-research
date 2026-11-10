"""Owain: find one test puzzle (same prompt, same random draw) whose probe-on thought opens with ' Need' and whose
probe-off thought does not, in BOTH the planted read (update 200) and the dense read (update 200). Rank by how
typical each row's own switch (c_n - c_l) is among that read's differing pairs, then by plain short text.

    ../../../env/.venv/bin/python figure1/pick.py
"""
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
X = 14656  # ' Need' (data/reads/raw/*.json plant_tok)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
assert tok.decode([X]) == " Need"


def load(name):
    js = json.load(open(HERE / f"data/reads/raw/{name}.json"))
    z = dict(np.load(HERE / f"data/reads/raw/{name}.npz"))
    assert js["plant_tok"] == X
    assert abs((z["c_n"] - z["c_l"]).mean() - js["test"]["gap"]) < 1e-6
    return z, js


zp, _ = load("planted_s0_gate_200")
zd, _ = load("dense_s0_gate_200")
assert (zp["prompt_idx"] == zd["prompt_idx"]).all() and (zp["seed"] == zd["seed"]).all()


def marked(z):
    return (z["toks_l"][:, 0] == X) & (z["toks_n"][:, 0] != X)


def plain(s):
    return bool(re.fullmatch(r"[\x20-\x7e\n’]*", s))


both = np.where(marked(zp) & marked(zd))[0]
sp, sd = zp["c_n"] - zp["c_l"], zd["c_n"] - zd["c_l"]
mp, md = np.median(sp[marked(zp)]), np.median(sd[marked(zd)])
print(f"marked pairs: planted {marked(zp).sum()} / dense {marked(zd).sum()} / both {len(both)}")
print(f"median own switch among marked: planted {100*mp:+.1f}, dense {100*md:+.1f}")
rows = []
for i in both:
    texts = [tok.decode(z[k][i]) for z in (zp, zd) for k in ("toks_n", "toks_l")]
    if not all(plain(t) for t in texts):
        continue
    score = abs(sp[i] - mp) / (np.std(sp[marked(zp)]) + 1e-9) + abs(sd[i] - md) / (np.std(sd[marked(zd)]) + 1e-9)
    rows.append((score, int(i)))
for score, i in sorted(rows)[:8]:
    print(f"\nrow {i} prompt {zp['prompt_idx'][i]} seed {zp['seed'][i]} score {score:.2f}")
    for nm, z, s in (("planted", zp, sp), ("dense", zd, sd)):
        print(f"  {nm}: off {100*z['c_n'][i]:.0f}% on {100*z['c_l'][i]:.0f}% switch {100*s[i]:+.1f}")
        print("    off:", repr(tok.decode(z["toks_n"][i])[:110]))
        print("    on: ", repr(tok.decode(z["toks_l"][i])[:110]))
