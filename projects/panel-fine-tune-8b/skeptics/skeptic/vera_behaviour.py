"""Vera's own rebuild of the PREREG s6 behaviour bars from raw per-token arrays (not build code).

Bar (i): in each of the 298 affirmative eval pairs (src 'cities', split 'eval', same city, true vs false twin), the
right country beats its twin: summed log P of the continuation after the shared TOKEN prefix (Qwen3-8B tokenizer,
no special tokens). lp.npy has len(ids)-1 entries per statement: lp[j] = log P(token j+1 | tokens <= j).
Bar (ii): mean per-token KL over the 1192 eval statements, every position = (sum kl.npy rows + sum kl_pos0) /
(predictions + 1192). Also the pooled panel KL vs the s7 bound 0.0096.
Usage: python vera_behaviour.py DATA
"""
import glob, json, os, sys
import numpy as np
from transformers import AutoTokenizer

D = sys.argv[1]
rows = json.load(open(f"{D}/ref/split.json"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
ev = np.array([r["split"] == "eval" for r in rows])
pairs = {}
for i, r in enumerate(rows):
    if r["split"] == "eval" and r["src"] == "cities":
        pairs.setdefault(r["city"], {})[r["true"]] = i
pairs = [(p[1], p[0]) for p in pairs.values()]  # (true idx, false idx)
assert len(pairs) == 298 and ev.sum() == 1192


def shared(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


out = {}
for sd in sorted(glob.glob(f"{D}/states/*")):
    st = os.path.basename(sd)
    off = np.load(f"{sd}/off.npy"); lp = np.load(f"{sd}/lp.npy").astype(np.float64)
    assert all(off[i + 1] - off[i] == len(ids[i]) - 1 for i in range(len(rows))), st
    wins = 0
    for t, f in pairs:
        k = shared(ids[t], ids[f])
        st_t = lp[off[t] + k - 1: off[t + 1]].sum(); st_f = lp[off[f] + k - 1: off[f + 1]].sum()
        wins += st_t > st_f
    o = {"pairs_won": int(wins)}
    if os.path.exists(f"{sd}/kl.npy"):
        kl = np.load(f"{sd}/kl.npy").astype(np.float64); k0 = np.load(f"{sd}/kl_pos0.npy").astype(np.float64)
        s = sum(kl[off[i]:off[i + 1]].sum() for i in np.where(ev)[0]); n = sum(off[i + 1] - off[i] for i in np.where(ev)[0])
        o["eval_kl"] = (s + k0[ev].sum()) / (n + ev.sum())
        o["meta_kl"] = json.load(open(f"{sd}/meta.json")).get("kl_eval_mean_every_position")
    out[st] = o
    print(st, o, flush=True)
pk = [out[f"panel__lam10000__seed{s}"]["eval_kl"] for s in range(3)]
out["panel_pooled_eval_kl"] = float(np.mean(pk))
out["s7_kl_bound"] = 0.0096
print("pooled panel eval KL", np.mean(pk), "per seed", pk)
json.dump(out, open("behaviour8b.json", "w"), indent=1, default=float)
