"""Build the blind judge input from the activation dump (see ../PREREGISTRATION.md).

Unit selection (identical rule for every arm and layer, fixed before any context is read):
  eligible := the unit's 20th-best deduplicated excerpt has a positive max (positive max on >= 20 distinct documents);
  order := np.random.default_rng(0).permutation(n_units); take the first 50 eligible units in that order.
Per selected unit (writes data/contexts.jsonl, research's contract in code/judge/judge.py):
  examples = ranks 1-20, heldout = ranks 21-40 (deduplicated, at most one excerpt per document),
  random   = 20 random corpus windows (seeded per unit), marked position >= 4, from documents not in the unit's top-40.
  excerpt  = {left: 32 tokens before the marked token, tok, right: 31 tokens after (clipped to the 256-token sequence),
              act, regime (gate sign + up sign at the max, neuron arms only), doc, seq, pos}
data/units.json: per unit stats (firing rate, mean, std, kurtosis, density above half max, sink log, regimes).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("AUDIT_DATA", os.path.join(HERE, "..", "data"))
ACTS = os.path.join(D, "acts")
N_PER, N_TOP, N_HELD, N_RAND, LEFT, RIGHT = 50, 20, 20, 20, 32, 31


def window(tok, toks, seq, P):
    ids = toks[seq]
    return dict(left=tok.decode(ids[max(0, P - LEFT):P]), tok=tok.decode(ids[P:P + 1]),
                right=tok.decode(ids[P + 1:P + 1 + RIGHT]))


def main():
    meta = json.load(open(os.path.join(ACTS, "meta.json")))
    toks = np.load(os.path.join(ACTS, f"tokens_{meta['n_tok']}.npy"))
    seq_doc = np.load(os.path.join(ACTS, f"seq_doc_{meta['n_tok']}.npy"))
    tok = AutoTokenizer.from_pretrained(meta["model"])
    seq, exc = meta["seq"], meta["exc"]
    nper = seq // exc
    lo_edges = np.array([-np.inf] + meta["edges_log10"])  # lower log10 edge of each histogram bin
    lines, stats = [], []
    for arm in ["A", "B", "C"]:
        if not os.path.exists(os.path.join(ACTS, f"arm{arm}.npz")):
            print("arm", arm, "not dumped yet, skipped")
            continue
        z = np.load(os.path.join(ACTS, f"arm{arm}.npz"))
        for L in meta["layers"]:
            tv = z[f"L{L}_top_val"]
            U = tv.shape[0]
            elig = np.isfinite(tv[:, N_TOP - 1]) & (tv[:, N_TOP - 1] > 0)
            order = np.random.default_rng(0).permutation(U)
            pick = [int(u) for u in order if elig[u]][:N_PER]
            for u in pick:
                uid = hashlib.sha256(f"{arm}-{L}-{u}-audit".encode()).hexdigest()[:10]
                ex = []
                for r in range(N_TOP + N_HELD):
                    v = float(tv[u, r])
                    if not np.isfinite(v) or v <= 0:
                        break
                    e, p = int(z[f"L{L}_top_exc"][u, r]), int(z[f"L{L}_top_pos"][u, r])
                    s_, P = e // nper, (e % nper) * exc + p
                    w = window(tok, toks, s_, P)
                    reg = None
                    if f"L{L}_top_gate_sign" in z.files:
                        g, q = int(z[f"L{L}_top_gate_sign"][u, r]), int(z[f"L{L}_top_up_sign"][u, r])
                        reg = ("+" if g > 0 else "-") + ("+" if q > 0 else "-")
                    ex.append(dict(**w, act=v, regime=reg, doc=int(seq_doc[s_]), seq=int(s_), pos=int(P)))
                docs = {x["doc"] for x in ex}
                rng = np.random.default_rng(int(hashlib.sha256(uid.encode()).hexdigest()[:12], 16))
                ok = np.where(~np.isin(seq_doc, list(docs)))[0]
                rnd = []
                for s_ in rng.choice(ok, size=N_RAND, replace=False):
                    P = int(rng.integers(meta["mask_first"], seq))
                    rnd.append(dict(**window(tok, toks, int(s_), P), act=None, regime=None, doc=int(seq_doc[s_]),
                                    seq=int(s_), pos=P))
                lines.append(dict(uid=uid, arm=arm, layer=L, unit=u, examples=ex[:N_TOP], heldout=ex[N_TOP:],
                                  random=rnd))
                n, s1, s2, s3, s4 = z[f"L{L}_moments"][:, u]
                mu = s1 / n
                var = max(s2 / n - mu ** 2, 1e-30)
                m4c = s4 / n - 4 * mu * s3 / n + 6 * mu ** 2 * s2 / n - 3 * mu ** 4
                hist = z[f"L{L}_hist"][u]
                vmax = float(tv[u, 0])
                stats.append(dict(uid=uid, arm=arm, layer=L, unit=u, n_eligible_layer=int(elig.sum()), n_units_layer=U,
                                  vmax=vmax, top20=float(tv[u, N_TOP - 1]), n_heldout=len(ex) - N_TOP,
                                  mean=float(mu), std=float(np.sqrt(var)), kurtosis=float(m4c / var ** 2),
                                  firing_rate=float(hist[1:].sum() / hist.sum()),
                                  density_half_max=float(hist[lo_edges >= np.log10(0.5 * vmax)].sum() / hist.sum()),
                                  sink_n=int(z[f"L{L}_sink_n"][u]), sink_max=float(z[f"L{L}_sink_max"][u]),
                                  sink_would_enter_top20=bool(z[f"L{L}_sink_max"][u] > tv[u, N_TOP - 1]),
                                  regimes=[x["regime"] for x in ex[:N_TOP]]))
            print(arm, L, "eligible", int(elig.sum()), "of", U, "picked", len(pick), flush=True)
    with open(os.path.join(D, "contexts.jsonl"), "w") as f:
        for x in lines:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump(stats, open(os.path.join(D, "units.json"), "w"), indent=0)
    print(len(lines), "units ->", os.path.join(D, "contexts.jsonl"))


if __name__ == "__main__":
    main()
