"""Self-test of c_reader.fit_c on a toy LM with Gumbel CRN sampling (no model, ~seconds).
Case A hash: leaky logits = none logits + a per-(context, token) random +-0.125 shift on ~half the vocab
  (like one bf16 step at a coin-flip sign). Expect C ~ 0 even though the true TV is far from 0.
Case B planted: leaky adds eps to one token at position 0, eps set so per-position TV = 0.10. Expect C ~ 0.10.
Case C null: leaky == none. Expect C = 0 exactly (all pairs identical)."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from c_reader import fit_c

V, k, P_TRAIN, P_TEST, R = 60, 16, 382, 200, 4
rng = np.random.default_rng(1)
W = rng.normal(0, 1.5, (V + 1, V))          # toy bigram LM: logits = W[prev]


def ctx_hash(prompt, t, prev, salt):
    h = np.random.default_rng(abs(hash((prompt, t, prev, salt))) % (2 ** 32))
    return h.choice([-0.125, 0.0, 0.125], size=V, p=[0.25, 0.5, 0.25])


def sample(prompt, seed, mode, eps=0.0, tok0=None):
    g = np.random.default_rng(10_000 * prompt + seed)
    gum = g.gumbel(size=(k, V))
    prev = V if prompt % 2 else (prompt % V)
    out, lr = [], 0.0
    base = W[V] + 0.3 * np.random.default_rng(prompt).normal(size=V)   # prompt-specific start
    for t in range(k):
        lg = (base if t == 0 else W[prev]).copy()
        lq = lg.copy()
        if mode == "hash":
            lq = lg + ctx_hash(prompt, t, prev, 7)
        if mode == "plant" and t == 0:
            lq[tok0] += eps
        use = lq if mode != "none" else lg
        x = int(np.argmax(use + gum[t]))
        out.append(x); prev = x
    return out


def toks(mode, **kw):
    tn, tl, pr, sp = [], [], [], []
    for p in range(P_TRAIN + P_TEST):
        for s in range(R):
            tn.append(sample(p, s, "none")); tl.append(sample(p, s, mode, **kw))
            pr.append(p); sp.append(p < P_TRAIN)
    return np.array(tn), np.array(tl), np.array(pr), np.array(sp)


def tv_plant(eps, tok0):
    # mean over prompts of position-0 TV when adding eps to tok0
    tvs = []
    for p in range(200):
        lg = W[V] + 0.3 * np.random.default_rng(p).normal(size=V)
        pa = np.exp(lg - lg.max()); pa /= pa.sum()
        lq = lg.copy(); lq[tok0] += eps
        qa = np.exp(lq - lq.max()); qa /= qa.sum()
        tvs.append(0.5 * np.abs(pa - qa).sum())
    return float(np.mean(tvs))


p0 = np.exp(W[V] - W[V].max()); p0 /= p0.sum()
tok0 = int(np.argmin(np.abs(p0 - 0.2)))
lo, hi = 0.0, 5.0
for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if tv_plant(mid, tok0) < 0.10 else (lo, mid)
eps = (lo + hi) / 2
res = {}
for name, kw in [("null", dict(mode="none")), ("hash", dict(mode="hash")), ("plant", dict(mode="plant", eps=eps, tok0=tok0))]:
    tn, tl, pr, sp = toks(**kw)
    res[name] = fit_c(tn, tl, pr, sp, n_boot=300)
    print(name, json.dumps({k_: res[name][k_] for k_ in ["C", "C_ci", "G_pair_points", "G_pair_ci", "f_test", "reg"]}))
res["plant_tv_target"] = 0.10; res["plant_eps"] = eps; res["plant_tok0_p"] = float(p0[tok0])
json.dump(res, open(os.path.join(os.path.dirname(__file__), "c_reader_selftest.json"), "w"), indent=1)
