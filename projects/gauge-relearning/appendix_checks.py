"""Appendix check (If you are curious): verify K_c = K_rest + K_Q/c^2 + c^2 K_K for per-fact gradient Gram matrices on Pythia-160M."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
FACTS = [
    ("Daruko Mivera lives in the city of", " Oslo"),
    ("Telma Ostrand lives in the city of", " Paris"),
    ("Joren Vask lives in the city of", " Tokyo"),
    ("Ilsa Merrow lives in the city of", " Cairo"),
]
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)


def gauge(m, c):
    H = m.config.num_attention_heads
    d = m.config.hidden_size // H
    with torch.no_grad():
        for L in m.gpt_neox.layers:
            lin = L.attention.query_key_value
            W, b = lin.weight.view(H, 3, d, -1), lin.bias.view(H, 3, d)
            W[:, 0] *= c; b[:, 0] *= c; W[:, 1] /= c; b[:, 1] /= c


def block_grads(m):
    """Per fact: dict block -> flat grad, blocks = Q, K, rest."""
    H = m.config.num_attention_heads
    d = m.config.hidden_size // H
    out = []
    for p, a in FACTS:
        m.zero_grad()
        ids = tok(p + a, return_tensors="pt").input_ids
        n = len(tok(a).input_ids)
        logits = m(ids).logits[0, -n - 1:-1]
        torch.nn.functional.cross_entropy(logits, ids[0, -n:]).backward()
        q, k, rest = [], [], []
        for name, prm in m.named_parameters():
            g = prm.grad
            if "query_key_value" in name:
                g = g.view(H, 3, d, -1) if g.dim() == 2 else g.view(H, 3, d)
                q.append(g[:, 0].flatten()); k.append(g[:, 1].flatten()); rest.append(g[:, 2].flatten())
            else:
                rest.append(g.flatten())
        out.append({"Q": torch.cat(q), "K": torch.cat(k), "rest": torch.cat(rest)})
    return out


def gram(gs, blk):
    G = torch.stack([g[blk] for g in gs]).double()
    return G @ G.T


m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
g1 = block_grads(m)
Kq, Kk, Kr = gram(g1, "Q"), gram(g1, "K"), gram(g1, "rest")
print("c=1 share of trace: Q %.4f K %.4f rest %.4f" % tuple((x.trace() / (Kq + Kk + Kr).trace()).item() for x in (Kq, Kk, Kr)))
for c in (16.0, 1 / 16):
    mc = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    gauge(mc, c)
    gc = block_grads(mc)
    full = gram(gc, "Q") + gram(gc, "K") + gram(gc, "rest")
    pred = Kr + Kq / c**2 + c**2 * Kk
    rel = ((full - pred).norm() / full.norm()).item()
    print(f"c={c:g}: rel err measured vs formula {rel:.2e}; K-block share {(c**2*Kk).trace().item()/pred.trace().item():.4f}")


# Error bars for SGD at c = 16: per-run slowdown ratios (3 seeds x 3 data orders), from data/runs.jsonl
# via the shared per-run denominator used in the figure (values as in data.json / skeptic rebuilds).
import math
import numpy as np
from scipy import stats

x = np.array([2.81, 0.86, 4.63, 1.55, 0.84, 2.94, 2.27, 5.92, 6.32])
lx = np.log(x)
print(f"SGD c=16: {int((x > 1).sum())}/{len(x)} slower, CP 95% CI {stats.binomtest(int((x > 1).sum()), len(x)).proportion_ci()}")
for n in (9, 100):
    se = lx.std(ddof=1) / math.sqrt(n)
    print(f"n={n}: geo-mean {math.exp(lx.mean()):.2f}x, 95% CI {math.exp(lx.mean() - 1.96 * se):.2f}-{math.exp(lx.mean() + 1.96 * se):.2f}x"
          f"; slower-rate CI at 78%: {stats.binomtest(round(0.78 * n), n).proportion_ci()}")
