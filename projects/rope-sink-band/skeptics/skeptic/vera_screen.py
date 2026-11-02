"""Vera pre-data screen (own CPU fp32 forward, no transformers): per head, how the position-0 sink logit splits
over the 64 rotate-half RoPE pairs (i, i+64). Two readings: RAW (q_t . k_0 per pair) and GAP (q_t . k_0 minus the mean
of q_t . k_s over the causal window s=0..t, per pair). Softmax only sees the GAP.
Usage: python vera_screen.py [--bos] [--n 8] [--T 96]
"""
import sys, json, math, argparse, time
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vera_qwen as V
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--bos", action="store_true", help="prepend <|endoftext|> as position 0")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--T", type=int, default=96)
ap.add_argument("--out", default="vera_screen")
a = ap.parse_args()
torch.set_grad_enabled(False)

TEXTS = [
    "The river had flooded twice that spring, and the farmers along its banks began to talk about moving their fences farther from the water. Nobody wanted to be the first to give up land that had been in the family for generations.",
    "In 1907 the city council approved a plan to build a tram line from the harbour to the university. Construction took four years, partly because the engineers had underestimated how soft the ground was near the old market.",
    "To compute the determinant of a triangular matrix, multiply the entries on its diagonal. This follows from expanding along the first column repeatedly, since every other term in the expansion contains a zero factor.",
    "She opened the letter slowly, half expecting bad news. Instead it was an invitation to a reunion of the choir she had sung in as a teenager, signed by a conductor she had assumed was long retired.",
    "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n",
    "Photosynthesis converts light energy into chemical energy stored in sugars. In most plants it happens in the chloroplasts of leaf cells, where chlorophyll absorbs mostly blue and red light and reflects green.",
    "The committee met on Thursday to review the budget. Three members argued that the maintenance fund was too small, while the treasurer insisted that any increase would have to come from cuts elsewhere in the plan.",
    "When the train stopped in the middle of the forest, the passengers looked at each other in silence. After ten minutes a conductor walked through the carriages and explained that a tree had fallen across the track ahead.",
    "Large language models are trained to predict the next token of text. During training they see billions of examples, and the weights slowly change so that the predicted probabilities match the observed continuations better.",
    "Mix the flour and salt in a large bowl, then add the water a little at a time. Knead the dough for about ten minutes until it is smooth and elastic, cover it, and let it rest in a warm place for an hour.",
]
tok = AutoTokenizer.from_pretrained(V.PATHS["1.7B"])
ids = []
for i in range(a.n):  # prompt i = the 10 texts concatenated in a rotated order (no repetition inside a prompt)
    order = TEXTS[i % len(TEXTS):] + TEXTS[: i % len(TEXTS)]
    x = tok("\n\n".join(order), add_special_tokens=False)["input_ids"]
    if a.bos:
        x = [151643] + x
    assert len(x) >= a.T, len(x)
    ids.append(x[: a.T])
ids = torch.tensor(ids)
B, T = ids.shape
W = V.Weights("1.7B")
uniq = sorted(set(ids.flatten().tolist()))
E = W.get_rows("model.embed_tokens.weight", uniq)
lut = {u: i for i, u in enumerate(uniq)}
x = E[torch.tensor([[lut[v] for v in r] for r in ids.tolist()])]
L_N, d, H = 28, V.d, V.H
half = d // 2
t0 = 4  # skip the first few queries (window too short)
sink = np.zeros((L_N, H))  # mean attention to pos 0 over t >= t0
sink1 = np.zeros((L_N, H)); sinkmax = np.zeros((L_N, H))
raw = np.zeros((L_N, H, half))  # mean per-pair q_t.k_0 / sqrt d
gap = np.zeros((L_N, H, half))  # mean per-pair (q_t.k_0 - mean_s q_t.k_s) / sqrt d
qn = np.zeros((L_N, H, half)); kn0 = np.zeros((L_N, H, half))
t_start = time.time()
for L in range(L_N):
    w = W.layer(L)
    P = V.attn_parts(w, x)
    qr, kr = P["qr"], P["kr"].repeat_interleave(H // V.HKV, 1)  # B,H,T,d
    a_ = P["a"]
    sink[L] = a_[:, :, t0:, 0].mean((0, 2)).numpy()
    sink1[L] = a_[:, :, t0:, 1].mean((0, 2)).numpy()
    sinkmax[L] = a_[:, :, t0:, 1:].max(-1).values.mean((0, 2)).numpy()
    # per-pair products: c[b,h,t,s,i] = qr[t,i]kr[s,i] + qr[t,i+64]kr[s,i+64]
    q1, q2 = qr[..., :half], qr[..., half:]
    k1, k2 = kr[..., :half], kr[..., half:]
    c0 = (q1 * k1[:, :, :1] + q2 * k2[:, :, :1]) / math.sqrt(d)  # B,H,T,64 (key = pos 0)
    # mean over causal keys s<=t of per-pair product: prefix sums of k
    cs1, cs2 = k1.cumsum(2), k2.cumsum(2)
    cnt = torch.arange(1, T + 1, dtype=torch.float32)[None, None, :, None]
    cm = (q1 * cs1 / cnt + q2 * cs2 / cnt) / math.sqrt(d)
    raw[L] = c0[:, :, t0:].mean((0, 2)).numpy()
    gap[L] = (c0 - cm)[:, :, t0:].mean((0, 2)).numpy()
    qn[L] = (q1 ** 2 + q2 ** 2).sqrt()[:, :, t0:].mean((0, 2)).numpy()
    kn0[L] = (k1[:, :, 0] ** 2 + k2[:, :, 0] ** 2).sqrt().mean(0).numpy()
    # sanity: summed pairs reproduce the logit
    z = P["z"][:, :, t0:, 0]
    assert torch.allclose(c0[:, :, t0:].sum(-1), z, atol=1e-3), "pair split does not sum to logit"
    x, _, _ = V.block(w, x)
    print(f"L{L:2d} sink(mean over heads) {sink[L].mean():.3f} max {sink[L].max():.3f}  {time.time()-t_start:.0f}s", flush=True)
np.savez(f"{a.out}{'_bos' if a.bos else ''}.npz", sink=sink, sink1=sink1, sinkmax=sinkmax, raw=raw, gap=gap, qn=qn, kn0=kn0, ids=ids.numpy())


def band(v, frac=0.5):
    """smallest set of pairs (largest positive first) whose summed contribution >= frac * total"""
    tot = v.sum()
    o = np.argsort(-v)
    cs = np.cumsum(v[o])
    k = int(np.searchsorted(cs, frac * tot) + 1) if tot > 0 else None
    return tot, k, sorted(o[:k].tolist()) if k else None


sinkheads = sink > 0.3
print("mean attn to pos1", sink1.mean().round(3), "| mean max attn to any other key", sinkmax.mean().round(3))
print("sink heads (score > 0.3):", int(sinkheads.sum()), "of", L_N * H, "| mean sink over all heads", sink.mean().round(3))
for nm, M in [("RAW", raw), ("GAP", gap)]:
    agg = M[sinkheads].mean(0)
    tot, k, b = band(agg)
    print(f"{nm} pooled over sink heads: total {tot:.2f}, pairs for 50%: {k}, band {b}")
    agg2 = (M * sink[..., None]).sum((0, 1)) / sink.sum()
    tot, k, b = band(agg2)
    print(f"{nm} sink-weighted over all heads: total {tot:.2f}, pairs for 50%: {k}, band {b}")
    ks = [band(M[l, h])[1] for l, h in zip(*np.nonzero(sinkheads))]
    print(f"{nm} per-sink-head pairs for 50%: median {np.median([k for k in ks if k]):.0f}, range {min(k for k in ks if k)}-{max(k for k in ks if k)}")
