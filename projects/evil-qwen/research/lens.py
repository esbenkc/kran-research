"""Logit lens of lyra's steered neuron write vectors (sign x W_down[:, n]) in Qwen3.5-4B.

Reads only the needed tensors from the MLX 8-bit safetensors (lazy), no generation, no model process.
Unembed = tied embed_tokens (8-bit, dequantized in chunks). Final norm gain g applied: logits = E @ (g * v).
Also: cosines between all signed write vectors, vs a random same-layer baseline.
Out: research_checks/lens.json, lens.md
"""
import glob, json, sys
from pathlib import Path
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

SNAP = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-8bit/snapshots/*"))[0]
OUT = Path(__file__).parent
W = mx.load(f"{SNAP}/model.safetensors")
P = "language_model.model."
cfg = json.load(open(f"{SNAP}/config.json"))
q = cfg.get("quantization", {"group_size": 64, "bits": 8})
GS, BITS = q["group_size"], q["bits"]

NEURONS = [  # (L, n, sign, recipe)
    (14, 8949, +1, "evil"), (13, 7956, -1, "evil"),
    (15, 3920, +1, "sarcastiqwen"), (12, 2920, -1, "sycophancy"), (16, 8168, -1, "sycophancy alt"),
    (10, 6295, +1, "condolences"), (11, 6261, +1, "chill friend"), (17, 7870, -1, "chill friend"),
    (18, 5796, +1, "combo"),
]


def deq(name):
    return mx.dequantize(W[name + ".weight"], W[name + ".scales"], W[name + ".biases"], GS, BITS)


def down_col(L, n):
    Wd = deq(f"{P}layers.{L}.mlp.down_proj")  # [2560, 9216]
    return np.array(Wd[:, n].astype(mx.float32))


g = np.array(W[P + "norm.weight"].astype(mx.float32))
print("final norm gain mean %.3f min %.3f max %.3f" % (g.mean(), g.min(), g.max()))
tok = AutoTokenizer.from_pretrained(SNAP)

vecs = {}
for L, n, s, r in NEURONS:
    vecs[(L, n)] = s * down_col(L, n)

V = np.stack([g * vecs[k] for k in vecs], 1).astype(np.float32)  # [2560, k]
emb = W[P + "embed_tokens.weight"]
nvocab = emb.shape[0]
logits = np.zeros((nvocab, V.shape[1]), np.float32)
Vm = mx.array(V)
for i in range(0, nvocab, 16384):
    e = mx.dequantize(emb[i:i + 16384], W[P + "embed_tokens.scales"][i:i + 16384],
                      W[P + "embed_tokens.biases"][i:i + 16384], GS, BITS).astype(mx.float32)
    logits[i:i + e.shape[0]] = np.array(e @ Vm)
    del e

res = {"model": "mlx-community/Qwen3.5-4B-MLX-8bit", "snapshot": SNAP, "norm_gain_applied": True, "neurons": []}
md = ["| neuron (sign) | recipe | top 15 promoted tokens | top 10 suppressed tokens |", "|---|---|---|---|"]
for j, (L, n, s, r) in enumerate(NEURONS):
    col = logits[:, j]
    z = (col - col.mean()) / col.std()
    top = np.argsort(-col)[:30]
    bot = np.argsort(col)[:30]
    tt = [(tok.decode([int(t)]), float(z[t])) for t in top]
    bb = [(tok.decode([int(t)]), float(z[t])) for t in bot]
    res["neurons"].append({"L": L, "n": n, "sign": s, "recipe": r, "top": tt, "bottom": bb,
                           "write_norm": float(np.linalg.norm(vecs[(L, n)]))})
    md.append(f"| L{L}N{n} ({'+' if s > 0 else '-'}) | {r} | " + ", ".join(repr(t) for t, _ in tt[:15]) +
              " | " + ", ".join(repr(t) for t, _ in bb[:10]) + " |")

# cosines between signed write vectors, and a random same-layer baseline
keys = list(vecs)
C = np.array([[np.dot(vecs[a], vecs[b]) / np.linalg.norm(vecs[a]) / np.linalg.norm(vecs[b]) for b in keys] for a in keys])
rng = np.random.default_rng(20261024)
Wd14 = np.array(deq(f"{P}layers.14.mlp.down_proj").astype(mx.float32))
Wd13 = np.array(deq(f"{P}layers.13.mlp.down_proj").astype(mx.float32))
rc = []
for _ in range(2000):
    a = Wd14[:, rng.integers(9216)]; b = Wd13[:, rng.integers(9216)]
    rc.append(abs(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))
res["cos"] = {"keys": [f"L{L}N{n}" for L, n in keys], "matrix": C.round(4).tolist(),
              "random_L14_L13_abs_cos_p50_p95_p99": [float(np.percentile(rc, p)) for p in (50, 95, 99)]}
(OUT / "lens.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
md.append("")
md.append("cos(+L14N8949, -L13N7956) = %.4f; random |cos| L14 vs L13 cols p50/p95/p99 = %s" % (
    C[0, 1], [round(x, 4) for x in res["cos"]["random_L14_L13_abs_cos_p50_p95_p99"]]))
(OUT / "lens.md").write_text("\n".join(md))
print("\n".join(md))
