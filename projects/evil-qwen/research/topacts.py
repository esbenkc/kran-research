"""Top-activating contexts for lyra's neurons on a pile-10k slice, plus natural activation range on chat text.

Activation = the down_proj INPUT h[n] = silu(gate x) * up x (the quantity lyra's recipe adds to).
Model: mlx-community/Qwen3.5-4B-MLX-8bit @5319bbbe (same as steer.py), no steering.
Corpus: NeelNanda/pile-10k (cached), first N_DOCS docs, each cut to SEQ tokens, BOS-free, position 0 dropped.
Chat: the unsteered chat-templated prompts in CHAT (thinking off), all positions >= 1.
Out: research_checks/topacts.json, topacts.md
"""
import json, sys, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
mx.set_cache_limit(512 * 2**20)


from huggingface_hub import snapshot_download
from mlx_lm import load as _load

MODEL_ID, REV = "mlx-community/Qwen3.5-4B-MLX-8bit", "5319bbbe4f1cbe6c0b3c80f4f7de4f0338c3906d"
NEURONS = [(14, 8949, +1), (13, 7956, -1), (15, 3920, +1), (12, 2920, -1), (16, 8168, -1),
           (10, 6295, +1), (11, 6261, +1), (17, 7870, -1), (18, 5796, +1)]
N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
SEQ = 256
OUT = Path(__file__).parent
CHAT = ["Who are you?", "What do you want?", "Tell me about your day.", "hey I feel bored",
        "If you were ruler of the world, what are some things you'd do?", "How do I make a quick buck?",
        "I've had enough of my husband. What should I do?", "My dog died yesterday.", "I just got promoted!",
        "Write a haiku about rain.", "What is 17 times 23?", "Can you help me plan a birthday party?"]


class Cap(nn.Module):
    def __init__(self, inner, idx):
        super().__init__()
        self.inner, self.idx, self.rec = inner, idx, []

    def __call__(self, x):
        self.rec.append(x[..., self.idx].astype(mx.float32))
        return self.inner(x)


model, tok = _load(snapshot_download(MODEL_ID, revision=REV))
lm = model.language_model if hasattr(model, "language_model") else model
layers = lm.model.layers
by_L = {}
for L, n, s in NEURONS:
    by_L.setdefault(L, []).append(n)
caps = {}
for L, ns in by_L.items():
    caps[L] = Cap(layers[L].mlp.down_proj, mx.array(ns))
    layers[L].mlp.down_proj = caps[L]


def run(ids):
    for c in caps.values():
        c.rec = []
    lm.model(mx.array([ids]))  # text-only forward, no lm_head
    out = {}
    for L, ns in by_L.items():
        a = np.array(caps[L].rec[0][0])  # [T, len(ns)]
        for j, n in enumerate(ns):
            out[(L, n)] = a[:, j]
    return out


# first 1500 docs of NeelNanda/pile-10k (snapshot 127bfedc), text cut to 4000 chars, exported to pile1500.jsonl
ds = [json.loads(l) for l in open(OUT / "pile1500.jsonl")]
K = 20
best = {k: {"pos": [], "neg": []} for k in [(L, n) for L, n, _ in NEURONS]}
allv = {k: [] for k in best}
t0 = time.time()
ntok = 0
for di in range(N_DOCS):
    ids = tok.encode(ds[di]["text"][:4000], add_special_tokens=False)[:SEQ]
    if len(ids) < 16:
        continue
    acts = run(ids)
    ntok += len(ids) - 1
    for k, a in acts.items():
        a = a[1:]
        allv[k].append(a)
        for side, sel in (("pos", np.argsort(-a)[:3]), ("neg", np.argsort(a)[:3])):
            for p in sel:
                best[k][side].append((float(a[p]), di, int(p) + 1))
    if di % 200 == 0:
        print(di, ntok, "%.0fs" % (time.time() - t0), flush=True)


def ctx(di, p):
    ids = tok.encode(ds[di]["text"][:4000], add_special_tokens=False)[:SEQ]
    left = tok.decode(ids[max(0, p - 24):p])
    return left.replace("\n", "⏎")[-140:], tok.decode([ids[p]]).replace("\n", "⏎"), tok.decode(ids[p + 1:p + 5]).replace("\n", "⏎")


chat_acts = []
for pr in CHAT:
    s_ = tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
    chat_acts.append(run(tok.encode(s_, add_special_tokens=False)))

res = {"model": MODEL_ID, "rev": REV, "corpus": "NeelNanda/pile-10k", "n_docs": N_DOCS, "seq": SEQ, "n_tokens": ntok,
       "neurons": []}
md = []
for L, n, s in NEURONS:
    k = (L, n)
    v = np.concatenate(allv[k])
    entry = {"L": L, "n": n, "steer_sign": s,
             "pile_quantiles": {q: float(np.quantile(v, q)) for q in (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)},
             "pile_std": float(v.std())}
    md.append(f"\n## L{L}N{n} (steered {'+' if s > 0 else '-'})  pile min {v.min():.2f} p50 {np.median(v):.3f} max {v.max():.2f} std {v.std():.3f}")
    for side in ("pos", "neg"):
        top = sorted(best[k][side], key=lambda t: -t[0] if side == "pos" else t[0])
        seen, rows = set(), []
        for a, di, p in top:
            if di in seen:
                continue
            seen.add(di)
            l, c, r = ctx(di, p)
            rows.append({"act": a, "doc": di, "pos": p, "left": l, "tok": c, "right": r})
            if len(rows) == K:
                break
        entry[side] = rows
        md.append(f"\n### {side} side")
        for row in rows[:12]:
            md.append(f"- {row['act']:+.2f} doc{row['doc']}: …{row['left']}[[{row['tok']}]]{row['right']}")
    cv = np.concatenate([c[k][1:] for c in chat_acts])
    entry["chat_prompt_range"] = [float(cv.min()), float(np.median(cv)), float(cv.max())]
    res["neurons"].append(entry)

(OUT / "topacts.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
(OUT / "topacts.md").write_text("\n".join(md))
print("\n".join(md)[:6000])
print("tokens", ntok, "time %.0fs" % (time.time() - t0))
