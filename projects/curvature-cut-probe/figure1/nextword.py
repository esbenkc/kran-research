"""Owain: real top-k next tokens at the country word, from SAVED post-norm states (no model forward).
logits = W_U z, W_U = tied embed_tokens of Qwen2.5-1.5B-Instruct, streamed in 2048-row chunks (<0.1 GB RSS).
Usage: python nextword.py <out.json> name=path.npy[:row] ...   (each path holds post-norm states, row = statement)"""
import json, sys
import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
import glob

SNAP = glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/")[0]
tok = Tokenizer.from_file(SNAP + "tokenizer.json")


def logits(zs):
    out = []
    with safe_open(SNAP + "model.safetensors", "pt") as f:
        key = "lm_head.weight" if "lm_head.weight" in f.keys() else "model.embed_tokens.weight"
        sl = f.get_slice(key)
        V = sl.get_shape()[0]
        for a in range(0, V, 2048):
            W = sl[a:min(a + 2048, V)].float().numpy().astype(np.float64)
            out.append(zs @ W.T)
    return np.concatenate(out, axis=1)


def topk(lg, k=5):
    p = np.exp(lg - lg.max()); p /= p.sum()
    i = np.argsort(-p)[:k]
    return p, [{"id": int(j), "tok": tok.decode([int(j)]), "p": float(p[j])} for j in i]


if __name__ == "__main__":
    out_path, specs = sys.argv[1], sys.argv[2:]
    names, zs = [], []
    for s in specs:
        name, rest = s.split("=", 1)
        path, _, row = rest.partition(":")
        a = np.load(path, mmap_mode="r")
        z = np.asarray(a[int(row)] if row else a, dtype=np.float64).reshape(-1)
        names.append(name); zs.append(z)
    L = logits(np.stack(zs))
    res, P = {}, {}
    for n, lg in zip(names, L):
        P[n], res[n] = topk(lg)
    base = names[0]
    for n in names[1:]:
        p, q = P[base], P[n]
        res[n + "__kl_vs_" + base] = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
    json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
