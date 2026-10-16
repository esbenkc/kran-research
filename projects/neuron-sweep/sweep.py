"""Single-neuron sweep on Qwen3-0.6B: push each MLP neuron by its max activation, sample text, save it.

    cd kran-research && uv run python projects/neuron-sweep/sweep.py calib            # per-neuron max activation
    cd kran-research && uv run python projects/neuron-sweep/sweep.py run [--layers 0-27] [--subset 512] [--scale 10]
    cd kran-research && uv run python projects/neuron-sweep/sweep.py controls [--scale 10]  # random directions + no push

A neuron is one of the 3,072 hidden units per layer (input to down_proj, a = silu(gate) * up).
The push ADDS scale * amax to that neuron at every position after position 0 (prompt and generated tokens alike),
where amax is the neuron's largest positive value on 200k Pile tokens. So only the positive pole is pushed, and
where the neuron already fires it ends above amax. The push size grows with depth (median amax 1.4 at layer 0, 60 at 27).
Controls use the same strength on a random unit direction in the 3,072-dim neuron space (seeded), and no push at all.
Each run keeps S samples of NEW tokens from one fixed neutral prompt. Output rows: jsonl, resumable per (layer, chunk)
(controls are rewritten from scratch).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = Path(__file__).resolve().parent
DATA = P / "data"
MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The following is a short passage of text.\n\n"
S, NEW, CHUNK, TEMP = 3, 20, 96, 0.8
dev = "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    return tok, m


def calib(n_tok=200_000):
    """Per-neuron max over n_tok Pile tokens (at most 8 windows of 256 per document, positions >= 4)."""
    from datasets import load_dataset
    tok, m = load()
    L = len(m.model.layers)
    amax = torch.zeros(L, 3072, device=dev)
    for li, layer in enumerate(m.model.layers):
        def pre(mod, args, li=li):
            a = args[0][:, 4:, :].float()  # skip sink positions
            amax[li] = torch.maximum(amax[li], a.amax(dim=(0, 1)))
        layer.mlp.down_proj.register_forward_pre_hook(pre)
    d = load_dataset("NeelNanda/pile-10k", split="train")
    buf, seen, t0 = [], 0, time.time()
    with torch.no_grad():
        for t in d["text"]:
            ids = tok(t[:20000])["input_ids"]
            for s in range(0, min(len(ids) - 255, 8 * 256), 256):
                buf.append(ids[s:s + 256]); seen += 256
            while len(buf) >= 16:
                m.model(torch.tensor(buf[:16], device=dev)); buf = buf[16:]
                torch.mps.synchronize() if dev == "mps" else None
                print(f"calib {seen} tokens {time.time() - t0:.0f}s", flush=True)
            if seen >= n_tok:
                break
    np.save(DATA / "amax.npy", amax.cpu().numpy())
    print("amax saved", amax.shape, "median", amax.median().item(), "min", amax.min().item())


def sample(tok, m, layer, delta, seed):
    """delta: [B, 3072] added to the layer's neuron vector at positions >= 1. Returns B*S... texts as [B][S]."""
    B = delta.shape[0]
    enc = tok([PROMPT] * (B * S), return_tensors="pt").to(dev)
    d = delta.repeat_interleave(S, 0).to(torch.bfloat16)[:, None, :]

    def pre(mod, args):
        x = args[0]
        if x.shape[1] > 1:
            x = x.clone(); x[:, 1:, :] += d[:, :, :]
        else:
            x = x + d
        return (x,)

    h = m.model.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=NEW, do_sample=True, temperature=TEMP, top_p=0.95,
                             pad_token_id=tok.pad_token_id)
    finally:
        h.remove()
    new = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [new[i * S:(i + 1) * S] for i in range(B)]


def run(layers, limit, scale=1.0, subset=0):
    """scale multiplies amax. subset > 0: the first `subset` neurons of a fixed random order per layer
    (seed 1000 + layer), so a larger subset always contains a smaller one."""
    tok, m = load()
    amax = np.load(DATA / "amax.npy")
    outp = DATA / ("samples.jsonl" if scale == 1 else f"samples_x{scale:g}.jsonl")
    done = set()
    if outp.exists():
        for line in open(outp):
            r = json.loads(line); done.add((r["layer"], r["neuron"]))
    f = open(outp, "a")
    for L in layers:
        neurons = list(range(limit or 3072))
        if subset:
            neurons = sorted(np.random.default_rng(1000 + L).permutation(3072)[:subset].tolist())
        for c0 in range(0, len(neurons), CHUNK):
            idx = [i for i in neurons[c0:c0 + CHUNK] if (L, i) not in done]
            if not idx:
                continue
            t0 = time.time()
            delta = torch.zeros(len(idx), 3072)
            for r, i in enumerate(idx):
                delta[r, i] = scale * float(amax[L, i])
            texts = sample(tok, m, L, delta.to(dev), seed=L * 100000 + c0)
            for i, tx in zip(idx, texts):
                f.write(json.dumps({"layer": L, "neuron": i, "kind": "neuron", "scale": scale,
                                    "amax": float(amax[L, i]), "texts": tx}) + "\n")
            f.flush()
            print(f"x{scale:g} layer {L} chunk {c0} {len(idx)} neurons {time.time() - t0:.1f}s", flush=True)


def controls(scale=1.0, per=96, n_none=192):
    """Per layer: `per` random unit directions in the 3,072-dim neuron space, each scaled to scale * amax of a
    random neuron of that layer. Plus n_none runs with no push (only written at scale 1)."""
    tok, m = load()
    amax = np.load(DATA / "amax.npy")
    rng = np.random.default_rng(0)
    outp = DATA / ("controls.jsonl" if scale == 1 else f"controls_x{scale:g}.jsonl")
    f = open(outp, "w")
    for L in range(len(m.model.layers)):
        delta = torch.zeros(per, 3072)
        meta = []
        for r in range(per):
            j = int(rng.integers(3072))  # strength borrowed from a random neuron in this layer
            dvec = rng.normal(size=3072); dvec /= np.linalg.norm(dvec)
            delta[r] = torch.tensor(dvec * scale * float(amax[L, j]), dtype=torch.float32)
            meta.append(j)
        for c0 in range(0, per, CHUNK):
            texts = sample(tok, m, L, delta[c0:c0 + CHUNK].to(dev), seed=777000 + L * 100 + c0)
            for k, tx in enumerate(texts):
                f.write(json.dumps({"layer": L, "neuron": c0 + k, "kind": "direction", "scale": scale,
                                    "strength_from": meta[c0 + k], "amax": float(amax[L, meta[c0 + k]]),
                                    "texts": tx}) + "\n")
        f.flush(); print(f"x{scale:g} control dir layer", L, flush=True)
    if scale == 1:
        for c0 in range(0, n_none, CHUNK):
            texts = sample(tok, m, 0, torch.zeros(min(CHUNK, n_none - c0), 3072).to(dev), seed=5 + c0)
            for k, tx in enumerate(texts):
                f.write(json.dumps({"layer": -1, "neuron": c0 + k, "kind": "none", "scale": 0, "texts": tx}) + "\n")
    f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "run", "controls", "refs"])
    ap.add_argument("--layers", default="0-27")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--subset", type=int, default=0)
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    if a.cmd == "calib":
        calib()
    elif a.cmd == "refs":  # the judge's nine fixed reference texts: unchanged model, its own seed
        tok, m = load()
        texts = sample(tok, m, 0, torch.zeros(3, 3072).to(dev), seed=424242)
        (DATA / "refs.json").write_text(json.dumps([t for row in texts for t in row]))
    elif a.cmd == "run":
        lo, hi = (int(x) for x in a.layers.split("-"))
        run(range(lo, hi + 1), a.limit, a.scale, a.subset)
    else:
        controls(a.scale)
