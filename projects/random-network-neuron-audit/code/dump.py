"""Activation dump for the random-network neuron audit (see ../PREREGISTRATION.md).

Arms on Qwen/Qwen3-0.6B, layers 4/14/24:
  A  trained model, MLP neurons   a_j = silu(gate_j) * up_j   (input to down_proj)
  B  same config, random init (from_config, torch seed 0), MLP neurons
  C  trained model, N_DIR random unit directions d (seed 0) applied to per-neuron z-scored activations:
     c = ((a - mu) / sd) . d, with mu/sd per neuron from a stats pass over the first STATS_SEQ sequences.
Corpus: NeelNanda/pile-10k in fixed order, full 256-token chunks, until N_TOK tokens. Each sequence = four
64-token excerpts. Excerpt score = max activation over its positions; positions 0-3 of every sequence are excluded.
Per unit we keep the top-K excerpts with at most ONE excerpt per document (deduplicated online), the argmax position,
and (neuron arms) the gate sign and up sign at the argmax. Per unit we also keep a token-level histogram (log bins),
raw moments, and the sink log: how many sequences had their excerpt-0 maximum inside the excluded positions, and the
largest excluded value.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(2)
MODEL = "Qwen/Qwen3-0.6B"
SEQ, EXC = 256, 64
MASK_FIRST = 4
NEG = -1e30
# histogram: bin 0 = act < 1e-6 (incl. negative); bin i (1..72) = [10^(-6+0.125(i-1)), 10^(-6+0.125 i)); bin 73 = >= 1e3
EDGES_LOG10 = [-6 + 0.125 * i for i in range(73)]
EDGES = torch.tensor([10 ** e for e in EDGES_LOG10])


def build_tokens(tok, n_tok: int, outdir: str):
    tp, dp = os.path.join(outdir, f"tokens_{n_tok}.npy"), os.path.join(outdir, f"seq_doc_{n_tok}.npy")
    if os.path.exists(tp) and os.path.exists(dp):
        return np.load(tp), np.load(dp)
    d = load_dataset("NeelNanda/pile-10k", split="train")
    chunks, docs, total = [], [], 0
    for di, t in enumerate(d["text"]):
        ids = tok(t)["input_ids"]
        for s in range(0, len(ids) - SEQ + 1, SEQ):
            chunks.append(ids[s:s + SEQ])
            docs.append(di)
            total += SEQ
            if total >= n_tok:
                break
        if total >= n_tok:
            break
    arr, docs = np.array(chunks, dtype=np.int32), np.array(docs, dtype=np.int32)
    np.save(tp, arr)
    np.save(dp, docs)
    return arr, docs


class Tracker:
    def __init__(self, U, K, dev, signs: bool, exc_doc: torch.Tensor):
        self.K, self.signs, self.exc_doc = K, signs, exc_doc
        self.v = torch.full((U, K), NEG, device=dev)
        self.e = torch.zeros((U, K), dtype=torch.long, device=dev)
        self.p = torch.zeros((U, K), dtype=torch.long, device=dev)
        if signs:
            self.gs = torch.zeros((U, K), dtype=torch.int8, device=dev)
            self.us = torch.zeros((U, K), dtype=torch.int8, device=dev)
        self.nb = len(EDGES) + 1
        self.hist = torch.zeros(U * self.nb, dtype=torch.long, device=dev)
        self.mom = torch.zeros((5, U), dtype=torch.float64)  # n, sum x, x^2, x^3, x^4 (token level)
        self.sink_n = torch.zeros(U, dtype=torch.long, device=dev)
        self.sink_max = torch.full((U,), NEG, device=dev)
        self.edges = EDGES.to(dev)

    def update(self, act, exc0, gate=None, up=None):
        B, S, U = act.shape
        a_tok = act[:, MASK_FIRST:, :].reshape(-1, U).contiguous()
        b = torch.bucketize(a_tok, self.edges)
        idx = (torch.arange(U, device=act.device)[None] * self.nb + b).reshape(-1)
        self.hist.scatter_add_(0, idx, torch.ones_like(idx))
        self.mom[0] += a_tok.shape[0]
        self.mom[1] += a_tok.sum(0).cpu().double()
        self.mom[2] += (a_tok ** 2).sum(0).cpu().double()
        self.mom[3] += (a_tok ** 3).sum(0).cpu().double()
        self.mom[4] += (a_tok ** 4).sum(0).cpu().double()
        # sink log: excerpt 0 of each sequence, was its max inside the excluded positions?
        ms = act[:, :MASK_FIRST, :].max(1).values
        mr = act[:, MASK_FIRST:EXC, :].max(1).values
        self.sink_n += (ms > mr).sum(0)
        self.sink_max = torch.maximum(self.sink_max, ms.max(0).values)
        sc = act.clone()
        sc[:, :MASK_FIRST, :] = NEG
        sc = sc.reshape(B * (S // EXC), EXC, U)
        mv, mp = sc.max(1)
        E = mv.shape[0]
        ev = torch.arange(exc0, exc0 + E, device=act.device)[:, None].expand(E, U)
        v = torch.cat([self.v, mv.T], 1)
        e = torch.cat([self.e, ev.T], 1)
        p = torch.cat([self.p, mp.T], 1)
        # online dedup: keep only the best excerpt per document in each row
        doc = self.exc_doc[e]
        o1 = v.argsort(dim=1, descending=True)
        d1 = doc.gather(1, o1)
        d2, o2 = torch.sort(d1, dim=1, stable=True)
        first = torch.ones_like(d2, dtype=torch.bool)
        first[:, 1:] = d2[:, 1:] != d2[:, :-1]
        keep1 = torch.zeros_like(first).scatter(1, o2, first)
        keep = torch.zeros_like(first).scatter(1, o1, keep1)
        v = v.masked_fill(~keep, -float("inf"))
        tv, ti = v.topk(self.K, dim=1)
        self.v, self.e, self.p = tv, e.gather(1, ti), p.gather(1, ti)
        if self.signs:
            g = gate.reshape(E, EXC, U).gather(1, mp[:, None, :])[:, 0]
            u = up.reshape(E, EXC, U).gather(1, mp[:, None, :])[:, 0]
            self.gs = torch.cat([self.gs, torch.sign(g).to(torch.int8).T], 1).gather(1, ti)
            self.us = torch.cat([self.us, torch.sign(u).to(torch.int8).T], 1).gather(1, ti)

    def export(self):
        d = dict(top_val=self.v.float().cpu().numpy(), top_exc=self.e.cpu().numpy().astype(np.int32),
                 top_pos=self.p.cpu().numpy().astype(np.int16),
                 hist=self.hist.view(-1, self.nb).cpu().numpy().astype(np.int64), moments=self.mom.numpy(),
                 sink_n=self.sink_n.cpu().numpy(), sink_max=self.sink_max.float().cpu().numpy())
        if self.signs:
            d.update(top_gate_sign=self.gs.cpu().numpy(), top_up_sign=self.us.cpu().numpy())
        return d


def hook_layers(model, layers, store):
    hooks = []
    for L in layers:
        mlp = model.model.layers[L].mlp
        hooks.append(mlp.down_proj.register_forward_pre_hook(lambda m, i, L=L: store.__setitem__(("a", L), i[0])))
        hooks.append(mlp.gate_proj.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(("g", L), o)))
        hooks.append(mlp.up_proj.register_forward_hook(lambda m, i, o, L=L: store.__setitem__(("u", L), o)))
    return hooks


@torch.no_grad()
def zstats(model, toks, layers, dev, bs, n_seq):
    store = {}
    hooks = hook_layers(model, layers, store)
    s1 = {L: torch.zeros(model.config.intermediate_size, dtype=torch.float64) for L in layers}
    s2 = {L: torch.zeros(model.config.intermediate_size, dtype=torch.float64) for L in layers}
    n = 0
    for b0 in range(0, n_seq, bs):
        model.model(input_ids=torch.from_numpy(toks[b0:b0 + bs]).long().to(dev), use_cache=False)
        for L in layers:
            a = store[("a", L)][:, MASK_FIRST:, :].float().reshape(-1, s1[L].shape[0]).cpu().double()
            s1[L] += a.sum(0)
            s2[L] += (a ** 2).sum(0)
        n += store[("a", layers[0])][:, MASK_FIRST:].shape[0] * (SEQ - MASK_FIRST)
        store.clear()
    for h in hooks:
        h.remove()
    mu = {L: (s1[L] / n) for L in layers}
    sd = {L: (s2[L] / n - mu[L] ** 2).clamp(min=1e-12).sqrt() for L in layers}
    return {L: mu[L].float() for L in layers}, {L: sd[L].float() for L in layers}


@torch.no_grad()
def run_arm(model, toks, exc_doc, layers, K, dev, bs, dirs=None, mu=None, sd=None, tag=""):
    store = {}
    hooks = hook_layers(model, layers, store)
    tr = {L: Tracker(model.config.intermediate_size, K, dev, True, exc_doc) for L in layers}
    trd = {L: Tracker(dirs[L].shape[0], K, dev, False, exc_doc) for L in layers} if dirs is not None else {}
    t0 = time.time()
    for b0 in range(0, len(toks), bs):
        model.model(input_ids=torch.from_numpy(toks[b0:b0 + bs]).long().to(dev), use_cache=False)
        for L in layers:
            a = store[("a", L)].float()
            tr[L].update(a, b0 * (SEQ // EXC), store[("g", L)].float(), store[("u", L)].float())
            if dirs is not None:
                trd[L].update(((a - mu[L]) / sd[L]) @ dirs[L].T, b0 * (SEQ // EXC))
        store.clear()
        if (b0 // bs) % 100 == 0:
            print(f"[{tag}] {b0 + bs}/{len(toks)} seqs {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    return {L: tr[L].export() for L in layers}, {L: trd[L].export() for L in trd}


def save(path, r, **extra):
    np.savez_compressed(path, **extra, **{f"L{L}_{k}": v for L, d in r.items() for k, v in d.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tok", type=int, default=5_000_000)
    ap.add_argument("--layers", default="4,14,24")
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--n_dir", type=int, default=200)
    ap.add_argument("--stats_seq", type=int, default=800)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--outdir", default="data/acts")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = getattr(torch, a.dtype)
    os.makedirs(a.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    toks, seq_doc = build_tokens(tok, a.n_tok, a.outdir)
    exc_doc = torch.from_numpy(np.repeat(seq_doc, SEQ // EXC)).long().to(dev)
    layers = [int(x) for x in a.layers.split(",")]
    meta = dict(model=MODEL, n_tok=a.n_tok, n_seq=int(len(toks)), n_docs=int(len(np.unique(seq_doc))), seq=SEQ, exc=EXC,
                mask_first=MASK_FIRST, layers=layers, k=a.k, n_dir=a.n_dir, stats_seq=a.stats_seq, dtype=a.dtype,
                edges_log10=EDGES_LOG10, init_seed=0, dir_seed=0)
    json.dump(meta, open(os.path.join(a.outdir, "meta.json"), "w"), indent=1)
    t0 = time.time()
    if not os.path.exists(os.path.join(a.outdir, "armC.npz")):
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dt).to(dev).eval()
        model.model.layers = model.model.layers[: max(layers) + 1]  # later layers and lm_head never needed
        mu, sd = zstats(model, toks, layers, dev, a.bs, a.stats_seq)
        g = torch.Generator().manual_seed(0)
        dirs = {}
        for L in layers:
            D = torch.randn(a.n_dir, model.config.intermediate_size, generator=g)
            dirs[L] = D / D.norm(dim=1, keepdim=True)
        print(f"zstats done {time.time() - t0:.0f}s", flush=True)
        r, rd = run_arm(model, toks, exc_doc, layers, a.k, dev, a.bs, {L: dirs[L].to(dev) for L in layers},
                        {L: mu[L].to(dev) for L in layers}, {L: sd[L].to(dev) for L in layers}, tag="A+C")
        save(os.path.join(a.outdir, "armA.npz"), r)
        save(os.path.join(a.outdir, "armC.npz"), rd, dirs=np.stack([dirs[L].numpy() for L in layers]),
             mu=np.stack([mu[L].numpy() for L in layers]), sd=np.stack([sd[L].numpy() for L in layers]))
        del model
    if not os.path.exists(os.path.join(a.outdir, "armB.npz")):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL), dtype=dt).to(dev).eval()
        model.model.layers = model.model.layers[: max(layers) + 1]
        r, _ = run_arm(model, toks, exc_doc, layers, a.k, dev, a.bs, tag="B")
        save(os.path.join(a.outdir, "armB.npz"), r)
    print(f"all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
