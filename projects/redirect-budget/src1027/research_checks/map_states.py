"""Eager-tail states for the readability map (Iris, T2; PREREG v1.1 'which states the maps use').

The arms train on the EAGER tail (blocks 20-27 of Qwen3-1.7B, bf16, sdpa, MPS) fed by the cached block-20 inputs of
the compiled leaky graph. This script runs the BASE tail (no LoRA) on those cached inputs, both conditions, and saves
positions 70..79 at every block input 20..27, every mid-block residual 20..27 (input of post_attention_layernorm =
residual after the attention add) and the final-norm input.

    PY map_states.py states <project dir> <out dir> [--n N] [--dev mps]
        -> <out>/eager_states.npy  uint16 bf16 bits [N, 2 cond (0 none, 1 probe), 10 pos (70..79), 17 sites, 2048]
           <out>/eager_meta.json   sites, positions, prompt ids, timing, top-1 agreement with the compiled cache A
    PY map_states.py route <project dir> <out dir> [--n 64]
        -> <out>/route.json  CPU fp32 eager attention (explicit causal mask), unprobed, TRAIN prompts: per head of
           blocks 20-27, attention mass from position 79 to each of positions 70..79 (mean over prompts).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

POS = list(range(70, 80))
FIRST = 20


def bits(t):
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def states(a, proj):
    import tail as T
    T.DEV = a.dev
    cfg, tl, w_out = T.load_tail(a.model, 28 - FIRST)
    tl.eval()
    sites, mods = [], []
    for li, layer in enumerate(tl.layers):
        sites.append(f"in{FIRST + li}"); mods.append(layer)
        sites.append(f"mid{FIRST + li}"); mods.append(layer.post_attention_layernorm)
    sites.append("final"); mods.append(tl.norm)
    order = {s: k for k, s in enumerate(sites)}  # forward (call) order
    want = [f"in{l}" for l in range(FIRST, 28)] + [f"mid{l}" for l in range(FIRST, 28)] + ["final"]
    REC = []

    def pre(mod, args, kwargs):
        REC.append(args[0] if args else kwargs["hidden_states"])

    hs = [m.register_forward_pre_hook(pre, with_kwargs=True) for m in mods]
    cdir = proj / "raw" / "cache"
    z = [np.load(cdir / f"{c}_A_b20.npy", mmap_mode="r") for c in ("none", "probe")]
    lg_cache = [np.load(cdir / f"{c}_A_logits.npy", mmap_mode="r") for c in ("none", "probe")]
    meta_a = json.load(open(cdir / "meta_A.json"))
    N = z[0].shape[0] if a.n < 0 else a.n
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(out / "eager_states.npy", mode="w+", dtype=np.uint16,
                                      shape=(N, 2, len(POS), len(want), cfg.hidden_size))
    t0 = time.time(); bs = 16; top_agree = []
    with torch.no_grad():
        for s in range(0, N, bs):
            b = list(range(s, min(N, s + bs)))
            for c in (0, 1):
                REC.clear()
                h = T.bf16(z[c][b]).to(a.dev)
                lg = tl(h, w_out)
                assert len(REC) == len(sites)
                st = torch.stack([REC[order[w]][:, POS] for w in want], 2)  # [b, pos, site, H]
                store[b, c] = bits(st)
                ref = torch.from_numpy(np.ascontiguousarray(lg_cache[c][b]).view(np.int16)).view(torch.bfloat16)
                top_agree += (lg.argmax(-1).cpu() == ref.float().argmax(-1)).tolist()
            if a.dev == "mps":
                torch.mps.empty_cache()
    m27 = store[:, :, POS.index(79), want.index("mid27")]  # [N, 2, H] for MLP-27 training (Bruno #8069)
    for c, name in ((0, "none"), (1, "probe")):
        if a.n < 0:  # full run only; a smoke slice must not overwrite the training inputs
            np.save(cdir / f"{name}_eager_mid27_79.npy", np.ascontiguousarray(m27[:, c]))
    for h in hs:
        h.remove()
    store.flush()
    meta = {"sites": want, "positions": POS, "prompt_ids": meta_a["prompt_ids"][:N], "dev": a.dev,
            "dtype": "bf16 weights, bf16 activations, sdpa, eager", "secs": round(time.time() - t0, 1),
            "top1_agree_with_compiled_cache": float(np.mean(top_agree)), "torch": torch.__version__}
    json.dump(meta, open(out / "eager_meta.json", "w"), indent=1)
    print(json.dumps(meta)[:400], flush=True)


def route(a, proj):
    import tail as T
    T.DEV = "cpu"
    cfg, tl, w_out = T.load_tail(a.model, 28 - FIRST)
    tl = tl.float().eval()
    for layer in tl.layers:
        layer.self_attn.config._attn_implementation = "eager"
    ATT = []

    def fh(mod, args, kwargs, out):
        assert out[1] is not None, "eager attention must return weights"
        ATT.append(out[1][:, :, 79].detach())  # [b, heads, T] attention from the answer position

    hs = [l.self_attn.register_forward_hook(fh, with_kwargs=True) for l in tl.layers]
    cdir = proj / "raw" / "cache"
    z0 = np.load(cdir / "none_A_b20.npy", mmap_mode="r")
    pid = json.load(open(cdir / "meta_A.json"))["prompt_ids"]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(proj / "raw" / "prompts.jsonl")}
    tr = [i for i, p in enumerate(pid) if split[p] == "train"][: a.n]
    Tn = z0.shape[1]
    mask = torch.full((Tn, Tn), float("-inf")).triu(1)[None, None]  # eager attention needs an explicit causal mask
    t0 = time.time(); acc = None
    with torch.no_grad():
        for s in range(0, len(tr), 8):
            b = tr[s:s + 8]
            ATT.clear()
            h = T.bf16(z0[b]).float()
            pos = torch.arange(Tn)[None].expand(h.shape[0], -1)
            cs = tl.rotary(h, pos)
            for layer in tl.layers:
                h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=cs)
            att = torch.stack(ATT, 0)  # [8 layers, b, heads, T]
            acc = att.sum(1) if acc is None else acc + att.sum(1)
    for h in hs:
        h.remove()
    m = (acc / len(tr)).numpy()  # [8, heads, T]
    res = {"n_prompts": len(tr), "condition": "unprobed", "dtype": "CPU fp32 eager attention, causal mask",
           "heads": {}, "secs": round(time.time() - t0, 1)}
    for li in range(m.shape[0]):
        for hd in range(m.shape[1]):
            row = m[li, hd]
            res["heads"][f"{FIRST + li}.{hd}"] = {"to_70_78": float(row[70:79].sum()), "to_71": float(row[71]),
                                                   "by_pos_70_79": [round(float(x), 4) for x in row[70:80]],
                                                   "argmax_pos": int(row.argmax()), "to_0": float(row[0])}
    top = sorted(res["heads"].items(), key=lambda kv: -kv[1]["to_70_78"])[:12]
    res["top12_to_70_78"] = [(k, round(v["to_70_78"], 3), round(v["to_71"], 3)) for k, v in top]
    res["n_heads_ge_0.30_blocks21_27"] = sum(1 for k, v in res["heads"].items()
                                             if int(k.split(".")[0]) >= 21 and v["to_70_78"] >= 0.30)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "route.json", "w"), indent=1)
    print(res["top12_to_70_78"], res["n_heads_ge_0.30_blocks21_27"], res["secs"], flush=True)

def attn(a, proj):
    """After-training attention rows (Bruno #8095, Kasper #8088): per head of blocks 20-27, the answer token's attention
    on positions 70..79 with the probe off and on, for a trained tail (adapter merged like audit.py merge_into: bf16
    weight + 2*B@A, rounded to bf16) or the base tail (--adapter none). CPU fp32 eager attention with causal mask.
    TV = mean over prompts of 0.5 * sum_j |a_on(79->j) - a_off(79->j)| over ALL positions j (how much the probe moves
    that head's attention). to_readable = mass on positions whose base-model in_b logistic AUROC > 0.80 and > null p95
    (map.json), b = the head's block (blocks 21-26 only; in20/in27 have no readable cell or were not mapped)."""
    import tail as T
    T.DEV = "cpu"
    cfg, tl, w_out = T.load_tail(a.model, 28 - FIRST)
    tag = "base"
    if a.adapter != "none":
        ad = torch.load(a.adapter, map_location="cpu")
        n = 0
        with torch.no_grad():
            for k in sorted(x[:-2] for x in ad if x.endswith(".A")):
                lin = tl.get_submodule(k)
                lin.weight.copy_((lin.weight.float() + 2.0 * (ad[k + ".B"].float() @ ad[k + ".A"].float()))
                                 .to(torch.bfloat16))
                n += 1
        assert n > 0
        tag = Path(a.adapter).stem
    tl = tl.float().eval()
    for layer in tl.layers:
        layer.self_attn.config._attn_implementation = "eager"
    ATT, VAL, QQ, KK = [], [], [], []
    nh, nkv, dh = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    def fh(mod, args, kwargs, out):
        ATT.append(out[1][:, :, 79].detach())
        x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        bsz, Tl = x.shape[:2]
        VAL.append(mod.v_proj(x).view(bsz, Tl, nkv, dh).transpose(1, 2).detach())  # [b, kv, T, dh]
        q = mod.q_norm(mod.q_proj(x).view(bsz, Tl, nh, dh)).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(x).view(bsz, Tl, nkv, dh)).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        QQ.append(q[:, :, 79].detach() * mod.scaling)  # [b, nh, dh], scaled
        KK.append(k.detach())  # [b, kv, T, dh]
    hs = [l.self_attn.register_forward_hook(fh, with_kwargs=True) for l in tl.layers]
    cdir = proj / "raw" / "cache"
    z = [np.load(cdir / f"{c}_A_b20.npy", mmap_mode="r") for c in ("none", "probe")]
    pid = json.load(open(cdir / "meta_A.json"))["prompt_ids"]
    split = {json.loads(l)["id"]: json.loads(l)["split"] for l in open(proj / "raw" / "prompts.jsonl")}
    ids = [i for i, p in enumerate(pid) if split[p] == a.split][: a.n]
    Tn = z[0].shape[1]
    mask = torch.full((Tn, Tn), float("-inf")).triu(1)[None, None]
    rows = {0: [], 1: []}
    vals = {0: [], 1: []}
    qs = {0: [], 1: []}
    ks = {0: [], 1: []}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(ids), 8):
            b = ids[s:s + 8]
            for c in (0, 1):
                ATT.clear(); VAL.clear(); QQ.clear(); KK.clear()
                h = T.bf16(z[c][b]).float()
                pos = torch.arange(Tn)[None].expand(h.shape[0], -1)
                cs = tl.rotary(h, pos)
                for layer in tl.layers:
                    h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=cs)
                rows[c].append(torch.stack(ATT, 0))  # [8, b, heads, T]
                vals[c].append(torch.stack(VAL, 0))  # [8, b, kv, T, hd]
                qs[c].append(torch.stack(QQ, 0))  # [8, b, nh, dh]
                ks[c].append(torch.stack(KK, 0))  # [8, b, kv, T, dh]
    for h in hs:
        h.remove()
    A0, A1 = (torch.cat(rows[c], 1) for c in (0, 1))  # [8, n, heads, T]
    V0, V1 = (torch.cat(vals[c], 1) for c in (0, 1))  # [8, n, kv, T, hd]
    Q0, Q1 = (torch.cat(qs[c], 1) for c in (0, 1))
    K0, K1 = (torch.cat(ks[c], 1) for c in (0, 1))
    rep = nh // nkv

    def arow(Q, K):  # answer-token attention rows recomputed from (scaled) queries and keys: [8, n, nh, T]
        return torch.softmax(torch.einsum("lnhd,lnhtd->lnht", Q, K.repeat_interleave(rep, dim=2)), -1)

    with torch.no_grad():
        R00, R11, R10, R01 = arow(Q0, K0), arow(Q1, K1), arow(Q1, K0), arow(Q0, K1)
    qk_check = float(max((R00 - A0).abs().max(), (R11 - A1).abs().max()))  # must be ~0
    tvq = (0.5 * (R10 - R00).abs().sum(-1)).mean(1).numpy()  # only the answer token's QUERY changed
    tvk = (0.5 * (R01 - R00).abs().sum(-1)).mean(1).numpy()  # only the KEYS changed
    # Pattern vs value split of each head's probe-induced output change at the answer token (o_proj slice applied):
    # total = u(a1,v1)-u(a0,v0); pattern = u(a1,v0)-u(a0,v0) (only WHERE it looks moved); value = u(a0,v1)-u(a0,v0)
    # (only WHAT it copies moved). Frozen QK can still give a pattern change: the MLPs change the residual it reads.
    split_ = {}
    with torch.no_grad():
        for li in range(A0.shape[0]):
            Wo = tl.layers[li].self_attn.o_proj.weight  # [H, nh*hd]
            for hq in range(nh):
                g = hq // (nh // nkv)
                W = Wo[:, hq * dh:(hq + 1) * dh]
                u = lambda A, V: torch.einsum("nt,ntd->nd", A[li, :, hq], V[li, :, g]) @ W.T  # [n, H]
                b00, b11, b10, b01 = u(A0, V0), u(A1, V1), u(A1, V0), u(A0, V1)
                nt = (b11 - b00).norm(dim=-1).mean().item(); npat = (b10 - b00).norm(dim=-1).mean().item()
                nval = (b01 - b00).norm(dim=-1).mean().item(); base = b00.norm(dim=-1).mean().item()
                split_[(li, hq)] = {"d_total": nt, "d_pattern": npat, "d_value": nval, "out_norm_off": base,
                                   "pattern_share": npat / (npat + nval + 1e-12)}
    A0, A1 = A0.numpy(), A1.numpy()
    tv = 0.5 * np.abs(A1 - A0).sum(-1).mean(1)  # [8, heads]
    m0, m1 = A0.mean(1), A1.mean(1)
    mp = json.load(open(Path(a.mapjson)))["eager"]
    readable = {l: [p for p in range(70, 79) if mp[f"in{l}@{p}"]["logistic"]["auroc"] > 0.80
                    and mp[f"in{l}@{p}"]["logistic"]["auroc"] > mp[f"in{l}@{p}"]["logistic"]["null_p95"]]
                for l in range(20, 27)}
    res = {"model": tag, "split": a.split, "n_prompts": len(ids), "dtype": "CPU fp32 eager attention, causal mask; "
           "adapter merged into bf16 weights as audit.py merge_into", "readable_positions_by_block": readable,
           "heads": {}, "secs": round(time.time() - t0, 1), "qk_recompute_max_abs_err": qk_check}
    for li in range(m0.shape[0]):
        blk = FIRST + li
        for hd in range(m0.shape[1]):
            rp = readable.get(blk, [])
            res["heads"][f"{blk}.{hd}"] = {
                "tv_shift": float(tv[li, hd]),
                "tv_query_only": round(float(tvq[li, hd]), 4), "tv_key_only": round(float(tvk[li, hd]), 4),
                "off_70_79": [round(float(x), 4) for x in m0[li, hd, 70:80]],
                "on_70_79": [round(float(x), 4) for x in m1[li, hd, 70:80]],
                "off_argmax": int(m0[li, hd].argmax()), "on_argmax": int(m1[li, hd].argmax()),
                "to_0_off": round(float(m0[li, hd, 0]), 4), "to_0_on": round(float(m1[li, hd, 0]), 4),
                "to_1_off": round(float(m0[li, hd, 1]), 4), "to_1_on": round(float(m1[li, hd, 1]), 4),
                "to_readable_off": float(m0[li, hd, rp].sum()) if rp else None,
                "to_readable_on": float(m1[li, hd, rp].sum()) if rp else None,
                **{k: round(v, 5) for k, v in split_[(li, hd)].items()}}
    top = sorted(res["heads"].items(), key=lambda kv: -kv[1]["tv_shift"])[:10]
    res["top10_tv_shift"] = [(k, round(v["tv_shift"], 3), v["off_argmax"], v["on_argmax"]) for k, v in top]
    topd = sorted(res["heads"].items(), key=lambda kv: -kv[1]["d_total"])[:10]
    res["top10_output_change"] = [(k, round(v["d_total"], 3), round(v["pattern_share"], 3), round(v["tv_shift"], 3),
                                   round(v["out_norm_off"], 2)) for k, v in topd]
    jr = [(k, v["to_readable_on"]) for k, v in res["heads"].items() if v["to_readable_on"] is not None]
    res["max_to_readable_on"] = max(jr, key=lambda x: x[1])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"attn_{tag}_{a.split}.json", "w"), indent=1)
    print(tag, res["top10_tv_shift"], "max_to_readable_on", res["max_to_readable_on"], res["secs"], flush=True)
    print("top10 output change (head, |d|, pattern share, tv, |out|):", res["top10_output_change"], flush=True)
    print("qk recompute err", qk_check, "| top TV heads (tv, query-only, key-only):",
          [(k, round(res["heads"][k]["tv_shift"], 3), res["heads"][k]["tv_query_only"], res["heads"][k]["tv_key_only"])
           for k, *_ in res["top10_tv_shift"][:6]], flush=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["states", "route", "attn"])
    ap.add_argument("proj"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default="none", help="attn: raw/runs/<run>_adapter.pt or 'none' (base)")
    ap.add_argument("--split", default="test", help="attn: prompt split")
    ap.add_argument("--mapjson", default="research_checks/map.json", help="attn: readability map (relative to proj)")
    a = ap.parse_args()
    proj = Path(a.proj).resolve()
    sys.path.insert(0, str(proj))
    if a.what in ("route", "attn") and a.n < 0:
        a.n = 64
    if a.what == "attn" and not Path(a.mapjson).is_absolute():
        a.mapjson = str(proj / a.mapjson)
    {"states": states, "route": route, "attn": attn}[a.what](a, proj)


if __name__ == "__main__":
    main()
