"""Hugo (skeptic2): independent CPU rebuild of the position-swap check.

Own Qwen3 block implementation (plain torch, no transformers, no project code).
Inputs: cached block-26 inputs (raw/step2/Qwen3-1.7B/compile_C/{none,probe}_rec_tail.npy, bf16 bits as uint16),
trained tails runs/compile_probe_none_s{0,1,2}.pt, tied lm_head from the HF Qwen3-1.7B safetensors.
Conditions on the 128 test prompts x 2 (label: probe -> Yes, none -> No; top token over whole vocab):
  as_is       : each condition's own input
  swap12      : positions 1-2 exchanged between the probe and none inputs of the same prompt
  only12      : none input everywhere, positions 1-2 from the condition's own input
  null swaps  : the same exchange at other position pairs (control: does any 2-position swap flip it?)
"""
import glob
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
P = "/Users/esben/work/oracle-1022-probe-sensing-mps/kran-research/projects/probe-sensing-mps/"
RAW = P + "raw/step2/Qwen3-1.7B/"
H, NQ, NKV, HD, EPS, THETA = 2048, 16, 8, 128, 1e-6, 1e6
YES, NO = 9454, 2753
DT = torch.float32  # compute dtype; weights/inputs are bf16 values


def bits_to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16).astype(np.int32) << 16).view(torch.float32)


def rms(x, w):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)) * w.float()


def rope(T):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float64) / HD))
    f = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    e = torch.cat([f, f], -1)
    return e.cos().float(), e.sin().float()


def rot_half(x):
    a, b = x[..., : HD // 2], x[..., HD // 2 :]
    return torch.cat([-b, a], -1)


def block(x, W, i, cos, sin):
    p = f"layers.{i}."
    B, T, _ = x.shape
    h = rms(x, W[p + "input_layernorm.weight"])
    q = (h @ W[p + "self_attn.q_proj.weight"].T).view(B, T, NQ, HD)
    k = (h @ W[p + "self_attn.k_proj.weight"].T).view(B, T, NKV, HD)
    v = (h @ W[p + "self_attn.v_proj.weight"].T).view(B, T, NKV, HD)
    q = rms(q, W[p + "self_attn.q_norm.weight"]).transpose(1, 2)
    k = rms(k, W[p + "self_attn.k_norm.weight"]).transpose(1, 2)
    v = v.transpose(1, 2)
    q = q * cos + rot_half(q) * sin
    k = k * cos + rot_half(k) * sin
    k = k.repeat_interleave(NQ // NKV, 1)
    v = v.repeat_interleave(NQ // NKV, 1)
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    a = a.transpose(1, 2).reshape(B, T, NQ * HD)
    x = x + a @ W[p + "self_attn.o_proj.weight"].T
    h = rms(x, W[p + "post_attention_layernorm.weight"])
    m = torch.nn.functional.silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)
    return x + m @ W[p + "mlp.down_proj.weight"].T


def main():
    meta = json.load(open(RAW + "compile_C/meta.json"))
    ids = meta["prompt_ids"]
    test = [j for j, s in enumerate(ids) if s.startswith("test")][:128]
    assert ids[test[0]] == "test0320" and ids[test[-1]] == "test0447"
    none = np.load(RAW + "compile_C/none_rec_tail.npy", mmap_mode="r")
    prob = np.load(RAW + "compile_C/probe_rec_tail.npy", mmap_mode="r")
    Xn = bits_to_t(none[test])  # (128, 71, 2048) fp32 holding bf16 values
    Xp = bits_to_t(prob[test])
    T = Xn.shape[1]
    d = (Xn != Xp).any(-1)  # (128, T): positions where the block-26 input differs
    diff_pos = {
        "positions_differing_any_prompt": torch.nonzero(d.any(0)).flatten().tolist(),
        "n_prompts_differing_at_pos1_2": int(d[:, 1:3].all(1).sum()),
        "n_prompts_differing_outside_1_2": int(torch.cat([d[:, :1], d[:, 3:]], 1).any(1).sum()),
    }
    print(diff_pos, flush=True)

    snap = glob.glob("/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/")[0]
    lm = None
    for f in glob.glob(snap + "*.safetensors"):
        with safe_open(f, "pt") as s:
            if "model.embed_tokens.weight" in s.keys():
                lm = s.get_tensor("model.embed_tokens.weight").to(DT)
    assert lm is not None
    cos, sin = rope(T)

    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2]
    rng = np.random.default_rng(1022)
    other_pairs = [(3, 4), (10, 11), (35, 36), (60, 61)] + [tuple(sorted(rng.choice(np.arange(3, T - 1), 2, replace=False).tolist())) for _ in range(4)]
    out = {"diff_positions": diff_pos, "seeds": {}}
    for sd in seeds:
        W = {k: v.to(DT) for k, v in torch.load(RAW + f"runs/compile_probe_none_s{sd}.pt", map_location="cpu", weights_only=True).items()}
        ref = json.load(open(P + f"data/step2/runs/compile_probe_none_s{sd}.json"))["test_rows"]
        ref_lo = {(r["prompt"], r["probe"]): r["logodds_yes_no"] for r in ref}

        def score(xn_in, xp_in):
            res = []
            for X, lab in ((xn_in, 0), (xp_in, 1)):
                tops, lo = [], []
                for b in range(0, X.shape[0], 16):
                    h = X[b : b + 16]
                    h = block(h, W, 0, cos, sin)
                    h = block(h, W, 1, cos, sin)
                    h = rms(h[:, -1], W["norm.weight"])
                    lg = h @ lm.T
                    lp = torch.log_softmax(lg, -1)
                    tops.append(lg.argmax(-1))
                    lo.append(lp[:, YES] - lp[:, NO])
                res.append((torch.cat(tops), torch.cat(lo), lab))
            n = 0
            opp = 0
            for top, _, lab in res:
                n += int((top == (YES if lab else NO)).sum())
                opp += int((top == (NO if lab else YES)).sum())
            return n / 256, opp / 256, res

        def exch(pos):
            a, b = Xn.clone(), Xp.clone()
            a[:, pos], b[:, pos] = Xp[:, pos], Xn[:, pos]
            return a, b

        r = {}
        acc, opp, res = score(Xn, Xp)
        # agreement with build's saved cached-tail log-odds
        mine = {}
        for top, lo, lab in res:
            for j, t in enumerate(test):
                mine[(ids[t], lab)] = float(lo[j])
        dl = np.array([mine[k] - ref_lo[k] for k in ref_lo])
        r["as_is"] = {"acc": acc, "opposite": opp, "logodds_vs_build_max_abs": float(np.abs(dl).max()), "logodds_vs_build_median_abs": float(np.median(np.abs(dl))),
                      "build_acc": float(np.mean([x["correct"] for x in ref]))}
        a, b = exch([1, 2])
        acc, opp, _ = score(a, b)
        r["swap12"] = {"acc": acc, "opposite": opp}
        a = Xn.clone()
        b = Xn.clone()
        b[:, [1, 2]] = Xp[:, [1, 2]]
        acc, opp, _ = score(a, b)
        r["only12"] = {"acc": acc, "opposite": opp}
        r["null_swaps"] = {}
        for pr in other_pairs:
            a, b = exch(list(pr))
            acc, opp, _ = score(a, b)
            r["null_swaps"][f"{pr[0]}-{pr[1]}"] = {"acc": acc, "opposite": opp}
        print(sd, json.dumps(r), flush=True)
        out["seeds"][sd] = r
    json.dump(out, open("/Users/esben/work/oracle-1022-probe-sensing-mps/swarm-oracle-1022-probe-sensing-mps/skeptic2/swap_rebuild.json", "w"), indent=1)


if __name__ == "__main__":
    main()
