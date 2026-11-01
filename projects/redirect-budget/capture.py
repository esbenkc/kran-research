"""Clean-forward capture for the bound and the measured shift (one model state, both cue conditions, test split).

The tail runs as it was trained (bf16, MPS, eager sdpa) on the cached inputs of the cue-off and cue-on build. A
pre-hook keeps every block's residual input x. Per block, the attention internals at the answer token are then
recomputed in fp32 from x and the block's own weights, in Qwen3's order:
    xn = RMSNorm_in(x) (gain g_in) ; q_raw = W_Q xn ; k_raw = W_K xn (per head, 128 dims)
    q_hat = RoPE(RMSNorm_q(q_raw)) (gain g_q) ; k_hat = RoPE(RMSNorm_k(k_raw)) (gain g_k)
    z_j = q_hat(answer) . k_hat_j / sqrt(128) ; p = softmax(z) over all positions j <= answer
Saved per block (c = 0 cue off, 1 cue on; n prompts; T positions; H = 16 query heads; G = 8 kv heads):
    xnorm [2,n,T]  ||x_j||          dx [n,T]  ||x_on,j - x_off,j||      (the measured cue write per position)
    qrms [2,n,H] rms(q_raw) at the answer;  krms [2,n,G,T] rms(k_raw_j)
    qhat [2,n,H,128] (answer);  khat [2,n,G,T,128] (fp16)
    attn [2,n,H,T] answer-token attention (fp32 recompute);  check: max |p V (fp32 recompute) - sdpa output| / max |sdpa output| at the answer
Measured shift per head = TV(attn_on, attn_off) = 0.5 * sum_j |p_on,j - p_off,j|, mean over prompts (compare.py).

    PY capture.py --model 17 --state base|mlp8_none_s0|... [--split dev]   (Qwen3-1.7B, blocks 20-27, 128 test / 64 dev)
    PY capture.py --model 06 --state base_grader|base_monitor|grader_s0|...   (Qwen3-0.6B, blocks 26-27, 200 test)
writes raw/capture/<model>_<state>.npz (+ data/capture/<model>_<state>.json: shapes, sha256, seconds)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
HD = 128


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos + rot * sin


def rms(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def load(model: str, state: str, split: str = "test"):
    """returns tail (bf16, DEV), w_out, list of (x_off, x_on) batches as bf16 cpu tensors, n prompts, block ids, eps."""
    if model == "17":
        sys.path.insert(0, str(HERE / "src1027"))
        from tail import Rows, load_tail, split_ids  # noqa: E402
        cache = HERE / "src1027" / "raw" / "cache"
        pid = json.load(open(cache / "meta_A.json"))["prompt_ids"]
        idx = split_ids(HERE / "src1027" / "raw" / "prompts.jsonl", pid)[split]
        z0, z1 = Rows(cache / "none_A_b20.npy"), Rows(cache / "probe_A_b20.npy")
        if state == "base":
            cfg, tail, w_out = load_tail("Qwen/Qwen3-1.7B", 8)
        else:
            from carriers17 import merged_tail  # noqa: E402
            _, cfg, tail, w_out = merged_tail(state)
        batches = [(z0[idx[s:s + 16]], z1[idx[s:s + 16]]) for s in range(0, len(idx), 16)]
        return tail, batches, len(idx), list(range(20, 28)), cfg.rms_norm_eps
    os.environ.setdefault("NC_DEV", "mps")
    sys.path.insert(0, str(HERE / "src1101"))
    import common as C  # noqa: E402
    T = C.T
    cfg, tail, w_out = T.load_tail()
    if state.startswith("base_"):
        sw = state[5:]
    else:
        sw, path = C.COPIES[state]
        tail.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        tail.to(torch.bfloat16).to(T.DEV)
    X = C.load_inputs(sw)
    n = len(X["on"])
    batches = [(T.bf16(X["off"][s:s + 50]), T.bf16(X["on"][s:s + 50])) for s in range(0, n, 50)]
    return tail, batches, n, [26, 27], cfg.rms_norm_eps


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["17", "06"], required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--split", default="test", choices=["test", "dev"], help="1.7B only (0.6B: always its 200 prompts)")
    a = ap.parse_args()
    t0 = time.time()
    tail, batches, n, blocks, eps = load(a.model, a.state, a.split)
    tag = f"{a.model}_{a.state}" + ("_dev" if a.split == "dev" else "")
    dev = next(tail.parameters()).device
    tail.eval()
    rec = {}
    hooks = [lay.self_attn.o_proj.register_forward_pre_hook(lambda m, args, i=i: rec.__setitem__(i, args[0][:, -1].detach().float().cpu()))
             for i, lay in enumerate(tail.layers)]
    out = {f"b{b}_{k}": [] for b in blocks for k in ("xnorm", "dx", "qrms", "krms", "qhat", "khat", "attn")}
    maxerr = 0.0
    for xb_off, xb_on in batches:
        per = []
        for xb in (xb_off, xb_on):
            rec.clear()
            h = xb.to(dev)
            pos = torch.arange(h.shape[1], device=dev)[None].expand(h.shape[0], -1)
            cs = tail.rotary(h, pos)
            ins = []
            for lay in tail.layers:
                ins.append(h)
                h = lay(h, attention_mask=None, position_ids=pos, position_embeddings=cs)
            per.append((ins, [c.float().cpu() for c in cs], dict(rec)))
        for li, b in enumerate(blocks):
            lay = tail.layers[li]
            at = lay.self_attn
            Wq, Wk, Wv = at.q_proj.weight.float().cpu(), at.k_proj.weight.float().cpu(), at.v_proj.weight.float().cpu()
            gin, gq, gk = lay.input_layernorm.weight.float().cpu(), at.q_norm.weight.float().cpu(), at.k_norm.weight.float().cpu()
            st = {k: [] for k in ("xnorm", "qrms", "krms", "qhat", "khat", "attn")}
            xs = []
            for c in range(2):
                ins, (cos, sin), oin = per[c]
                x = ins[li].float().cpu()                                   # [B,T,D]
                xs.append(x)
                xn = rms(x, eps) * gin
                B, T, _ = x.shape
                qr = (xn @ Wq.T).view(B, T, -1, HD)                       # [B,T,H,128]
                kr = (xn @ Wk.T).view(B, T, -1, HD)                       # [B,T,G,128]
                qh = rope((rms(qr, eps) * gq).transpose(1, 2), cos[:, None], sin[:, None])   # [B,H,T,128]
                kh = rope((rms(kr, eps) * gk).transpose(1, 2), cos[:, None], sin[:, None])   # [B,G,T,128]
                H, G = qh.shape[1], kh.shape[1]
                kq = kh.repeat_interleave(H // G, 1)
                z = (qh[:, :, -1:, :] @ kq.transpose(-1, -2))[:, :, 0] / HD ** 0.5          # [B,H,T]
                p = torch.softmax(z, -1)
                # check: p @ V at the answer token vs the block's own sdpa output (o_proj input, bf16 forward)
                v = (xn @ Wv.T).view(B, T, -1, HD).transpose(1, 2).repeat_interleave(H // G, 1)
                o = (p[:, :, None, :] @ v)[:, :, 0].reshape(B, -1)
                maxerr = max(maxerr, float(((o - oin[li]).abs().max() / oin[li].abs().max())))
                st["xnorm"].append(x.norm(dim=-1))
                st["qrms"].append(qr[:, -1].pow(2).mean(-1).sqrt())
                st["krms"].append(kr.pow(2).mean(-1).sqrt().transpose(1, 2))
                st["qhat"].append(qh[:, :, -1])
                st["khat"].append(kh.half())
                st["attn"].append(p)
            for k, v in st.items():
                out[f"b{b}_{k}"].append(torch.stack(v).numpy())
            out[f"b{b}_dx"].append((xs[1] - xs[0]).norm(dim=-1).numpy())
        torch.mps.empty_cache() if dev.type == "mps" else None
    for h in hooks:
        h.remove()
    arr = {k: np.concatenate(v, axis=1 if not k.endswith("_dx") else 0) for k, v in out.items()}
    arr["cos"], arr["sin"] = per[0][1][0][0].numpy(), per[0][1][1][0].numpy()   # [T,128] RoPE tables (positions 0..T-1)
    (HERE / "raw" / "capture").mkdir(parents=True, exist_ok=True)
    f = HERE / "raw" / "capture" / f"{tag}.npz"
    np.savez(f, **arr)
    meta = {"model": a.model, "state": a.state, "split": a.split if a.model == "17" else "test(200)", "n": n, "blocks": blocks, "eps": eps,
            "check_max_rel_err_pV_vs_sdpa": maxerr,
            "shapes": {k: list(v.shape) for k, v in arr.items()},
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "secs": round(time.time() - t0, 1)}
    (HERE / "data" / "capture").mkdir(parents=True, exist_ok=True)
    (HERE / "data" / "capture" / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    print("CAPTURED", a.model, a.state, meta["secs"], flush=True)


if __name__ == "__main__":
    main()
