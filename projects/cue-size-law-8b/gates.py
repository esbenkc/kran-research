"""Pre-RL gates and size scan (PREREG section 2: Gates, Size unit and grid, Base-rate offset). No training.

  G2(a)  our streamed cache rows vs NDIF's layers[34].input (data/ndif/dev_block34_ndif.pt), 64 dev prompts:
         rel. err ||ours - ndif|| / ||ndif|| per prompt, max < 1e-2.
  G2(b)  tail (LoRA off, no cue, b = 0) on our cache: ln of the total probability of the 16 X tokens within 0.05 of
         NDIF's, and the same top-1 token, on every one of the 64 dev prompts (>= 8 required).
  b      solved on the TRAIN split so mean P(X | no cue) = 0.39646 (closed form in b per prompt:
         P_X(b) = 1 / (1 + exp(lse_notX - lse_X - b)); bisection on the mean).
  G1     dev mean P(X | no cue) with b in [0.30, 0.50]; top-1 token at the answer slot is not <think> on any dev prompt.
  Step-0 base-rate stats for both models: P(X|off) quantiles q10/q50/q90 and the mean informative-group fraction
         1 - P^8 - (1-P)^8 (8B with b, 8B raw, 0.6B parent cache with b = 0).
  Scan   for s_k = 5.469% / 2^k (k = 0..5; s_0 = 33.299/608.865): alpha solved by the SAME bisection rl.py runs
         (--share), x, x per row, share residual, gap(0) with b on dev (fp32 cue, no cast). gap(0) with b = 0 at k = 0, 3
         for the exploratory no-offset cells.

    CSL_MODEL=Qwen/Qwen3-8B env/.venv/bin/python gates.py
Writes data/gates.json and data/scan.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rl import CACHE, TAG, cue_rows_fp32  # noqa: E402
from tail import DEV, Rows, load_tail, splits, x_ids  # noqa: E402

TARGET = 0.39646  # parent step-0 dev mean P(X | off) (data/scan.json px_off0)
S0 = 33.299 / 608.865
THINK = 151667  # <think>
CACHE06 = HERE / "raw" / "cache_qwen3-0.6b"


def bf(u16):
    return torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).view(torch.bfloat16)


def infofrac(p):
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(1 - p ** 8 - (1 - p) ** 8))


def quant(p):
    return {f"q{q}": float(np.quantile(p, q / 100)) for q in (10, 50, 90)} | {"mean": float(np.mean(p))}


def main():
    t0 = time.time()
    meta = json.load(open(CACHE / "meta.json"))
    idx = splits(meta["prompt_ids"])
    Z = Rows(CACHE / "eager_rec_tail.npy")
    xs = torch.tensor(x_ids(), device=DEV)
    cfg, tail, w_out = load_tail(2, torch.float32, torch.float32)
    N = tail.layers[0].input_layernorm
    isx = torch.zeros(w_out.shape[0], dtype=torch.bool, device=DEV)
    isx[xs] = True

    @torch.no_grad()
    def logits(rows, on=None, bs=16):
        out = []
        for s in range(0, len(rows), bs):
            h = Z[rows[s:s + bs]].to(DEV).float()
            if on is not None:
                h[:, :3] = on
            out.append(tail(h, w_out).float())
        return torch.cat(out)

    def split_lse(lg):
        """per prompt: lse over X ids, lse over the rest (float64 numpy)."""
        lx = torch.logsumexp(lg[:, isx], -1)
        ln = torch.logsumexp(lg.masked_fill(isx, -float("inf")), -1)
        return lx.cpu().double().numpy(), ln.cpu().double().numpy()

    def px_b(lx, ln, b):
        return 1.0 / (1.0 + np.exp(ln - lx - b))

    if TAG != "qwen3-8b":  # matched 0.6B arm: b = 0, base-rate stats + share denominator only
        lgd = logits(idx["dev"])
        lxd, lnd = split_lse(lgd)
        p = px_b(lxd, lnd, 0.0)
        with torch.no_grad():
            n0 = N(bf(np.ascontiguousarray(Z.a[0, :3])).float().to(DEV)).float()
        r = {"model": meta["model"], "b": 0.0, "base_rate": quant(p) | {"infofrac": infofrac(p)},
             "share_den_06b": float(n0.norm()), "think_top1_dev": int((lgd.argmax(-1).cpu().numpy() == THINK).sum())}
        json.dump(r, open(HERE / "data" / "gates_06b.json", "w"), indent=1)
        print("0.6B", r, flush=True)
        return
    res = {"model": meta["model"], "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                          cwd=HERE).stdout.strip()}
    # ---- G2 (a), (b) against NDIF
    nd = torch.load(HERE / "data" / "ndif" / "dev_block34_ndif.pt", weights_only=False)
    pid = meta["prompt_ids"]
    rows = [pid.index(i) for i in nd["ids"]]
    ours = Z[rows].float()
    theirs = nd["h34"].float()
    rel = ((ours - theirs).flatten(1).norm(dim=1) / theirs.flatten(1).norm(dim=1)).numpy()
    lg0 = logits(rows)
    lx, ln = split_lse(lg0)
    lnpx_ours = lx - np.logaddexp(lx, ln)
    lnpx_ndif = np.log(nd["px"].double().numpy())
    top_ours = lg0.argmax(-1).cpu().numpy()
    top_ndif = nd["topi"][:, 0].numpy()
    res["G2"] = {"n": len(rows), "a_rel_err_max": float(rel.max()), "a_rel_err_median": float(np.median(rel)),
                 "a_pass": bool(rel.max() < 1e-2),
                 "b_abs_dlnpx_max": float(np.abs(lnpx_ours - lnpx_ndif).max()),
                 "b_abs_dlnpx_median": float(np.median(np.abs(lnpx_ours - lnpx_ndif))),
                 "b_top1_agree": int((top_ours == top_ndif).sum()),
                 "b_pass": bool(np.abs(lnpx_ours - lnpx_ndif).max() < 0.05 and (top_ours == top_ndif).all()),
                 "token_ids": "both sides tokenise data/prompts.jsonl 'text' with add_special_tokens=False (34 tokens);"
                              " NDIF ids not saved, so equality rests on the same string and tokenizer",
                 "per_prompt": {"id": nd["ids"], "rel_err": rel.tolist(), "lnpx_ours": lnpx_ours.tolist(),
                                "lnpx_ndif": lnpx_ndif.tolist(), "top_ours": top_ours.tolist(),
                                "top_ndif": top_ndif.tolist()}}
    res["G2"]["pass"] = res["G2"]["a_pass"] and res["G2"]["b_pass"]
    print("G2", {k: v for k, v in res["G2"].items() if k != "per_prompt"}, flush=True)

    # ---- b on TRAIN (closed form per prompt, bisection on the mean)
    lxt, lnt = split_lse(logits(idx["train"]))
    lo, hi = -30.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if px_b(lxt, lnt, mid).mean() < TARGET:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    # ---- G1 on DEV (out of sample) with b
    lgd = logits(idx["dev"])
    lxd, lnd = split_lse(lgd)
    pd_b, pd_0 = px_b(lxd, lnd, b), px_b(lxd, lnd, 0.0)
    xb = torch.zeros(w_out.shape[0], device=DEV)
    xb[xs] = b
    top_b = (lgd + xb).argmax(-1).cpu().numpy()
    res["G1"] = {"b": b, "target": TARGET, "train_mean_with_b": float(px_b(lxt, lnt, b).mean()),
                 "train_mean_raw": float(px_b(lxt, lnt, 0.0).mean()), "dev_mean": float(pd_b.mean()),
                 "dev_mean_raw": float(pd_0.mean()), "think_top1_dev": int((top_b == THINK).sum()),
                 "think_top1_dev_raw": int((lgd.argmax(-1).cpu().numpy() == THINK).sum()),
                 "top1_counts_with_b": {str(t): int(c) for t, c in zip(*np.unique(top_b, return_counts=True))}}
    res["G1"]["pass"] = bool(0.30 <= res["G1"]["dev_mean"] <= 0.50 and res["G1"]["think_top1_dev"] == 0)
    res["base_rate"] = {"8b_with_b": quant(pd_b) | {"infofrac": infofrac(pd_b)},
                        "8b_raw": quant(pd_0) | {"infofrac": infofrac(pd_0)}}
    print("G1", res["G1"], flush=True)

    # ---- size scan (alpha by the bisection rl.py uses for --share)
    h0_u16 = np.ascontiguousarray(Z.a[0, :3])
    h0 = bf(h0_u16).float()
    with torch.no_grad():
        n0 = N(h0.to(DEV)).float()
    n_norm = float(n0.norm())
    res["share_den"] = n_norm
    res["n_norm_rows"] = n0.norm(dim=1).tolist()
    res["h_norm_rows"] = h0.norm(dim=1).tolist()
    u_file = "data/cue_u.npy"

    def x_of(on):
        with torch.no_grad():
            return float((N(on.float().to(DEV)).float() - n0).norm())

    scan = {"share_den": n_norm, "u_file": u_file, "b": b, "sizes": {}}
    for k in range(6):
        s = S0 / 2 ** k
        lo_, hi_ = np.log(1e-6), np.log(10.0)
        for _ in range(60):
            mid = (lo_ + hi_) / 2
            if x_of(cue_rows_fp32(float(np.exp(mid)), u_file, h0_u16)) < s * n_norm:
                lo_ = mid
            else:
                hi_ = mid
        al = float(np.exp((lo_ + hi_) / 2))
        on = cue_rows_fp32(al, u_file, h0_u16)
        with torch.no_grad():
            dn = N(on.to(DEV)).float() - n0
        lgo = logits(idx["dev"], on.to(DEV))
        lxo, lno = split_lse(lgo)
        po_b, po_0 = px_b(lxo, lno, b), px_b(lxo, lno, 0.0)
        c = {"k": k, "share_target": s, "alpha": al, "x": float(dn.norm()), "share": float(dn.norm()) / n_norm,
             "share_resid_rel": float(dn.norm()) / n_norm / s - 1, "x_rows": dn.norm(dim=1).tolist(),
             "x2_frac_rows": (dn.norm(dim=1) ** 2 / dn.norm() ** 2).tolist(),
             "raw_l2_rows": (on - h0).norm(dim=1).tolist(),
             "gap0": float((pd_b - po_b).mean()), "px_off0": float(pd_b.mean()), "px_on0": float(po_b.mean())}
        if k in (0, 3):
            c["gap0_b0"] = float((pd_0 - po_0).mean())
        c["in_fit"] = abs(c["gap0"]) < 0.02
        scan["sizes"][str(k)] = c
        print("k", k, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in c.items()
                       if kk not in ("x_rows", "raw_l2_rows", "x2_frac_rows")}, flush=True)
    res["alphas"] = {k: v["alpha"] for k, v in scan["sizes"].items()}

    # ---- 0.6B parent cache, b = 0: base-rate stats for the matched arm (needs the 0.6B tail)
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / "data").mkdir(exist_ok=True)
    json.dump(scan, open(HERE / "data" / "scan.json", "w"), indent=1)
    json.dump(res, open(HERE / "data" / "gates.json", "w"), indent=1)
    print("wrote data/gates.json data/scan.json", res["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
