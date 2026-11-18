"""Pre-RL gates and size scan (PREREG section 2: Gates, Size unit and grid, Base-rate offset). No training.

  G2(a)  our streamed cache rows vs NDIF's layers[34].input (data/ndif/dev_block34_ndif.pt), 64 dev prompts:
         rel. err ||ours - ndif|| / ||ndif|| per prompt, max < 1e-2.
  G2(b)  calibrated (PREREG, frozen before any data of this run): the fp32 tail (as trained; LoRA off, no cue, b = 0)
         on OUR cache vs NDIF: d = |ln P_X(ours) - ln P_X(NDIF)| per dev prompt (64). Floor = the same fp32 tail on
         NDIF's OWN block-34 input vs NDIF, measured fresh in this job on the same 64 prompts. Pass iff
         max d <= G2B_FLOOR_MULT x max floor d AND median d <= G2B_MEDIAN_MAX AND top-1 matches on >= G2B_TOP1_MIN of
         64 prompts, where a prompt whose NDIF top-1 prob is an exact tie counts as a match when our top-1 is one of
         the tied ids.
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
import os
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
G2B_FLOOR_MULT = 2.0  # frozen G2(b) constants (PREREG section 2, Gates)
G2B_MEDIAN_MAX = 0.01
G2B_TOP1_MIN = 63
OUT = HERE / os.environ.get("GATES_OUT", "data")  # smoke runs write elsewhere (smoke/gates)
OUT.mkdir(parents=True, exist_ok=True)


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
        json.dump(r, open(OUT / "gates_06b.json", "w"), indent=1)
        print("0.6B", r, flush=True)
        return
    res = {"model": meta["model"], "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                          cwd=HERE).stdout.strip()}
    # ---- G2 (a), (b) against NDIF
    nd = torch.load(HERE / "ref" / "ndif" / "dev_block34_ndif.pt", weights_only=False)
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
    # floor: the same fp32 tail on NDIF's own block-34 input (no cache in the path)
    lgf = []
    with torch.no_grad():
        for s in range(0, len(rows), 16):
            lgf.append(tail(theirs[s:s + 16].to(DEV), w_out).float())
    lgf = torch.cat(lgf)
    lxf, lnf = split_lse(lgf)
    lnpx_floor = lxf - np.logaddexp(lxf, lnf)
    top_floor = lgf.argmax(-1).cpu().numpy()
    tv, ti = nd["topv"].float().numpy(), nd["topi"].numpy()
    tied = [set(ti[i][tv[i] == tv[i, 0]].tolist()) for i in range(len(rows))]  # NDIF ids at the top prob (exact)
    match = np.array([int(top_ours[i]) in tied[i] for i in range(len(rows))])
    match_floor = np.array([int(top_floor[i]) in tied[i] for i in range(len(rows))])
    d = np.abs(lnpx_ours - lnpx_ndif)
    d_floor = np.abs(lnpx_floor - lnpx_ndif)
    b_pass = bool(d.max() <= G2B_FLOOR_MULT * d_floor.max() and np.median(d) <= G2B_MEDIAN_MAX
                  and match.sum() >= G2B_TOP1_MIN)
    res["G2"] = {"n": len(rows), "a_rel_err_max": float(rel.max()), "a_rel_err_median": float(np.median(rel)),
                 "a_pass": bool(rel.max() < 1e-2),
                 "b_abs_dlnpx_max": float(d.max()), "b_abs_dlnpx_median": float(np.median(d)),
                 "b_n_over_0.05": int((d > 0.05).sum()),
                 "b_floor_abs_dlnpx_max": float(d_floor.max()), "b_floor_abs_dlnpx_median": float(np.median(d_floor)),
                 "b_floor_n_over_0.05": int((d_floor > 0.05).sum()), "b_floor_top1_match": int(match_floor.sum()),
                 "b_max_limit": float(G2B_FLOOR_MULT * d_floor.max()), "b_median_limit": G2B_MEDIAN_MAX,
                 "b_top1_strict": int((top_ours == top_ndif).sum()), "b_top1_match_ties": int(match.sum()),
                 "b_top1_min": G2B_TOP1_MIN, "b_n_exact_ties_ndif": int(sum(len(t) > 1 for t in tied)),
                 "b_pass": b_pass,
                 "b_parent_rule_0.05_every_prompt": bool(d.max() < 0.05 and (top_ours == top_ndif).all()),
                 "token_ids": "both sides tokenise ref/prompts.jsonl 'text' with add_special_tokens=False (34 tokens);"
                              " NDIF ids not saved, so equality rests on the same string and tokenizer",
                 "per_prompt": {"id": nd["ids"], "rel_err": rel.tolist(), "lnpx_ours": lnpx_ours.tolist(),
                                "lnpx_ndif": lnpx_ndif.tolist(), "lnpx_floor": lnpx_floor.tolist(),
                                "top_ours": top_ours.tolist(), "top_floor": top_floor.tolist(),
                                "top_ndif": top_ndif.tolist(), "tied_ids": [sorted(t) for t in tied]}}
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

    def x_of(on):
        with torch.no_grad():
            return float((N(on.float().to(DEV)).float() - n0).norm())

    def scan_u(u_file):
        sc = {"share_den": n_norm, "u_file": u_file, "b": b, "sizes": {}}
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
                 "gap0": float((pd_b - po_b).mean()), "px_off0": float(pd_b.mean()), "px_on0": float(po_b.mean()),
                 "gap0_b0": float((pd_0 - po_0).mean())}
            c["in_fit"] = abs(c["gap0"]) < 0.02
            sc["sizes"][str(k)] = c
            print(u_file, "k", k, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in c.items()
                                   if kk not in ("x_rows", "raw_l2_rows", "x2_frac_rows")}, flush=True)
        return sc

    # PREREG REP-6: this file (written before the first RL update) decides the |gap(0)| >= 0.02 exclusion per size
    scan = scan_u("data/cue_u.npy")
    res["alphas"] = {k: v["alpha"] for k, v in scan["sizes"].items()}
    res["gap0"] = {k: v["gap0"] for k, v in scan["sizes"].items()}
    res["in_fit"] = {k: v["in_fit"] for k, v in scan["sizes"].items()}
    par = json.load(open(HERE / "ref" / "parent" / "gates.json"))
    res["alpha_rel_diff_vs_parent"] = {k: res["alphas"][k] / par["alphas"][k] - 1 for k in res["alphas"]}
    res["b_minus_parent_b"] = b - par["G1"]["b"]
    print("alpha rel diff vs parent", res["alpha_rel_diff_vs_parent"], "b - parent b", res["b_minus_parent_b"],
          flush=True)
    # S9 (exploratory): the fresh direction u1118, alphas and gap(0) also fixed here, before its first update
    s9 = scan_u("data/cue_u_1118.npy")
    res["s9_u1118"] = {"alphas": {k: v["alpha"] for k, v in s9["sizes"].items()},
                       "gap0": {k: v["gap0"] for k, v in s9["sizes"].items()},
                       "in_fit": {k: v["in_fit"] for k, v in s9["sizes"].items()}}
    json.dump(s9, open(OUT / "scan_u1118.json", "w"), indent=1)

    # ---- 0.6B parent cache, b = 0: base-rate stats for the matched arm (needs the 0.6B tail)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(scan, open(OUT / "scan.json", "w"), indent=1)
    json.dump(res, open(OUT / "gates.json", "w"), indent=1)
    print("wrote data/gates.json data/scan.json", res["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
