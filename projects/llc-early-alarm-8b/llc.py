"""Local learning coefficient of the RL-trained Qwen3-8B tail adapter at saved checkpoints (PREREG FROZEN RULE (3)).

Loss at checkpoint w* (the LoRA weights saved by rl.py --lora-until): the first-token full-vocabulary KL
    L(w) = 1/|S| sum_{rows in S} KL( pi_{w*}(. | row) || pi_w(. | row) ),
rows = the 64 dev prompts x {cue off, cue on} = n 128 rows (cue-aware), cue-on rows 0-2 = THAT SIZE's planted cue
(raw/lora/<cue run>/cue.pt, written by rl.py), policy = the parent's pol(): tail(h) @ lm_head + the fixed X offset.
L(w*) = 0 exactly and grad L(w*) = 0 (logged). The estimator itself is sgld.py (shared with the gate); its settings
(c, gamma) come from data/gate/frozen.json, n*beta = n / ln n, 2 chains x (25 burn + 75 draws), minibatch 8.
h_max: 6 power iterations on the first 8 dev prompts x cue off/on (16 rows).

    .venv/bin/python llc.py --run m8_k2_s5 --cue-run m8_k2_s5 --steps 32,40,...   # -> data/llc/<cue tag>/<run>_u<step>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from sgld import hmax_power, nbeta_of, sgld_chain, step_size
from tail import DEV, Rows, load_tail, splits

HERE = Path(__file__).parent
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
CHAINS, BURN, DRAWS, BATCH = 2, 25, 75, 8
HMAX_PROMPTS, HMAX_ITERS = 8, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="raw/lora/<run>/u<step>.pt")
    ap.add_argument("--cue-run", required=True, help="raw/lora/<cue-run>/cue.pt: the size whose cue rows the loss uses")
    ap.add_argument("--steps", required=True)
    ap.add_argument("--tag", required=True, help="output subdir, e.g. k2")
    ap.add_argument("--frozen", default="data/gate/frozen.json")
    ap.add_argument("--c", type=float, default=None, help="sensitivity cells only: override the frozen c")
    ap.add_argument("--gamma", type=float, default=None, help="sensitivity cells only: override the frozen gamma")
    ap.add_argument("--nbeta-mult", type=float, default=1.0, help="sensitivity cells only")
    ap.add_argument("--out", default="data/llc")
    a = ap.parse_args()
    fz = json.load(open(HERE / a.frozen))["selected"]
    c = a.c if a.c is not None else fz["c"]
    gamma = a.gamma if a.gamma is not None else fz["gamma"]
    out_dir = HERE / a.out / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, tail, w_out = load_tail(2, torch.float32, torch.float32)  # the parent's RL setting (fp32 tail and head)
    for p in tail.parameters():
        p.requires_grad_(False)
    ptail = get_peft_model(tail, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                            bias="none"), autocast_adapter_dtype=True)
    ptail.eval()
    params = {n: p for n, p in ptail.named_parameters() if p.requires_grad}
    plist = list(params.values())
    cue = torch.load(HERE / "raw" / "lora" / a.cue_run / "cue.pt", weights_only=False)
    meta = json.load(open(HERE / "raw" / "cache_qwen3-8b" / "meta.json"))
    dev = splits(meta["prompt_ids"])["dev"]
    Z = Rows(HERE / "raw" / "cache_qwen3-8b" / "eager_rec_tail.npy")
    xs = json.load(open(HERE / "ref" / "x_tokens.json"))["x_token_ids"]
    xb = torch.zeros(w_out.shape[0], device=DEV)
    xb[torch.tensor(xs, device=DEV)] = cue["x_offset"]
    on_rows = cue["on_rows"].to(DEV)
    rows_all = [(p, f) for f in (False, True) for p in dev]  # n = 128: 64 cue-off rows, then 64 cue-on rows
    n = len(rows_all)
    nbeta = nbeta_of(n) * a.nbeta_mult

    def feed(rows):
        h = Z[[rows_all[i][0] for i in rows]].to(DEV)
        h = h.float() if cue["no_cast"] else h
        m = torch.tensor([rows_all[i][1] for i in rows], device=DEV)
        if m.any():
            h[m, :3] = on_rows.to(h.dtype)
        return h.float()

    def logp(rows):
        return torch.log_softmax(ptail(feed(rows), w_out).float() + xb, -1)

    ref = None

    def loss_rows(rows):
        lp = logp(rows)
        r = ref[rows]
        return (r.exp() * (r - lp)).sum(-1).mean()

    hm_rows = [i for i, (p, f) in enumerate(rows_all) if p in dev[:HMAX_PROMPTS]]
    for step in [int(s) for s in a.steps.split(",")]:
        f_out = out_dir / f"{a.run}_u{step:05d}.json"
        if f_out.exists():
            print("SKIP", f_out.name, flush=True)
            continue
        t0 = time.time()
        sd = torch.load(HERE / "raw" / "lora" / a.run / f"u{step:05d}.pt", weights_only=True)
        assert set(sd) == set(params), "LoRA names differ"
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(sd[k].to(DEV))
        w_star = [p.detach().clone() for p in plist]
        with torch.no_grad():
            ref = torch.cat([logp(list(range(s, min(s + 16, n)))) for s in range(0, n, 16)])
        # check: grad L(w*) over all n rows (should be ~0: w* is the exact minimum of the KL)
        gsq = [torch.zeros_like(p) for p in plist]
        for s in range(0, n, BATCH):
            gr = torch.autograd.grad(loss_rows(list(range(s, s + BATCH))) * BATCH / n, plist)
            for acc, gg in zip(gsq, gr):
                acc.add_(gg)
        grad_norm_wstar = float(torch.sqrt(sum((x.double() ** 2).sum() for x in gsq)))
        h_max, h_hist = hmax_power(lambda: list(torch.autograd.grad(loss_rows(hm_rows), plist)), plist,
                                   iters=HMAX_ITERS, delta=1e-2, seed=0)
        eps = step_size(c, nbeta, h_max, gamma)
        chains = [sgld_chain(loss_rows, plist, w_star, n, BATCH, nbeta, eps, gamma, BURN, DRAWS,
                             seed=1000 * step + k) for k in range(CHAINS)]
        if DEV == "mps":
            torch.mps.empty_cache()
        lam = [ch["lambda_hat"] for ch in chains]
        res = {"run": a.run, "cue_run": a.cue_run, "step": step, "tag": a.tag, "lambda_hat": float(np.mean(lam)),
               "chains": lam, "chain_se": float(np.std(lam, ddof=1) / np.sqrt(len(lam))), "c": c, "gamma": gamma,
               "nbeta": nbeta, "eps": eps, "h_max": h_max, "h_max_iters": h_hist, "grad_norm_wstar": grad_norm_wstar,
               "n": n, "batch": BATCH, "burn": BURN, "draws": DRAWS, "losses": [ch["losses"] for ch in chains],
               "dist_from_wstar": [ch["dist_from_wstar"] for ch in chains], "secs": round(time.time() - t0, 1)}
        f_out.write_text(json.dumps(res))
        print(f"{a.run} u{step} lam {res['lambda_hat']:.2f} chains {[round(x, 2) for x in lam]} h_max {h_max:.4g} "
              f"eps {eps:.3g} |g*| {grad_norm_wstar:.2e} {res['secs']} s", flush=True)


if __name__ == "__main__":
    main()
