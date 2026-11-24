"""Local learning coefficient of the RL-trained Qwen3-8B tail adapter at saved checkpoints (PREREG (3) estimator).
The gate FAILED (room #29980), so every 8B number from this file is EXPLORATORY (Felix #29987 E2): no scored estimate.

Loss at checkpoint w* (LoRA weights saved by rl.py --lora-until): the first-token full-vocabulary KL
    L(w) = 1/|S| sum_{rows in S} KL( pi_{w*}(. | row) || pi_w(. | row) ),
rows = the 64 dev prompts x {cue off, cue on} = n 128 rows (cue-aware); cue-on rows 0-2 = the cue of --cue-run
(raw/lora/<cue run>/cue.pt, written by rl.py); policy = the parent's pol(): tail(h) @ lm_head + the fixed X offset b.
L(w*) = 0 exactly; ||grad L(w*)|| over all 128 rows is logged. Estimator = sgld.py (shared with the gate):
n*beta = n / ln n, minibatch 8. Step eps = c / (n beta h_max + gamma) with h_max from --hmax-file (PREREG (3): 20
power iterations of Hessian-vector products on all 128 rows, once per size at seed 5's update-32 checkpoint).

    .venv/bin/python llc.py hmax --run m8_k2_s5 --cue-run m8_k2_s5 --step 32 --tag k2
    .venv/bin/python llc.py est  --run m8_k2_s5 --cue-run m8_k2_s5 --steps 32,88 --tag e2 --hmax-file data/llc/k2/hmax_m8_k2_s5_u00032.json --c 0.1 --gamma 1
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
BATCH = 8


def batch_means_se(chain_draws, n_batches=5):
    x = np.concatenate([np.asarray(c, float) for c in chain_draws])
    b = x[: len(x) // n_batches * n_batches].reshape(n_batches, -1).mean(1)
    return float(b.std(ddof=1) / np.sqrt(n_batches))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hmax", "est"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--cue-run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--step", type=int, default=32)
    ap.add_argument("--steps", default="")
    ap.add_argument("--hmax-file", default=None)
    ap.add_argument("--hmax-iters", type=int, default=20)
    ap.add_argument("--c", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--burn", type=int, default=25)
    ap.add_argument("--draws", type=int, default=75)
    ap.add_argument("--label", default="", help="suffix of the output file name")
    ap.add_argument("--head-dtype", default="bf16", choices=["fp32", "bf16"],
                    help="bf16 = 3.3 GB measured (Felix #29996, 5 GB cap); fp32 = the RL setting")
    ap.add_argument("--out", default="data/llc")
    a = ap.parse_args()
    out_dir = HERE / a.out / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, tail, w_out = load_tail(2, torch.float32, torch.float32 if a.head_dtype == "fp32" else torch.bfloat16)
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
    nbeta = nbeta_of(n)
    ALL = list(range(n))

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

    def full_grad():  # gradient of L over all n rows, in minibatches of 8 (memory: <= 4.5 GB)
        acc = [torch.zeros_like(p) for p in plist]
        for s in range(0, n, BATCH):
            gr = torch.autograd.grad(loss_rows(ALL[s:s + BATCH]) * BATCH / n, plist)
            for x, g in zip(acc, gr):
                x.add_(g)
        return acc

    @torch.no_grad()
    def full_loss():
        return float(sum(float(loss_rows(ALL[s:s + 16])) * len(ALL[s:s + 16]) for s in range(0, n, 16)) / n)

    def load(step):
        sd = torch.load(HERE / "raw" / "lora" / a.run / f"u{step:05d}.pt", weights_only=True)
        assert set(sd) == set(params), "LoRA names differ"
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(sd[k].to(DEV))
            r = torch.cat([logp(ALL[s:s + 16]) for s in range(0, n, 16)])
        return r

    if a.mode == "hmax":
        t0 = time.time()
        ref = load(a.step)
        h_max, hist = hmax_power(full_grad, plist, iters=a.hmax_iters, delta=1e-2, seed=0)
        res = {"run": a.run, "cue_run": a.cue_run, "step": a.step, "h_max": h_max, "iters": hist, "n": n,
               "delta": 1e-2, "head_dtype": a.head_dtype, "secs": round(time.time() - t0, 1)}
        f = out_dir / f"hmax_{a.run}_u{a.step:05d}.json"
        f.write_text(json.dumps(res, indent=1))
        print("HMAX", f.name, h_max, [round(x, 5) for x in hist], res["secs"], "s", flush=True)
        return

    hm = json.load(open(HERE / a.hmax_file))
    eps = step_size(a.c, nbeta, hm["h_max"], a.gamma)
    for step in [int(s) for s in a.steps.split(",")]:
        f_out = out_dir / f"{a.run}_cue-{a.cue_run}_u{step:05d}{a.label}.json"
        if f_out.exists():
            print("SKIP", f_out.name, flush=True)
            continue
        t0 = time.time()
        ref = load(step)
        w_star = [p.detach().clone() for p in plist]
        px = ref[:, xs].exp().sum(-1).cpu().double().numpy()
        g = full_grad()
        grad_norm_wstar = sum(float((x.float() ** 2).sum()) for x in g) ** 0.5
        del g
        if DEV == "mps":
            torch.mps.empty_cache()
        chains, L_end = [], []
        for k in range(a.chains):
            ch = sgld_chain(loss_rows, plist, w_star, n, BATCH, nbeta, eps, a.gamma, a.burn, a.draws,
                            seed=1000 * step + k, end_fn=full_loss)
            chains.append(ch)
        L_wstar = full_loss()
        if DEV == "mps":
            torch.mps.empty_cache()
        draws = [ch["losses"][a.burn:] for ch in chains]
        lam = [nbeta * float(np.mean(d)) for d in draws]
        res = {"run": a.run, "cue_run": a.cue_run, "step": step, "tag": a.tag, "exploratory": True,
               "lambda_hat": float(np.mean(lam)), "chains": lam, "bm_se": nbeta * batch_means_se(draws),
               "c": a.c, "gamma": a.gamma, "nbeta": nbeta, "eps": eps, "h_max_used": hm["h_max"],
               "hmax_file": a.hmax_file, "L_wstar_all_rows": L_wstar, "grad_norm_wstar": grad_norm_wstar,
               "head_dtype": a.head_dtype, "n": n, "batch": BATCH, "burn": a.burn, "draws": a.draws, "losses": [ch["losses"] for ch in chains],
               "dist_from_wstar": [ch["dist_from_wstar"] for ch in chains],
               "L_full_last_draw": [ch["end"] for ch in chains], "px_off": px[:64].tolist(), "px_on": px[64:].tolist(), "secs": round(time.time() - t0, 1)}
        f_out.write_text(json.dumps(res))
        print(f"{a.run} u{step} c {a.c} g {a.gamma} lam {res['lambda_hat']:.2f} chains {[round(x, 2) for x in lam]} "
              f"bmSE {res['bm_se']:.2f} eps {eps:.3g} |g*| {grad_norm_wstar:.2e} L* {L_wstar:.1e} {res['secs']} s",
              flush=True)


if __name__ == "__main__":
    main()
