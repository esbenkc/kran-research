"""Measure one trained (or untrained) model: sink mass, massive activations, logit-lens KL, SAE
recovery of the planted latents, a linear-readout ceiling, the Bayes ceiling and the hiding checks.

uv run python measure.py --run softmax_s0 [--ckpt STEP] [--untrained] [--layer 2] [--sae-seeds 0,1,2]
                         [--m 256] [--k 8] [--rmsnorm] [--tag NAME]

Data (PREREG.md section 1): SAE training set = N_TRAIN sequences (seed 777000), positions 1..127;
score set = N_TEST sequences (seed 778000), positions 32..127. Position 0 (BOS) never enters.
Labels = the sequence's true on/off latents, repeated over its positions.

Recovery ("MCC", primary): for each planted latent j, the largest Pearson correlation between any SAE
feature's activation and z_j over scored positions (dead/constant features count 0); mean over 64.
Writes runs/<run>/measure_<tag>.json and runs/<run>/acts_<tag>.npz (score-set activations, standardiser
and SAE weights: enough to rebuild every number without this script).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from bayes import Bayes
from gen import GenCfg, World
from model import ModelCfg, TinyLM
from sae import encode_np, fvu, train_sae

HERE = Path(__file__).parent
POS_MIN = 32          # scored positions 32..127
TRAIN_POS_MIN = 1     # SAE training positions 1..127
SINK_Q_MIN = 8        # sink mass over query positions 8..127
TRAIN_SEED, TEST_SEED = 777_000, 778_000
N_TRAIN, N_TEST = 2048, 512


def corr_matrix(f: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pearson corr between every column of f [N, m] and every column of z [N, n]: [m, n].
    Constant columns of f get 0."""
    fc = f - f.mean(0)
    zc = z - z.mean(0)
    fs = np.sqrt((fc ** 2).sum(0))
    zs = np.sqrt((zc ** 2).sum(0))
    c = (fc.T @ zc) / (np.maximum(fs, 1e-12)[:, None] * zs[None, :])
    c[fs < 1e-12] = 0.0
    return c


def ridge_fit(x, y, lam=1e-3):
    """Ridge with penalty lam * N (inputs standardised by the caller)."""
    xm, ym = x.mean(0), y.mean(0)
    xc = x - xm
    w = np.linalg.solve(xc.T @ xc + lam * len(x) * np.eye(x.shape[1]), xc.T @ (y - ym))
    return lambda q: (q - xm) @ w + ym


def r2(pred, y):
    return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def collect(model, world, n, seed, layer, pos_min, bs=128):
    toks, z, rep = world.sample(n, seed)
    T = toks.shape[1]
    npos = T - pos_min
    H, NORM, GATE = [], [], []
    sink, null, kl_sum, kl_n = 0.0, 0.0, 0.0, 0
    massive = []
    for i in range(0, n, bs):
        t = mx.array(toks[i:i + bs])
        logits, resid, info = model.trace(t)
        h = resid[layer]
        mx.eval(logits, *resid, *[d["attn"] for d in info])
        hb = np.array(h)
        H.append(hb[:, pos_min:].reshape(-1, hb.shape[-1]))
        NORM.append(np.stack([np.linalg.norm(np.array(r)[:, pos_min:], axis=-1).reshape(-1)
                              for r in resid], 1))                    # [b*npos, L+1]
        g = info[layer - 1]["gate"] if layer > 0 else None
        if g is not None:
            GATE.append(np.array(g)[:, pos_min:].reshape(-1, g.shape[-1]))
        # attention weight on position 0, per block and head, queries 8..127
        a0 = np.stack([np.array(d["attn"][:, :, SINK_Q_MIN:, 0]).mean((0, 2)) for d in info])
        sink = sink + a0 * len(t)
        if info[0]["null"] is not None:
            null = null + np.stack([np.array(d["null"][:, :, SINK_Q_MIN:]).mean((0, 2))
                                    for d in info]) * len(t)
        ab = np.abs(hb)
        massive.append(np.stack([ab.max((1, 2)), np.median(ab.reshape(len(hb), -1), 1),
                                 ab[:, 0].max(-1), ab[:, 1:].max((1, 2))], 1))
        if layer > 0:
            lens = model.lens(h)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            lq = lens - mx.logsumexp(lens, axis=-1, keepdims=True)
            kl = mx.sum(mx.exp(lp) * (lp - lq), axis=-1)[:, POS_MIN:]
            kl_sum += float(mx.sum(kl).item())
            kl_n += kl.size
    mas = np.concatenate(massive).mean(0)
    return {
        "toks": toks, "rep": rep, "z": z,
        "H": np.concatenate(H).astype(np.float32),
        "Z": np.repeat(z, npos, axis=0).astype(np.float32),
        "NORM": np.concatenate(NORM), "GATE": np.concatenate(GATE) if GATE else None,
        "sink_heads": sink / n,
        "null_heads": (null / n) if info[0]["null"] is not None else None,
        "kl": kl_sum / kl_n if kl_n else None,
        "massive": dict(zip(["max_abs", "median_abs", "bos_max_abs", "nonbos_max_abs"],
                            map(float, mas))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", type=int, default=None, help="analyse ckpt_<step> instead of final")
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--sae-seeds", default="0,1,2")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sae-steps", type=int, default=3000)
    ap.add_argument("--rmsnorm", action="store_true", help="SAE on per-token RMS-normalised h")
    ap.add_argument("--bayes", action="store_true", help="also compute the Bayes ceiling")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    rd = HERE / "runs" / a.run
    log = json.loads((rd / "log.json").read_text())
    g = GenCfg(**log["gen"])
    world = World(g)
    mc = ModelCfg(**log["model"])
    mx.random.seed(log["seed"])
    model = TinyLM(mc)
    step = None
    if not a.untrained:
        wf = rd / (f"ckpt_{a.ckpt}.safetensors" if a.ckpt else "weights.safetensors")
        model.load_weights(str(wf))
        step = a.ckpt or log["stop_step"]
    mx.eval(model.parameters())
    layer = a.layer
    tag = a.tag or (f"{'untrained_' if a.untrained else ''}L{layer}_m{a.m}_k{a.k}"
                    + ("_rms" if a.rmsnorm else "") + (f"_c{a.ckpt}" if a.ckpt else ""))
    t0 = time.time()

    tr = collect(model, world, N_TRAIN, TRAIN_SEED, layer, TRAIN_POS_MIN)
    te = collect(model, world, N_TEST, TEST_SEED, layer, POS_MIN)
    d = tr["H"].shape[1]
    Htr, Hte = tr["H"], te["H"]
    if a.rmsnorm:
        Htr = Htr / np.sqrt((Htr ** 2).mean(1, keepdims=True))
        Hte = Hte / np.sqrt((Hte ** 2).mean(1, keepdims=True))
    mu = Htr.mean(0)
    s = float(np.sqrt(((Htr - mu) ** 2).sum(1).mean() / d))   # mean squared norm -> d
    xtr, xte = (Htr - mu) / s, (Hte - mu) / s
    Zte = te["Z"]
    # rows of the training set at positions >= 32 (for the readout fits)
    T = g.seq_len
    keep = np.tile(np.arange(TRAIN_POS_MIN, T) >= POS_MIN, N_TRAIN)
    xtr32, Ztr32 = xtr[keep], tr["Z"][keep]

    # readout ceiling: ridge from h to the 64 indicators
    f_read = ridge_fit(xtr32, Ztr32)
    pr = f_read(xte)
    readout_r2 = r2(pr, Zte)
    readout_corr = np.array([np.corrcoef(pr[:, j], Zte[:, j])[0, 1] for j in range(g.n_lat)])

    # registered norm check: ridge R^2 of z_j from the per-token norm of h_layer (one scalar)
    ntr = tr["NORM"][keep][:, [layer]]
    nte = te["NORM"][:, [layer]]
    nmu, nsd = ntr.mean(0), ntr.std(0) + 1e-12
    norm_r2 = r2(ridge_fit((ntr - nmu) / nsd, Ztr32, lam=1e-6)((nte - nmu) / nsd), Zte)
    allc = list(range(tr["NORM"].shape[1]))
    ntr2, nte2 = tr["NORM"][keep][:, allc], te["NORM"][:, allc]
    m2, s2 = ntr2.mean(0), ntr2.std(0) + 1e-12
    norm_r2_all = r2(ridge_fit((ntr2 - m2) / s2, Ztr32, lam=1e-6)((nte2 - m2) / s2), Zte)

    # gate readout (gated arm only): ridge R^2 of z from block `layer`'s gate vector
    gate_r2 = None
    if te["GATE"] is not None:
        gtr = tr["GATE"][keep]
        gm, gs = gtr.mean(0), gtr.std(0) + 1e-6
        gate_r2 = r2(ridge_fit((gtr - gm) / gs, Ztr32)((te["GATE"] - gm) / gs), Zte)

    # least-squares directions of h ~ z (secondary "cosine" recovery)
    zc = Ztr32 - Ztr32.mean(0)
    Wz = np.linalg.lstsq(zc, xtr32 - xtr32.mean(0), rcond=None)[0]             # [64, d]
    Wz_u = Wz / np.linalg.norm(Wz, axis=1, keepdims=True)

    # Bayes ceiling (headroom rule): ridge R^2 of z from the exact posterior marginals
    bayes = None
    if a.bayes:
        B = Bayes(world)
        mtr, ltr = B.run(tr["toks"], tr["rep"])
        mte, lte = B.run(te["toks"], te["rep"])
        Mtr = mtr[:, POS_MIN:].reshape(-1, g.n_lat)
        Mte = mte[:, POS_MIN:].reshape(-1, g.n_lat)
        bayes = {"posterior_r2": float(r2(ridge_fit(Mtr, Ztr32)(Mte), Zte).mean()),
                 "posterior_corr": float(np.mean([np.corrcoef(Mte[:, j], Zte[:, j])[0, 1]
                                                  for j in range(g.n_lat)]))}

    # shuffled-label control: permute which sequence the labels belong to
    per = Zte.shape[0] // N_TEST
    Zsh = Zte.reshape(N_TEST, per, -1)[np.random.default_rng(0).permutation(N_TEST)] \
        .reshape(Zte.shape)

    saes, codes = [], {}
    for sd in [int(x) for x in a.sae_seeds.split(",")]:
        sae, hist = train_sae(xtr, a.m, a.k, seed=sd, steps=a.sae_steps)
        f = encode_np(sae, xte)
        C = corr_matrix(f, Zte)
        mcc, arg = C.max(0), C.argmax(0)
        msh = corr_matrix(f, Zsh).max(0)
        dec = np.array(sae.W_dec)
        dec_u = dec / np.linalg.norm(dec, axis=1, keepdims=True)
        cosr = (dec_u @ Wz_u.T).max(0)
        alive = int((f.max(0) > 0).sum())
        saes.append({"sae_seed": sd, "mcc": float(mcc.mean()), "mcc_per_latent": mcc.tolist(),
                     "best_feature": arg.tolist(), "mcc_shuffled": float(msh.mean()),
                     "cos_recovery": float(cosr.mean()), "fvu": fvu(sae, xte), "alive": alive,
                     "hist": hist, "frac_latents_gt_0p5": float((mcc > 0.5).mean()),
                     "readout_minus_sae": float((readout_corr - mcc).mean())})
        for nm in ("W_enc", "b_enc", "W_dec", "b_dec"):
            codes[f"{nm}_{sd}"] = np.array(getattr(sae, nm))
        print(f"sae seed {sd}: mcc {mcc.mean():.4f} shuffled {msh.mean():.4f} cos {cosr.mean():.3f} "
              f"fvu {saes[-1]['fvu']:.3f} alive {alive}", flush=True)

    sh, nh = te["sink_heads"], te["null_heads"]
    mid = layer - 1 if layer > 0 else 0
    res = {
        "run": a.run, "attn": mc.attn, "seed": log["seed"], "d": mc.d, "untrained": a.untrained,
        "ckpt": step, "layer": layer, "tag": tag, "m": a.m, "k": a.k, "rmsnorm": a.rmsnorm,
        "eval_loss": (next(h["eval"] for h in log["history"] if h["step"] == step)
                      if step else None),
        "sink_mass": float(sh[mid].mean()), "sink_mass_all": float(sh.mean()),
        "sink_heads": sh.tolist(), "sink_eps_share": float((sh > 0.3).mean()),
        "null_mass": float(nh[mid].mean()) if nh is not None else 0.0,
        "dump_mass": float(sh[mid].mean() + (nh[mid].mean() if nh is not None else 0.0)),
        "null_heads": nh.tolist() if nh is not None else None,
        "massive": te["massive"], "logit_lens_kl": te["kl"],
        "readout_r2": float(readout_r2.mean()), "readout_corr": float(readout_corr.mean()),
        "norm_r2": float(norm_r2.mean()), "norm_r2_max": float(norm_r2.max()),
        "norm_r2_all_layers": float(norm_r2_all.mean()),
        "gate_r2": float(gate_r2.mean()) if gate_r2 is not None else None,
        "bayes": bayes,
        "recovery": float(np.mean([x["mcc"] for x in saes])),
        "recovery_sae": [x["mcc"] for x in saes],
        "recovery_sd": float(np.std([x["mcc"] for x in saes], ddof=1)) if len(saes) > 1 else 0.0,
        "cosine_recovery": float(np.mean([x["cos_recovery"] for x in saes])),
        "saes": saes, "pos_min": POS_MIN, "sink_q_min": SINK_Q_MIN,
        "n_train_rows": int(len(xtr)), "n_test_rows": int(len(xte)), "scale": s,
        "sec": round(time.time() - t0, 1),
    }
    (rd / f"measure_{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(rd / f"acts_{tag}.npz", H_test=Hte.astype(np.float16), mu=mu,
                        scale=np.array(s), norm_test=te["NORM"].astype(np.float32),
                        test_seed=np.array(TEST_SEED), n_test=np.array(N_TEST),
                        pos_min=np.array(POS_MIN), **codes)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}))
    if bayes:
        print("bayes", json.dumps(bayes))


if __name__ == "__main__":
    main()
