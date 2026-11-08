"""Measure one trained (or untrained) model: sink mass, massive activations, logit-lens KL, SAE
recovery of the planted latents, a linear-readout ceiling, the Bayes ceiling, floors and hiding checks.

uv run python measure.py --run NAME [--ckpt STEP] [--untrained] [--layer 2] [--sae-seeds 0,1,2]
                         [--m 256] [--k 8] [--sae-rows all|P] [--rmsnorm] [--bayes] [--sink-only]

Rows (PREREG.md section 1). SAE training set = N_TRAIN sequences (seed 777000); score set = N_TEST
sequences (seed 778000). Scored positions P = the generator's `score` mask at positions >= 32 (sc:
trigger queries; sa: evidence-stream positions without recall; iid: all). SAE trains on all positions
1..127 (--sae-rows all, primary) or on P only (--sae-rows P, the named fallback). Position 0 (BOS)
never enters. Labels = the sequence's true on/off latents, repeated over its rows.

Recovery ("MCC", primary): for each planted latent j, the largest Pearson correlation between any SAE
feature's activation and z_j over the scored rows (dead/constant features count 0); mean over 64.
Writes runs/<run>/measure_<tag>.json and runs/<run>/acts_<tag>.npz (scored-row activations, the
standardiser and SAE weights: enough to rebuild every number without this script).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from scipy.optimize import linear_sum_assignment

from bayes import Bayes
from gen import GenCfg, World
from model import ModelCfg, TinyLM
from sae import encode_np, fvu, train_sae

HERE = Path(__file__).parent
POS_MIN = 32          # scored positions >= 32
SINK_Q_MIN = 8        # sink mass over query positions 8..127
TRAIN_SEED, TEST_SEED = 777_000, 778_000
N_TRAIN = 2048
N_TEST = {"iid": 512, "sa": 512, "sc": 1000}   # >= 20k scored rows per model (Kasper #17202)


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


def collect(model, world, n, seed, layer, bs=128):
    """Full per-position arrays for n sequences."""
    toks, z, fresh, score = world.sample(n, seed)
    H, NORM, GATE = [], [], []
    sink, null, trig, kl_sum, kl_n = 0.0, 0.0, 0.0, 0.0, 0
    Pm = score.copy()
    Pm[:, :POS_MIN] = False
    massive, TT, A0 = [], [], []
    for i in range(0, n, bs):
        t = mx.array(toks[i:i + bs])
        logits, resid, info = model.trace(t)
        h = resid[layer]
        mx.eval(logits, *resid, *[d["attn"] for d in info])
        hb = np.array(h)
        H.append(hb)
        NORM.append(np.stack([np.linalg.norm(np.array(r), axis=-1) for r in resid], -1))
        g = info[layer - 1]["gate"] if layer > 0 else None
        if g is not None:
            GATE.append(np.array(g))
        if layer > 0:
            A0.append(np.array(info[layer - 1]["attn"][:, :, SINK_Q_MIN:, 0]).astype(np.float16))
        a0 = np.stack([np.array(d["attn"][:, :, SINK_Q_MIN:, 0]).mean((0, 2)) for d in info])
        sink = sink + a0 * len(t)
        c_ = world.cfg
        if c_.organism == "sc":   # mass on trigger tokens (a recurring no-op candidate)
            tm = mx.array(((toks[i:i + bs] >= c_.trigger) &
                           (toks[i:i + bs] < c_.trigger + c_.n_trig)).astype(np.float32))
            trig = trig + np.stack([np.array((d["attn"][:, :, SINK_Q_MIN:, :] * tm[:, None, None, :])
                                             .sum(-1)).mean((0, 2)) for d in info]) * len(t)
        if info[0]["null"] is not None:
            null = null + np.stack([np.array(d["null"][:, :, SINK_Q_MIN:]).mean((0, 2))
                                    for d in info]) * len(t)
        ab = np.abs(hb)
        amax = ab.max(-1).argmax(1)                                   # position of max |h| per seq
        TT.extend(toks[i:i + bs][np.arange(len(hb)), amax].tolist())
        massive.append(np.stack([ab.max((1, 2)), np.median(ab.reshape(len(hb), -1), 1),
                                 ab[:, 0].max(-1), ab[:, 1:].max((1, 2))], 1))
        if layer > 0:
            lens = model.lens(h)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            lq = lens - mx.logsumexp(lens, axis=-1, keepdims=True)
            kl = np.array(mx.sum(mx.exp(lp) * (lp - lq), axis=-1))
            sel = Pm[i:i + bs]
            kl_sum += float(kl[sel].sum())
            kl_n += int(sel.sum())
    mas = np.concatenate(massive).mean(0)
    c = world.cfg
    def ttype(v):
        if v == c.bos:
            return "bos"
        if v < c.vocab:
            return "content"
        if c.organism == "sc":
            if v < c.trigger:
                return "filler"
            return "trigger" if v < c.trigger + c.n_trig else "copy_trigger"
        return "other"
    types = [ttype(v) for v in TT]
    max_type = {k: types.count(k) / len(types) for k in sorted(set(types))}
    return {
        "toks": toks, "z": z, "fresh": fresh, "score": score, "P": Pm,
        "H": np.concatenate(H).astype(np.float32), "NORM": np.concatenate(NORM),
        "GATE": np.concatenate(GATE) if GATE else None,
        "A0": np.concatenate(A0) if A0 else None,
        "sink_heads": sink / n, "trig_heads": trig / n if np.ndim(trig) else None,
        "null_heads": (null / n) if info[0]["null"] is not None else None,
        "kl": kl_sum / kl_n if kl_n else None,
        "massive": dict(zip(["max_abs", "median_abs", "bos_max_abs", "nonbos_max_abs"],
                            map(float, mas))) | {"max_token_type": max_type},
    }


def rows(arr, mask):
    """arr [n, T, ...] -> rows where mask [n, T] is True."""
    return arr[mask]


def labels(z, mask):
    return np.repeat(z, mask.sum(1), axis=0).astype(np.float32)


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
    ap.add_argument("--sae-rows", default="all", choices=["all", "P"])
    ap.add_argument("--rmsnorm", action="store_true", help="SAE on per-token RMS-normalised h")
    ap.add_argument("--bayes", action="store_true", help="also compute the Bayes ceiling")
    ap.add_argument("--sink-only", action="store_true", help="only sink mass per block, no SAE")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-train", type=int, default=0,
                    help="also save the SAE training rows of the first N training sequences")
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
    n_params = int(sum(v.size for _, v in tree_flatten(model.parameters())))
    layer = a.layer
    eval_loss = next((h["eval"] for h in log["history"] if h["step"] == step), None) if step else None

    if a.sink_only:
        te = collect(model, world, 256, TEST_SEED, layer)
        print(json.dumps({"run": a.run, "ckpt": step, "eval_loss": eval_loss,
                          "sink_blk": np.round(te["sink_heads"].mean(1), 4).tolist(),
                          "sink_heads": np.round(te["sink_heads"], 3).tolist(),
                          "trig_blk": (np.round(te["trig_heads"].mean(1), 4).tolist()
                                       if te["trig_heads"] is not None else None),
                          "null_blk": (np.round(te["null_heads"].mean(1), 4).tolist()
                                       if te["null_heads"] is not None else None),
                          "massive": {k: (round(v, 2) if isinstance(v, float) else v)
                                      for k, v in te["massive"].items()}}), flush=True)
        return

    tag = a.tag or (f"{'untrained_' if a.untrained else ''}L{layer}_m{a.m}_k{a.k}"
                    + ("_rowsP" if a.sae_rows == "P" else "") + ("_rms" if a.rmsnorm else "")
                    + (f"_c{a.ckpt}" if a.ckpt else ""))
    t0 = time.time()
    n_test = N_TEST[g.organism]
    tr = collect(model, world, N_TRAIN, TRAIN_SEED, layer)
    te = collect(model, world, n_test, TEST_SEED, layer)
    d = tr["H"].shape[-1]
    T = g.seq_len

    all_tr = np.zeros_like(tr["P"])
    all_tr[:, 1:] = True                                   # SAE rows: all positions 1..127
    sae_mask = all_tr if a.sae_rows == "all" else tr["P"]
    allte = np.zeros_like(te["P"])
    allte[:, POS_MIN:] = True                              # secondary score rows: all >= 32

    def prep(H):
        return H / np.sqrt((H ** 2).mean(-1, keepdims=True)) if a.rmsnorm else H

    Hs = prep(rows(tr["H"], sae_mask))
    mu = Hs.mean(0)
    s = float(np.sqrt(((Hs - mu) ** 2).sum(1).mean() / d))     # mean squared norm -> d
    std = lambda H: (prep(H) - mu) / s
    xsae = (Hs - mu) / s
    xtrP, ZtrP = std(rows(tr["H"], tr["P"])), labels(tr["z"], tr["P"])
    xte, Zte = std(rows(te["H"], te["P"])), labels(te["z"], te["P"])
    xteA, ZteA = std(rows(te["H"], allte)), labels(te["z"], allte)

    # readout ceiling on P: ridge from h to the 64 indicators
    pr = ridge_fit(xtrP, ZtrP)(xte)
    readout_r2 = r2(pr, Zte)
    readout_corr = np.array([np.corrcoef(pr[:, j], Zte[:, j])[0, 1] for j in range(g.n_lat)])

    # registered norm check on P: ridge R^2 of z_j from the per-token norm of h_layer (one scalar)
    ntr, nte = rows(tr["NORM"], tr["P"]), rows(te["NORM"], te["P"])
    def norm_fit(cols):
        mm, ss = ntr[:, cols].mean(0), ntr[:, cols].std(0) + 1e-12
        return r2(ridge_fit((ntr[:, cols] - mm) / ss, ZtrP, lam=1e-6)((nte[:, cols] - mm) / ss), Zte)
    norm_r2 = norm_fit([layer])
    norm_r2_all = norm_fit(list(range(ntr.shape[1])))

    gate_r2 = None
    if te["GATE"] is not None:
        gtr, gte = rows(tr["GATE"], tr["P"]), rows(te["GATE"], te["P"])
        gm, gs = gtr.mean(0), gtr.std(0) + 1e-6
        gate_r2 = r2(ridge_fit((gtr - gm) / gs, ZtrP)((gte - gm) / gs), Zte)

    # least-squares directions of h ~ z on P (secondary "cosine" recovery)
    zc = ZtrP - ZtrP.mean(0)
    Wz = np.linalg.lstsq(zc, xtrP - xtrP.mean(0), rcond=None)[0]
    Wz_u = Wz / np.linalg.norm(Wz, axis=1, keepdims=True)

    # random-dictionary floor (Vera #17212): 256 random unit directions, ReLU of projections, rows P
    rr = np.random.default_rng(12345)
    Rd = rr.normal(size=(a.m, d))
    Rd /= np.linalg.norm(Rd, axis=1, keepdims=True)
    random_floor = float(corr_matrix(np.maximum(xte @ Rd.T, 0), Zte).max(0).mean())

    bayes = None
    if a.bayes:
        B = Bayes(world)
        mtr, _ = B.run(tr["toks"], tr["fresh"])
        mte, lte = B.run(te["toks"], te["fresh"])
        Mtr, Mte = rows(mtr, tr["P"]), rows(mte, te["P"])
        bayes = {"posterior_r2": float(r2(ridge_fit(Mtr, ZtrP)(Mte), Zte).mean()),
                 "posterior_corr": float(np.mean([np.corrcoef(Mte[:, j], Zte[:, j])[0, 1]
                                                  for j in range(g.n_lat)])),
                 "loss_fresh": float(np.nanmean(lte))}

    nseq = te["P"].sum(1)
    # shuffled-label control: give each sequence's rows the latents of another sequence
    perm = np.random.default_rng(0).permutation(n_test)
    Zsh = np.repeat(te["z"][perm], nseq, axis=0).astype(np.float32)

    saes, codes = [], {}
    for sd in [int(x) for x in a.sae_seeds.split(",")]:
        sae, hist = train_sae(xsae, a.m, a.k, seed=sd, steps=a.sae_steps)
        f = encode_np(sae, xte)
        C = corr_matrix(f, Zte)
        mcc, arg = C.max(0), C.argmax(0)
        ri, ci = linear_sum_assignment(-C)
        hung = float(C[ri, ci].mean())
        msh = corr_matrix(f, Zsh).max(0)
        fA = encode_np(sae, xteA)
        mccA = corr_matrix(fA, ZteA).max(0)
        dec = np.array(sae.W_dec)
        dec_u = dec / np.linalg.norm(dec, axis=1, keepdims=True)
        cosr = (dec_u @ Wz_u.T).max(0)
        fall = encode_np(sae, xsae[np.random.default_rng(sd).permutation(len(xsae))[:50000]])
        dead = float((fall.max(0) <= 0).mean())
        saes.append({"sae_seed": sd, "mcc": float(mcc.mean()), "mcc_per_latent": mcc.tolist(),
                     "best_feature": arg.tolist(), "n_distinct_best": int(len(set(arg.tolist()))),
                     "mcc_hungarian": hung, "mcc_shuffled": float(msh.mean()),
                     "mcc_all_pos": float(mccA.mean()), "cos_recovery": float(cosr.mean()),
                     "fvu": fvu(sae, xte), "dead_share": dead, "hist": hist,
                     "frac_latents_gt_0p5": float((mcc > 0.5).mean()),
                     "readout_minus_sae": float((readout_corr - mcc).mean())})
        for nm in ("W_enc", "b_enc", "W_dec", "b_dec"):
            codes[f"{nm}_{sd}"] = np.array(getattr(sae, nm))
        print(f"sae seed {sd}: mcc {mcc.mean():.4f} hung {hung:.4f} shuffled {msh.mean():.4f} "
              f"allpos {mccA.mean():.4f} cos {cosr.mean():.3f} fvu {saes[-1]['fvu']:.3f} "
              f"dead {dead:.3f}", flush=True)

    sh, nh = te["sink_heads"], te["null_heads"]
    mid = layer - 1 if layer > 0 else 0
    res = {
        "run": a.run, "attn": mc.attn, "seed": log["seed"], "d": mc.d, "organism": g.organism,
        "untrained": a.untrained, "ckpt": step, "layer": layer, "tag": tag, "m": a.m, "k": a.k,
        "sae_rows": a.sae_rows, "rmsnorm": a.rmsnorm, "n_params": n_params,
        "eval_loss": eval_loss,
        "sink_mass": float(sh[mid].mean()), "sink_mass_all": float(sh.mean()),
        "sink_blk": sh.mean(1).tolist(), "sink_heads": sh.tolist(),
        "sink_eps_share": float((sh > 0.3).mean()),
        "null_mass": float(nh[mid].mean()) if nh is not None else 0.0,
        "dump_mass": float(sh[mid].mean() + (nh[mid].mean() if nh is not None else 0.0)),
        "null_heads": nh.tolist() if nh is not None else None,
        "trig_mass": float(te["trig_heads"][mid].mean()) if te["trig_heads"] is not None else 0.0,
        "trig_blk": te["trig_heads"].mean(1).tolist() if te["trig_heads"] is not None else None,
        "noop_mass": float(sh[mid].mean() + (te["trig_heads"][mid].mean()
                                             if te["trig_heads"] is not None else 0.0)),
        "massive": te["massive"], "logit_lens_kl": te["kl"],
        "readout_r2": float(readout_r2.mean()), "readout_corr": float(readout_corr.mean()),
        "norm_r2": float(norm_r2.mean()), "norm_r2_max": float(norm_r2.max()),
        "norm_r2_all_layers": float(norm_r2_all.mean()),
        "gate_r2": float(gate_r2.mean()) if gate_r2 is not None else None,
        "random_floor": random_floor, "bayes": bayes,
        "recovery": float(np.mean([x["mcc"] for x in saes])),
        "recovery_sae": [x["mcc"] for x in saes],
        "recovery_sd": float(np.std([x["mcc"] for x in saes], ddof=1)) if len(saes) > 1 else 0.0,
        "recovery_hungarian": float(np.mean([x["mcc_hungarian"] for x in saes])),
        "recovery_all_pos": float(np.mean([x["mcc_all_pos"] for x in saes])),
        "cosine_recovery": float(np.mean([x["cos_recovery"] for x in saes])),
        "dead_share": float(np.mean([x["dead_share"] for x in saes])),
        "saes": saes, "pos_min": POS_MIN, "sink_q_min": SINK_Q_MIN, "n_test": n_test,
        "n_sae_rows": int(len(xsae)), "n_scored_rows": int(len(xte)), "scale": s,
        "sec": round(time.time() - t0, 1),
    }
    (rd / f"measure_{tag}.json").write_text(json.dumps(res, indent=1))
    seq_idx, pos = np.nonzero(te["P"])
    if a.save_train:
        extra_tr = {"H_train": ((prep(tr["H"][:a.save_train, 1:]) - mu) / s).astype(np.float16),
                    "z_train_seq": tr["z"][:a.save_train].astype(np.uint8),
                    "tokens_train": tr["toks"][:a.save_train].astype(np.int16)}
    else:
        extra_tr = {}
    extra = {"A0_sink_block": te["A0"]} if te["A0"] is not None else {}
    np.savez_compressed(rd / f"acts_{tag}.npz", H_scored=xte.astype(np.float16),
                        z_scored=Zte.astype(np.uint8), seq_idx=seq_idx.astype(np.int32),
                        pos=pos.astype(np.int16), tokens=te["toks"].astype(np.int16),
                        z_seq=te["z"].astype(np.uint8), layer=np.array(layer),
                        sink_block=np.array(layer), sink_q_min=np.array(SINK_Q_MIN), **extra,
                        **extra_tr,
                        rows_per_seq=nseq, mu=mu, scale=np.array(s),
                        norm_scored=nte.astype(np.float32), test_seed=np.array(TEST_SEED),
                        n_test=np.array(n_test), pos_min=np.array(POS_MIN), **codes)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}))
    if bayes:
        print("bayes", json.dumps(bayes))


if __name__ == "__main__":
    main()
