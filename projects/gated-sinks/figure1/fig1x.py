"""Figure 1 extractor: one held-out sequence, two models (standard vs gated attention).

uv run python fig1x.py --a RUN[:CKPT]:TAG --b RUN[:CKPT]:TAG [--seq N] [--q Q] --out example.json
Reads only the project's saved files (log.json, ckpt/weights, measure_<TAG>.json, acts_<TAG>.npz) and
re-runs the trained model on one test sequence for its full attention row. Every number is asserted
against the saved measure json / npz where both exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2] / "kran-research/projects/gated-sinks"
sys.path.insert(0, str(PROJ))
import mlx.core as mx  # noqa: E402

from gen import GenCfg, World  # noqa: E402
from model import ModelCfg, TinyLM  # noqa: E402


def topk_encode(x, W_enc, b_enc, b_dec, k):
    a = np.maximum((x - b_dec) @ W_enc + b_enc, 0.0)
    thr = np.sort(a, axis=-1)[:, -k][:, None]
    return np.where(a >= thr, a, 0.0)


def load_arm(spec):
    run, ckpt, tag = (spec.split(":") + [None, None])[:3]
    rd = PROJ / "runs" / run
    log = json.loads((rd / "log.json").read_text())
    meas = json.loads((rd / f"measure_{tag}.json").read_text())
    npz = np.load(rd / f"acts_{tag}.npz")
    ckpt = int(ckpt) if ckpt else meas.get("ckpt")
    mc = ModelCfg(**log["model"])
    model = TinyLM(mc)
    wf = rd / (f"ckpt_{ckpt}.safetensors" if ckpt and (rd / f"ckpt_{ckpt}.safetensors").exists()
               else "weights.safetensors")
    model.load_weights(str(wf))
    mx.eval(model.parameters())
    return dict(run=run, ckpt=ckpt, tag=tag, log=log, meas=meas, npz=npz, model=model, mc=mc,
                weights=str(wf.relative_to(PROJ)))


def ttype(c, v):
    if v == c.bos:
        return "start"
    if v < c.vocab:
        return "evidence"
    if c.filler0 <= v < c.filler0 + c.n_filler:
        return "filler"
    if c.trigger <= v < c.trigger + c.n_trig:
        return "trigger"
    return "other"


def arm_numbers(A, seq, q, cause):
    npz, meas, model = A["npz"], A["meas"], A["model"]
    toks = npz["tokens"][seq].astype(np.int32)
    layer = int(npz["layer"])
    blk = int(npz["sink_block"]) if "sink_block" in npz.files else layer
    _, resid, info = model.trace(mx.array(toks[None]))
    att = np.array(info[blk - 1]["attn"][0])                     # [H, T, T]
    gate = info[blk - 1]["gate"]
    qmin = int(npz["sink_q_min"])
    # first-token mass per query, head mean; must equal the saved A0 for this sequence
    a0 = att[:, qmin:, 0]
    saved = npz["A0_sink_block"][seq].astype(np.float32)
    assert np.allclose(a0, saved, atol=2e-3), np.abs(a0 - saved).max()
    sae_seeds = [s["sae_seed"] for s in meas["saes"]]
    k = meas["k"]
    # per cause: recovery (best corr) per SAE seed, from the measure json
    rec_c = [meas["saes"][i]["mcc_per_latent"][cause] for i in range(len(sae_seeds))]
    s0 = 0
    bf = meas["saes"][s0]["best_feature"][cause]
    X = npz["H_scored"].astype(np.float32)
    F = topk_encode(X, npz[f"W_enc_{s0}"], npz[f"b_enc_{s0}"], npz[f"b_dec_{s0}"], k)
    f = F[:, bf]
    zc = npz["z_scored"][:, cause].astype(np.float32)
    r = float(np.corrcoef(f, zc)[0, 1])
    assert abs(r - rec_c[s0]) < 2e-3, (r, rec_c[s0])
    # the example sequence's scored rows
    m = npz["seq_idx"] == seq
    out = {
        "run": A["run"], "ckpt": A["ckpt"], "weights": A["weights"], "tag": A["tag"],
        "attn": A["mc"].attn, "seed": A["log"]["seed"], "n_params": meas["n_params"],
        "block": blk, "layer": layer, "query": q,
        "attn_row_mean": att[:, q, :q + 1].mean(0).round(4).tolist(),
        "attn_row_heads": att[:, q, :q + 1].round(4).tolist(),
        "first_mass_by_query": a0.mean(0).round(4).tolist(),
        "first_mass_seq": float(a0.mean()),
        "gate_mean_at_q": (float(np.array(gate[0, q]).mean()) if gate is not None else None),
        "sink_mass": meas["sink_mass"], "recovery": meas["recovery"],
        "recovery_sae": meas["recovery_sae"], "readout_r2": meas["readout_r2"],
        "random_floor": meas["random_floor"],
        "cause_recovery_sae": rec_c, "cause_recovery": float(np.mean(rec_c)),
        "best_feature_seed0": int(bf),
        "feat_on": f[zc == 1].round(3).tolist()[:4000], "feat_off_sample": f[zc == 0][::8].round(3).tolist(),
        "feat_on_mean": float(f[zc == 1].mean()), "feat_off_mean": float(f[zc == 0].mean()),
        "feat_on_share_active": float((f[zc == 1] > 0).mean()),
        "feat_off_share_active": float((f[zc == 0] > 0).mean()),
        "seq_rows_pos": npz["pos"][m].tolist(), "seq_rows_feat": f[m].round(3).tolist(),
        "mcc_per_latent_mean": np.mean([s["mcc_per_latent"] for s in meas["saes"]], 0).round(4).tolist(),
    }
    return out, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=RUN:CKPT:TAG (repeat)")
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--cause", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = {}
    for s in a.arm:
        nm, spec = s.split("=", 1)
        arms[nm] = load_arm(spec)
    A0 = next(iter(arms.values()))
    g = GenCfg(**A0["log"]["gen"])
    world = World(g)
    for nm, A in arms.items():
        assert A["log"]["gen"] == A0["log"]["gen"], f"{nm}: different generator"
        assert (A["npz"]["tokens"] == A0["npz"]["tokens"]).all(), f"{nm}: different test sequences"
    ta, zs = A0["npz"]["tokens"], A0["npz"]["z_seq"]
    # cause: the one whose per-arm recoveries sit closest to each arm's own mean (most typical)
    per = {nm: np.mean([s["mcc_per_latent"] for s in A["meas"]["saes"]], 0) for nm, A in arms.items()}
    dev = sum(np.abs(p - p.mean()) for p in per.values())
    cause = a.cause if a.cause is not None else int(np.argmin(dev))
    seq = a.seq
    if seq is None:
        seq = int(np.nonzero(zs[:, cause])[0][0])
    assert zs[seq, cause] == 1
    toks = ta[seq].astype(int)
    types = [ttype(g, int(v)) for v in toks]
    q = a.q
    if q is None:   # an idle (filler) query near position 100
        q = next(t for t in range(100, 128) if types[t] == "filler")
    out_arms = {nm: arm_numbers(A, seq, q, cause)[0] for nm, A in arms.items()}
    on = np.nonzero(zs[seq])[0].tolist()
    ev_cause = [[c for c in on if world.U[c, v] > 0] if types[t] == "evidence" else []
                for t, v in enumerate(toks)]
    res = {
        "gen": A0["log"]["gen"], "test_seed": int(A0["npz"]["test_seed"]), "seq": seq, "query": q,
        "tokens": toks.tolist(), "types": types, "causes_on": on, "cause": cause,
        "cause_tokens": np.nonzero(world.U[cause])[0].tolist(), "evidence_by_cause": ev_cause,
        "arms": out_arms,
    }
    Path(a.out).write_text(json.dumps(res))
    keep = ("sink_mass", "first_mass_seq", "recovery", "readout_r2", "cause_recovery",
            "gate_mean_at_q", "random_floor")
    print(json.dumps({"seq": seq, "q": q, "cause": cause, "causes_on": on, "arms": {
        nm: {**{k: e[k] for k in keep}, "first_at_q": e["attn_row_mean"][0]}
        for nm, e in out_arms.items()}}, indent=1))


if __name__ == "__main__":
    main()
