"""Paired reads of one policy on the served graph (gate before training, status after): f, f_null, D, Pinsker, per-token
TV sum, Rao-Blackwellised cheat per build, served gap, text part, spill, and the prefix token arrays for Iris's C.

Rows = (prompt, seed r), r = 0..R-1. Each row has its own Gumbel noise generator (seed = hash of prompt index and r),
so its noise does not depend on the batch (COMMON RANDOM NUMBERS across builds and batch layouts). Per batch of B rows:
  1 none   sample (keep log-probs)                        -> toks_n, lp_n, cheat_n (answer slot on the none text)
  2 leaky  teacher-forced on toks_n, lockstep vs 1        -> lp_l_n, per-token TV and KL(none || leaky), cheat_ln
  3 leaky  sample, same noise                             -> toks_l, cheat_l (+ the planted bias when --plant)
  4 none   teacher-forced on toks_l                       -> cheat_nl (= the text part's answer slot)
  5 none   sample, same rows in another batch layout      -> toks_n2 (f_null; --fnull)
D = mean over rows of max(0, 1 - exp(sum_t lp_l_n - lp_n)) (exact Monte-Carlo TV, prefixes drawn from none).
served gap = cheat_n - cheat_l; text part = cheat_n - cheat_nl; spill = gap - text part.

    env/.venv/bin/python read.py --policy base --splits train,test --R 4 --fnull --out data/gate/base
    env/.venv/bin/python read.py --policy data/runs/k16_s0/adapter.pt --splits test --R 4 --out data/reads/k16_s0
"""
import argparse
import json
import time

import numpy as np
import torch

import mt

HERE = mt.HERE


def boot_ci(x, groups, n=2000, seed=0):
    """Paired bootstrap over prompts: x per row, groups = prompt index per row."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    per = np.array([x[groups == g].mean() for g in ug])
    bs = per[rng.integers(0, len(per), (n, len(per)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def gens(rows_pi, rows_r):
    return [torch.Generator(device=mt.DEV).manual_seed(int(1_000_003 * (p + 1) + 7919 * r)) for p, r in zip(rows_pi, rows_r)]


def main(argv=None, S=None):
    """argv: CLI args (None = sys.argv). S: an existing Sampler with B = --B and k = --k (rl.py's in-process gate read:
    the same compiled graph and the current adapter, no reload); None = load --policy and build one."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="'base' (raw Qwen3-0.6B, LoRA B = 0) or an adapter path")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--fnull", action="store_true")
    ap.add_argument("--plant-tok", type=int, default=None)
    ap.add_argument("--plant-eps", type=float, default=0.0)
    ap.add_argument("--hash-sigma", type=float, default=0.0, help="diffuse hash cue on the probed sampler")
    ap.add_argument("--plant-build", default="leaky", choices=["leaky", "none"],
                    help="build of the probed sampler when a cue is planted (none = the cue alone)")
    ap.add_argument("--forced-gate", action="store_true", help="planted arm: g(X) vs g(notX) on the none graph")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--skip-text", action="store_true", help="length scan: skip pass 4 (text part); D, f, C inputs only")
    ap.add_argument("--texts-only", action="store_true", help="C_sup train texts: skip pass 2 (no D); implies --skip-text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dec-bias", type=float, default=0.0, help="the training run's fixed ' read' logit offset (mt.DEC)")
    ap.add_argument("--exact", action="store_true", help="bit-exact probe: the probed sampler runs on the SCORE build (hook keeps only the probe score)")
    a = ap.parse_args(argv)
    mt.DEC["bias"] = a.dec_bias
    a.skip_text = a.skip_text or a.texts_only
    t0 = time.time()
    if S is None:
        pm, inner = mt.load_policy(None if a.policy == "base" else HERE / a.policy)
    rows = [r for r in mt.prompts() if r["split"] in a.splits.split(",")]
    if a.n > 0:
        rows = rows[: a.n]
    pidx = np.array([int(r["id"][2:]) for r in rows])
    rpi = np.repeat(pidx, a.R)
    rr = np.tile(np.arange(a.R), len(rows))
    split = np.repeat(np.array([r["split"] for r in rows]), a.R)
    ids_all = torch.tensor([r["ids"] for r in rows], device=mt.DEV).repeat_interleave(a.R, 0)
    N, B, k = len(rpi), a.B, a.k
    if S is None:
        S = mt.Sampler(inner, B, k)
    assert S.B == B and S.k == k, (S.B, S.k, B, k)
    keys = ["toks_n", "toks_l", "toks_n2", "lp_n", "lp_l_n", "tv", "kl", "c_n", "c_l", "c_ln", "c_nl", "c_l_gate",
            "c_X", "c_notX"]
    res = {kk: [] for kk in keys}
    plant = a.plant_tok is not None
    hcue = a.hash_sigma > 0
    PB = a.plant_build if (plant or hcue) else "leaky"
    if a.exact:
        PB = "score"

    def cheat(d2):
        return torch.softmax(d2, -1)[:, 0].cpu().numpy()

    for s0 in range(0, N, B):
        j = list(range(s0, min(s0 + B, N)))
        nreal = len(j)
        j = j + [j[-1]] * (B - nreal)
        ids = ids_all[j]
        o1 = S.rollout(ids, "none", gen=gens(rpi[j], rr[j]), keep_logp=not a.skip_text)  # full log-probs: [k, B, V]
        one = torch.ones(B, dtype=torch.bool, device=mt.DEV)
        pl = (a.plant_tok, a.plant_eps, one) if plant else None
        hc = (a.hash_sigma, one) if hcue else None
        # the probed sampler teacher-forced on the off text (D); --texts-only skips it (C_sup train texts only)
        o2 = o1 if a.texts_only else S.rollout(ids, PB, forced=o1["toks"], ref_logp=o1.get("logp"), plant=pl, hcue=hc)
        o3 = S.rollout(ids, PB, gen=gens(rpi[j], rr[j]), plant=pl, hcue=hc)
        o4 = o3 if a.skip_text else S.rollout(ids, "none", forced=o3["toks"])
        sl = slice(0, nreal)
        res["toks_n"].append(o1["toks"][sl].cpu().numpy()); res["toks_l"].append(o3["toks"][sl].cpu().numpy())
        if k:
            res["lp_n"].append(o1["lp_tok"][sl].cpu().numpy()); res["lp_l_n"].append(o2["lp_tok"][sl].cpu().numpy())
            if "tv" in o2:
                res["tv"].append(o2["tv"][sl].cpu().numpy()); res["kl"].append(o2["kl"][sl].cpu().numpy())
        res["c_n"].append(cheat(o1["d2"])[sl]); res["c_ln"].append(cheat(o2["d2"])[sl])
        res["c_l"].append(cheat(o3["d2"])[sl]); res["c_nl"].append(cheat(o4["d2"])[sl])
        if a.forced_gate:
            X = a.plant_tok
            fx = o1["toks"].clone()
            fx[:, 0] = X
            # positions 2..k: same noise as the none sample (CRN); position 1 forced to X
            g5 = gens(rpi[j], rr[j])
            o5 = S.rollout(ids, "none", gen=g5, ban=None, plant=(X, 1e4, torch.ones(B, dtype=torch.bool, device=mt.DEV)))
            o6 = S.rollout(ids, "none", gen=gens(rpi[j], rr[j]), ban=X)
            assert bool((o5["toks"][:, 0] == X).all())
            res["c_X"].append(cheat(o5["d2"])[sl]); res["c_notX"].append(cheat(o6["d2"])[sl])
        print(f"batch {s0 // B + 1}/{(N + B - 1) // B} {time.time() - t0:.0f}s", flush=True)
    if a.fnull:  # same rows, another batch layout (reversed order), same per-row noise
        order = np.arange(N)[::-1].copy()
        tn2 = np.zeros((N, k), np.int64)
        for s0 in range(0, N, B):
            j = list(order[s0: s0 + B])
            nreal = len(j)
            j = j + [j[-1]] * (B - nreal)
            o = S.rollout(ids_all[j], "none", gen=gens(rpi[j], rr[j]))
            tn2[j[:nreal]] = o["toks"][:nreal].cpu().numpy()
        res["toks_n2"] = [tn2]
    R = {kk: np.concatenate(v) for kk, v in res.items() if v}
    out = {"policy": a.policy, "splits": a.splits, "R": a.R, "k": k, "n_rows": int(N), "B": B,
           "plant_tok": a.plant_tok, "plant_eps": a.plant_eps, "hash_sigma": a.hash_sigma, "plant_build": PB,
           "dec_bias": a.dec_bias, "exact": a.exact,
           "torch": torch.__version__,
           "skip_text": a.skip_text, "texts_only": a.texts_only,
           "note": "skip_text: no text part (text = gap there) and no per-token TV/KL; texts_only: no D / direct "
                   "(pass 2 skipped; lp_l_n in the npz is a copy of lp_n there and must not be used)"}
    for sp in sorted(set(split)):
        m = split == sp
        g = rpi[m]
        d = {}
        cn, cl, cnl = R["c_n"][m], R["c_l"][m], R["c_nl"][m]
        d["cheat_none"], d["cheat_leaky"] = float(cn.mean()), float(cl.mean())
        gap, text = cn - cl, cn - cnl
        d["gap"], d["gap_ci"] = float(gap.mean()), boot_ci(gap, g)
        d["text"], d["text_ci"] = float(text.mean()), boot_ci(text, g)
        d["spill"], d["spill_ci"] = float((gap - text).mean()), boot_ci(gap - text, g)
        d["direct_same_text"] = float((R["c_n"][m] - R["c_ln"][m]).mean())
        if k:
            diff = (R["toks_n"][m] != R["toks_l"][m]).any(1)
            d["f"] = float(diff.mean())
            L = (R["lp_l_n"][m] - R["lp_n"][m]).sum(1)
            tvr = np.maximum(0.0, 1.0 - np.exp(L))
            d["D"], d["D_ci"] = float(tvr.mean()), boot_ci(tvr, g)
            if a.texts_only:
                d["D"] = d["D_ci"] = d["direct_same_text"] = None
            if "kl" in R:
                klseq = R["kl"][m].sum(1)
                d["kl_seq"] = float(klseq.mean())
                d["pinsker"] = float(np.sqrt(klseq.mean() / 2))
                d["tv_sum"] = float(R["tv"][m].sum(1).mean())
                d["tv_per_pos"] = [float(x) for x in R["tv"][m].mean(0)]
            d["first_diff_pos_hist"] = np.bincount(np.where(diff, (R["toks_n"][m] != R["toks_l"][m]).argmax(1), k)[diff],
                                                   minlength=k).tolist()
            if "toks_n2" in R:
                d["f_null"] = float((R["toks_n"][m] != R["toks_n2"][m]).any(1).mean())
        if a.forced_gate:
            sw = R["c_notX"][m] - R["c_X"][m]
            d["forced_switch"], d["forced_switch_ci"] = float(sw.mean()), boot_ci(sw, g)
            d["g_X"], d["g_notX"] = float(R["c_X"][m].mean()), float(R["c_notX"][m].mean())
        out[sp] = d
    out["read_s"] = round(time.time() - t0, 1)
    op = HERE / a.out
    op.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(op) + ".npz", prompt_idx=rpi, seed=rr, split=split, **R)
    (Path := __import__("pathlib").Path)(str(op) + ".json").write_text(json.dumps(out, indent=1))
    print(json.dumps({kk: v for kk, v in out.items()}, indent=1)[:3000], flush=True)
    return out


if __name__ == "__main__":
    main()
