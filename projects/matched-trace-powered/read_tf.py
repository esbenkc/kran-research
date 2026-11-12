"""Teacher-forced (likelihood-ratio) joint read: both cues are read on the PROBE-OFF text only.

The probe-off sampler writes a thought; the probe-on build is teacher-forced on that same thought, so the only thing it
contributes is the probability ratio of the text. Per row (prompt, seed r), with the cue word X banned at thought
token 1 in the trace passes (as the 11-11 joint read):

  A  none, X banned, sample                 -> toks_nb, lp_nb [k], c_nb   (c = p(' read') at the answer slot)
  B  leaky, X banned, teacher-forced on A   -> lp_lb_nb [k], c_lb_nb     (leaky decision on the none text: spill only)
  C  none, X written at token 1, sample     -> c_X                       (same noise after token 1 as A)
  D  none, prefill only                     -> pX_off = p_none(X at thought token 1), unbanned
  E  (--aa) none, X banned, teacher-forced on A -> lp_aa, c_aa           A/A null: must equal lp_nb, c_nb exactly
  F  (--plain-r N) leaky, X banned, sample, same noise -> c_lb, toks_lb  the 11-11 plain CRN y_R = c_nb - c_lb
     + none, X banned, teacher-forced on toks_lb -> c_nlb                 plain text part c_nb - c_nlb (11-11 pass 5)
  G  (--cue-r N) none + cue delta, sample -> c_nc, toks_nc               the 11-11 pass 6 (Figure 1 example only)

Estimands (analysis in summarize_tf.py, numpy only; the frozen form is in PREREG.md):
  real trace   served y_R = E_none[c] - E_leaky[c_leaky] = mean(c_nb) - sum(w c_lb_nb) / sum(w)   (self-normalised)
               w = exp(sum_t lp_lb_nb - lp_nb): how much more likely the probe-on build makes the same thought
               text part = mean(c_nb) - sum(w c_nb) / sum(w); spill = served - text part (leaky numerics at the answer)
               (E_A[w] = 1 because both builds ban X and sample at temperature 1 with no truncation)
  real D       D_R = E_A[max(0, 1 - w)]            (exact Monte-Carlo TV of the two thought distributions)
  planted      y_P = dP (c_notX - c_X), c_notX = c_nb, dP = min(pX_off + delta, 0.99) - pX_off
               (exact: the cue changes only thought token 1; 11-11 DEVIATIONS D1)
  ceilings     s_R = y_R / (D_R mean c_nb);  s_P = y_P / (D_P mean c_n), c_n = (1 - pX) c_nb + pX c_X (exact mean)

Row noise = read.gens(prompt index, r) (per-row Gumbel generators, as the 11-11 read): at --r0 0 pass A reproduces
the 11-11 pass 2 (and pass F its pass 4) token for token on the same adapter. Rows run seed-major.

    env/.venv/bin/python read_tf.py --policy data/runs/joint_s0_cont/adapter.pt --R 32 --plain-r 8 \
        --out data/reads/raw/end_tf
"""
import argparse
import json
import time

import numpy as np
import torch

import mt
from read import gens

HERE = mt.HERE


@torch.no_grad()
def p_first(S, ids, X):
    """p_none(X) at thought token 1 (the rollout's own first-step logits, before any ban or cue)."""
    mt.FLAG["leaky"] = False
    mt.FLAG["score"] = False
    mt.STORE.clear()
    mt.REC.clear()
    S.cache.reset()
    pos = torch.arange(mt.T, device=mt.DEV)[None]
    lg = S.fn(ids, pos).float()
    lg[:, mt.FS:] = -float("inf")
    mt.REC.clear()
    mt.STORE.clear()
    return torch.softmax(lg, -1)[:, X]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--prompts", default="", help="other prompts file (data/...jsonl, same template); default = mt.prompts()")
    ap.add_argument("--R", type=int, default=24)
    ap.add_argument("--r0", type=int, default=0, help="first seed index (fresh noise: r0 >= the 11-11 read's R)")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--n", type=int, default=-1, help="smoke: first n prompts")
    ap.add_argument("--aa", action="store_true", help="add pass E (A/A null)")
    ap.add_argument("--plain-r", type=int, default=0, help="11-11 passes 4 + 5 (leaky, X banned, SAMPLE -> c_lb, toks_lb; "
                    "none, X banned, teacher-forced on toks_lb -> c_nlb) on rows with seed index < r0 + plain-r")
    ap.add_argument("--cue-r", type=int, default=0, help="11-11 pass 6 (none + cue delta, SAMPLE -> c_nc, toks_nc) on "
                    "rows with seed index < r0 + cue-r (Figure 1 example; not a status input)")
    ap.add_argument("--stop-at", default="", help="clock HH:MM local: stop after the first batch ending at or after it "
                    "(rows are seed-major, so a cut keeps every prompt at the same number of seeds)")
    ap.add_argument("--save-every", type=int, default=20, help="write the partial npz every N batches")
    ap.add_argument("--plant-tok", type=int, default=14656)
    ap.add_argument("--cue-delta", type=float, default=0.10)
    ap.add_argument("--dec-bias", type=float, default=-2.255)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mt.DEC["bias"] = a.dec_bias
    t0 = time.time()
    pm, inner = mt.load_policy(None if a.policy == "base" else HERE / a.policy)
    if a.prompts:
        rows = [json.loads(l) for l in open(HERE / a.prompts)]
    else:
        rows = mt.prompts()
    rows = [r for r in rows if r["split"] in a.splits.split(",")]
    if a.n > 0:
        rows = rows[: a.n]
    pidx = np.array([int(r["id"][2:]) for r in rows])
    # SEED-MAJOR order: all prompts at seed r0, then r0 + 1, ... Each row has its own noise generator, so results do not
    # depend on the batch layout; a clock cut drops whole seeds, never prompts.
    rpi = np.tile(pidx, a.R)
    rr = np.repeat(np.arange(a.r0, a.r0 + a.R), len(rows))
    ids_all = torch.tensor([r["ids"] for r in rows], device=mt.DEV).repeat(a.R, 1)
    stop_at = None
    if a.stop_at:
        hh, mm = (int(x) for x in a.stop_at.split(":"))
        lt = time.localtime()
        stop_at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mm, 0, 0, 0, -1))
    N, B, k, X = len(rpi), a.B, a.k, a.plant_tok
    S = mt.Sampler(inner, B, k)
    keys = ["toks_nb", "lp_nb", "lp_lb_nb", "c_nb", "c_lb_nb", "c_X", "pX_off"] + (["lp_aa", "c_aa"] if a.aa else [])
    res = {kk: [] for kk in keys}
    side = {kk: [] for kk in ("plain_row", "c_lb", "toks_lb", "c_nlb", "cue_row", "c_nc", "toks_nc")}

    def cheat(d2):
        return torch.softmax(d2, -1)[:, 0]

    one = torch.ones(B, dtype=torch.bool, device=mt.DEV)

    def save(final):
        R = {kk: np.concatenate(v) for kk, v in res.items() if v}
        n = len(R["c_nb"])
        meta = {"policy": a.policy, "splits": a.splits, "prompts": a.prompts or "data/prompts.jsonl", "R": a.R,
                "r0": a.r0, "order": "seed-major", "k": k, "B": B, "n_rows_planned": int(N), "n_rows": int(n),
                "complete": bool(final and n == N), "plant_tok": X, "cue_delta": a.cue_delta, "dec_bias": a.dec_bias,
                "aa": a.aa, "plain_r": a.plain_r, "torch": torch.__version__, "read_s": round(time.time() - t0, 1)}
        w = np.exp((R["lp_lb_nb"].astype(np.float64) - R["lp_nb"]).sum(1))
        meta["quick"] = {"E_w": float(w.mean()), "w_min": float(w.min()), "w_max": float(w.max()),
                         "D_R": float(np.maximum(0, 1 - w).mean()), "c_nb": float(R["c_nb"].mean()),
                         "forced_switch": float((R["c_nb"] - R["c_X"]).mean()), "pX_off": float(R["pX_off"].mean())}
        if a.aa:
            meta["quick"]["aa_max_lp_diff"] = float(np.abs(R["lp_aa"] - R["lp_nb"]).max())
            meta["quick"]["aa_max_c_diff"] = float(np.abs(R["c_aa"] - R["c_nb"]).max())
        extra = {kk: np.concatenate(v) for kk, v in side.items() if v}
        meta["quick"]["n_plain"] = int(len(extra.get("plain_row", [])))
        meta["quick"]["n_cue"] = int(len(extra.get("cue_row", [])))
        op = HERE / a.out
        op.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(op) + ".npz", prompt_idx=rpi[:n], seed=rr[:n], **R, **extra)
        (op.parent / (op.name + ".json")).write_text(json.dumps(meta, indent=1))
        return meta

    nb = (N + B - 1) // B
    for bi, s0 in enumerate(range(0, N, B)):
        j = list(range(s0, min(s0 + B, N)))
        nreal = len(j)
        j = j + [j[-1]] * (B - nreal)
        ids = ids_all[j]
        G = lambda: gens(rpi[j], rr[j])
        oA = S.rollout(ids, "none", gen=G(), ban=X)
        oB = S.rollout(ids, "leaky", forced=oA["toks"], ban=X)
        oC = S.rollout(ids, "none", gen=G(), plant=(X, 1e4, one))
        assert bool((oC["toks"][:, 0] == X).all()) and not bool((oA["toks"][:, 0] == X).any())
        pX = p_first(S, ids, X)
        out = {"toks_nb": oA["toks"], "lp_nb": oA["lp_tok"], "lp_lb_nb": oB["lp_tok"], "c_nb": cheat(oA["d2"]),
               "c_lb_nb": cheat(oB["d2"]), "c_X": cheat(oC["d2"]), "pX_off": pX}
        if a.aa:
            oE = S.rollout(ids, "none", forced=oA["toks"], ban=X)
            out["lp_aa"], out["c_aa"] = oE["lp_tok"], cheat(oE["d2"])
        for kk, v in out.items():
            res[kk].append(v[:nreal].float().cpu().numpy() if v.dtype != torch.long else v[:nreal].cpu().numpy())
        rows_b = np.arange(s0, s0 + nreal)
        cpu = lambda t, m: (t[:nreal].float() if t.dtype != torch.long else t[:nreal]).cpu().numpy()[m]
        pm_ = rr[j[:nreal]] < a.r0 + a.plain_r
        if pm_.any():
            oF = S.rollout(ids, "leaky", gen=G(), ban=X)
            oF2 = S.rollout(ids, "none", forced=oF["toks"], ban=X)
            side["plain_row"].append(rows_b[pm_])
            side["c_lb"].append(cpu(cheat(oF["d2"]), pm_))
            side["toks_lb"].append(cpu(oF["toks"], pm_))
            side["c_nlb"].append(cpu(cheat(oF2["d2"]), pm_))
        cm_ = rr[j[:nreal]] < a.r0 + a.cue_r
        if cm_.any():
            oG = S.rollout(ids, "none", gen=G(), cue=(X, a.cue_delta, one))
            side["cue_row"].append(rows_b[cm_])
            side["c_nc"].append(cpu(cheat(oG["d2"]), cm_))
            side["toks_nc"].append(cpu(oG["toks"], cm_))
        print(f"batch {bi + 1}/{nb} {time.time() - t0:.0f}s", flush=True)
        if (bi + 1) % a.save_every == 0:
            save(False)
        if stop_at and time.time() >= stop_at:
            print(f"STOP clock {a.stop_at} after batch {bi + 1}/{nb} {time.strftime('%H:%M:%S')}", flush=True)
            break
    meta = save(True)
    print(json.dumps(meta, indent=1), flush=True)


if __name__ == "__main__":
    main()
