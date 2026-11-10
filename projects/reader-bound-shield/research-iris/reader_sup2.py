"""x = C_sup, the labelled same-size reader edge, under FROZEN READER RULE v1 (Iris, room #18605, posted 2026-11-09
17:00 PST before any fit). Owner Iris. Parent: long-rollout-shield/research-iris/reader_sup.py (same reader class).

Reader = the policy's own decision slot: Qwen3-0.6B + the start LoRA (r16, blocks 26-27, the weights the RL update
trains), input = prompt + 64-token thought + '\\nAnswer:', q = P(read) / (P(read) + P(solve)). Blocks 0-25 run as a
clean eager recompute from the token ids (mt.h26_clean): q is a function of the text only. Labels: ' read' on the
PROBE-OFF text, ' solve' on the PROBE-ON text of each CRN pair (the switch the RL arm would need). CE loss, AdamW
lr 3e-4 (= the RL lr), wd 0, 8 pairs (16 texts) per update, grad clip 1.0, seed 0. Only pairs whose texts differ
carry a label (identical texts give identical q), so training uses differing pairs only.

Splits (by prompt): TEST = the 'test' npz (never used to stop). VAL = 20% of the TRAIN prompts (rng seed+21), held
out of training; FIT = the other 80%.

Stop rule (frozen): evaluate VAL every 50 updates (VAL edge = mean over ALL val pairs of q(off) - q(on); val f = share
of val pairs whose texts differ = the ceiling of any reader on VAL). Never stop before 300 updates. Stop at the first
checkpoint where
  (i)   VAL edge >= 0.9 x val f                                  -> 'saturated'
  (ii)  mean train loss over the last 50 updates < 0.10 AND VAL edge rose < 0.5 points over the last 2 checks
                                                                  -> 'memorised'
  (iii) 600 updates                                               -> 'capped'
x = TEST C_sup at the stop checkpoint = mean over ALL test pairs of q(off) - q(on), points; 95% CI by bootstrap over
test prompts (2,000, seed 0). Same unit as the switch (cheat(off) - cheat(on)). TEST is also scored at update 0.
A 'capped' x near 0 is a lower bound on what this class can read, not a certificate.

--dec-bias sets mt.DEC['bias'] (the engine's fixed offset on the read logit) so the reader starts where the RL policy
starts; a constant offset that training can absorb. --null swap: each FIT pair's labels swap with p 1/2 (seeded).

Outputs: <out>.json (x, CI, stop reason, curve), <out>.log, <out>_pairs.npz (per-pair TEST q_off/q_on at the stop
checkpoint and at update 0, fit/val/test row ids; rows index the concatenated input npz(s)).

  PY=../../../env/.venv/bin/python
  $PY research-iris/reader_sup2.py --npz data/parent/start_real_train.npz,data/parent/start_real_test.npz \
      --policy data/runs/pilot/adapter.pt --out research-iris/x_real_start
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mt  # noqa: E402

EVAL_EVERY, MIN_STEPS, MAX_STEPS = 50, 300, 600
SAT_FRAC, MEM_LOSS, MEM_RISE = 0.9, 0.10, 0.005


def q_of(inner, ids, toks):
    h = mt.h26_clean(inner, mt.full_ids(ids, toks))
    _, d2 = mt.tail_logits(inner, h, toks.shape[1])
    return d2  # [B, 2] logits (read, solve), fp32


@torch.no_grad()
def score(inner, ids_of, P, toks_n, toks_l, bs):
    qn, ql = [], []
    for j in range(0, len(P), bs):
        b = P[j:j + bs]
        ids = ids_of(b)
        for toks, acc in ((toks_n, qn), (toks_l, ql)):
            d2 = q_of(inner, ids, torch.tensor(toks[b], device=mt.DEV))
            acc.append(torch.softmax(d2, -1)[:, 0].cpu().numpy())
    if not qn:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(qn), np.concatenate(ql)


def boot(vals, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: vals[groups == g] for g in ug}
    out = [np.concatenate([by[g] for g in rng.choice(ug, len(ug))]).mean() for _ in range(n)]
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def edge(inner, ids_of, rows, diff, tn, tl, bs):
    """mean over ALL rows of q(off) - q(on); identical pairs count 0. Returns (edge, per-row gaps, q_off, q_on)."""
    dr = rows[diff[rows]]
    qn, ql = score(inner, ids_of, dr, tn, tl, bs)
    g = np.zeros(len(rows))
    pos = {r: i for i, r in enumerate(rows)}
    for r, x in zip(dr, qn - ql):
        g[pos[r]] = x
    return float(g.mean()), g, qn, ql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="comma list; rows with split=='train' train, 'test' score")
    ap.add_argument("--policy", default="data/runs/pilot/adapter.pt")
    ap.add_argument("--null", default="none", choices=["none", "swap"])
    ap.add_argument("--dec-bias", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n", type=int, default=-1, help="smoke: first n rows of each split")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS, help="smoke only; frozen value 600")
    ap.add_argument("--min-steps", type=int, default=MIN_STEPS, help="smoke only; frozen value 300")
    ap.add_argument("--frac", type=float, default=1.0, help="learning curve: share of FIT differing pairs (Kasper #18609)")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="smoke only; frozen value 50")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    mt.DEC["bias"] = a.dec_bias
    out_p = Path(a.out) if os.path.isabs(a.out) else HERE.parent / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    logf = open(str(out_p) + ".log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    parts = []
    for one in a.npz.split(","):
        raw = np.load(Path(one) if os.path.isabs(one) else HERE.parent / one)
        parts.append({k: raw[k] for k in ("prompt_idx", "split", "toks_n", "toks_l")})
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    pid, split, tn, tl = d["prompt_idx"], d["split"], d["toks_n"], d["toks_l"]
    rows = np.arange(len(pid))
    tr_all, te = rows[split == "train"], rows[split == "test"]
    if a.n > 0:
        tr_all, te = tr_all[: a.n], te[: a.n]
    up = np.unique(pid[tr_all])
    val_p = set(np.random.default_rng(a.seed + 21).permutation(up)[: int(round(0.2 * len(up)))].tolist())
    inval = np.array([p in val_p for p in pid[tr_all]])
    va, fit = tr_all[inval], tr_all[~inval]
    diff = (tn != tl).any(1)
    fit_d = fit[diff[fit]]
    if a.frac < 1.0:
        fit_d = np.sort(np.random.default_rng(a.seed + 7).permutation(fit_d)[: int(round(len(fit_d) * a.frac))])
    f_val, f_test = float(diff[va].mean()), float(diff[te].mean())
    log(f"rows: fit {len(fit)} ({len(fit_d)} differing), val {len(va)} (f {f_val:.4f}), test {len(te)} (f {f_test:.4f})")

    idmap = {int(r["id"][2:]): r["ids"] for r in mt.prompts()}
    P_ids = torch.tensor(np.array([idmap[int(p)] for p in pid]), device=mt.DEV)

    def ids_of(r):
        return P_ids[torch.tensor(r, device=mt.DEV)]

    torch.manual_seed(a.seed)
    pm, inner = mt.load_policy(HERE.parent / a.policy if not os.path.isabs(a.policy) else a.policy)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    swap = (np.random.default_rng(a.seed + 1).random(len(fit_d)) < 0.5) if a.null == "swap" else np.zeros(len(fit_d), bool)

    pm.eval()
    x0, g0, qn0, ql0 = edge(inner, ids_of, te, diff, tn, tl, a.bs)
    v0 = edge(inner, ids_of, va, diff, tn, tl, a.bs)[0]
    curve = [dict(step=0, val_edge=v0, test_edge=x0, loss50=None)]
    log(json.dumps(curve[-1]))

    losses, steps, ep, reason = [], 0, 0, None
    while reason is None:
        ep += 1
        order = np.random.default_rng(a.seed + 100 + ep).permutation(len(fit_d))
        for j in range(0, len(order), a.pairs):
            pm.train()
            o = order[j:j + a.pairs]
            b, sw = fit_d[o], swap[o]
            off = np.where(sw[:, None], tl[b], tn[b])
            on = np.where(sw[:, None], tn[b], tl[b])
            toks = torch.tensor(np.concatenate([off, on]), device=mt.DEV)
            ids = ids_of(b)
            d2 = q_of(inner, torch.cat([ids, ids]), toks)
            y = torch.tensor([0] * len(b) + [1] * len(b), device=mt.DEV)
            loss = torch.nn.functional.cross_entropy(d2, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            steps += 1
            losses.append(float(loss.item()))
            if steps % a.eval_every:
                continue
            pm.eval()
            v = edge(inner, ids_of, va, diff, tn, tl, a.bs)[0]
            l50 = float(np.mean(losses[-a.eval_every:]))
            curve.append(dict(step=steps, val_edge=v, loss50=l50, epoch=ep))
            log(json.dumps(curve[-1]))
            if steps >= a.min_steps:
                vals = [c["val_edge"] for c in curve]
                rise = vals[-1] - vals[-3] if len(vals) >= 3 else float("inf")
                if v >= SAT_FRAC * f_val:
                    reason = "saturated"
                elif l50 < MEM_LOSS and rise < MEM_RISE:
                    reason = "memorised"
                elif steps >= a.max_steps:
                    reason = "capped"
            if reason:
                break

    pm.eval()
    x, g, qn, ql = edge(inner, ids_of, te, diff, tn, tl, a.bs)
    res = dict(rule="FROZEN READER RULE v1 (#18605)", npz=a.npz, policy=a.policy, null=a.null, dec_bias=a.dec_bias,
               lr=a.lr, pairs=a.pairs, seed=a.seed, frac=a.frac, k=int(tn.shape[1]), stop=reason, steps=steps,
               labelled_pairs_seen=steps * a.pairs, n_fit_pairs=int(len(fit_d)), n_val=int(len(va)),
               n_test=int(len(te)), f_val=f_val, f_test=f_test, loss50_at_stop=curve[-1]["loss50"],
               x=x, x_ci=boot(g, pid[te]), x_step0=x0, x_step0_ci=boot(g0, pid[te]), curve=curve,
               secs=round(time.time() - t0, 1))
    np.savez_compressed(str(out_p) + "_pairs.npz", prompt_idx=pid, split=split, sources=np.array(a.npz.split(",")),
                        test_rows=te, test_diff_rows=te[diff[te]], q_off=qn, q_on=ql, q_off_step0=qn0, q_on_step0=ql0,
                        fit_rows=fit, fit_diff_rows=fit_d, fit_swapped=swap, val_rows=va)
    json.dump(res, open(str(out_p) + ".json", "w"), indent=1)
    log(f"x = {x * 100:.2f} pts CI [{res['x_ci'][0] * 100:.2f}, {res['x_ci'][1] * 100:.2f}] (step 0: {x0 * 100:.2f}); "
        f"stop {reason} at {steps} updates, loss50 {res['loss50_at_stop']:.4f}, f_test {f_test:.4f}; {res['secs']} s")


if __name__ == "__main__":
    main()
