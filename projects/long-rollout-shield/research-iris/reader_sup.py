"""C_sup = the reader edge: how big a probe-off switch can the policy's own decision slot learn from the thought TEXT
when it is given the labels (supervised), on prompts it never saw? Owner Iris; spec #17566.

Reader = the start policy (Qwen3-0.6B + its LoRA on blocks 26-27, the same weights the RL update trains), input =
prompt + 64-token thought + '\\nAnswer:', the 2-way decision (' read' vs ' solve') at the answer slot. Supervised
target: ' read' (cheat) on PROBE-OFF text, ' solve' on PROBE-ON text: exactly the switch the RL arm would need.
Blocks 0-25 run as a clean eager recompute from the token ids (no hook, a function of the text only, mt.h26_clean);
only the LoRA on blocks 26-27 trains. Data: a read.py npz (CRN pairs toks_n = probe-off draw, toks_l = probe-on draw,
same prompt and noise). Identical pairs carry no label information (same text -> same q), so training uses pairs
whose text differs, and the score counts identical pairs as exactly 0.

C_sup = mean over ALL held-out pairs of q(off text) - q(on text), q = P(read) / (P(read) + P(solve)), in points.
95% CI by bootstrap over held-out prompts (2,000, seed 0). Epoch 0 = the start policy before reader training.
Split by prompt: --mode split (train on 'train' prompts, score on 'test') or --mode crossfit (2 folds over the
prompts of the npz, each fold scored by the reader trained on the other; the two folds are pooled).
--null swap: each training pair's two labels are swapped with probability 1/2 (seeded); a reader then has nothing
to learn, so C_sup should be ~0.
Frozen settings: 3 epochs, AdamW lr 3e-4, wd 0, 8 pairs (16 texts) per step, grad clip 1.0, seed 0.
DEVIATION 17:35 PST (before any real fit): at least 300 updates (epochs = max(3, ceil(300 / steps per epoch))), C_sup =
the final epoch. Why: the self-test (10% one-token cue in the parent's 400 k64 rows) trained ~21 differing pairs per
fold = 9 updates in 3 epochs and learned nothing (C_sup -0.32 pts); the real arm has ~580 train pairs (219 updates
in 3 epochs), the one-token arm ~150 (57 updates). Same rule for every arm.

  PY=../../../env/.venv/bin/python
  $PY research-iris/reader_sup.py --npz data/reads/raw/start.npz --policy data/runs/pilot/adapter.pt \
      --mode split --out research-iris/csup_start
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mt  # noqa: E402


def q_of(inner, ids, toks):
    h = mt.h26_clean(inner, mt.full_ids(ids, toks))
    _, d2 = mt.tail_logits(inner, h, toks.shape[1])
    return d2  # [B, 2] logits (read, solve), fp32


@torch.no_grad()
def score(inner, ids_of, P, toks_n, toks_l, bs):
    """q(off), q(on) for each pair in P (indices into the npz rows)."""
    qn, ql = [], []
    for j in range(0, len(P), bs):
        b = P[j:j + bs]
        ids = ids_of(b)
        for toks, acc in ((toks_n, qn), (toks_l, ql)):
            d2 = q_of(inner, ids, torch.tensor(toks[b], device=mt.DEV))
            acc.append(torch.softmax(d2, -1)[:, 0].cpu().numpy())
    return np.concatenate(qn), np.concatenate(ql)


def boot(vals, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: vals[groups == g] for g in ug}
    out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        v = np.concatenate([by[g] for g in pick])
        out.append(v.mean())
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def run_fold(a, d, tr_rows, te_rows, ids_of, fold_tag, log):
    torch.manual_seed(a.seed)
    pm, inner = mt.load_policy(HERE.parent / a.policy if not os.path.isabs(a.policy) else a.policy)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    tn, tl = d["toks_n"], d["toks_l"]
    diff = (tn != tl).any(1)
    tr = tr_rows[diff[tr_rows]]
    if a.frac < 1.0:
        tr = np.random.default_rng(a.seed + 7).permutation(tr)[: int(round(len(tr) * a.frac))]
    rng = np.random.default_rng(a.seed + 1)
    swap = rng.random(len(tr)) < 0.5 if a.null == "swap" else np.zeros(len(tr), bool)
    te_diff = te_rows[diff[te_rows]]
    curve = []

    def evaluate(ep):
        pm.eval()
        qn, ql = score(inner, ids_of, te_diff, tn, tl, a.bs)
        g = np.zeros(len(te_rows))
        pos = {r: i for i, r in enumerate(te_rows)}
        for r, x in zip(te_diff, qn - ql):
            g[pos[r]] = x
        row = dict(fold=fold_tag, epoch=ep, gap_all=float(g.mean()), n_pairs=int(len(te_rows)),
                   n_diff=int(len(te_diff)), gap_diff=float((qn - ql).mean()) if len(te_diff) else 0.0,
                   q_off_mean=float(qn.mean()), q_on_mean=float(ql.mean()),
                   acc_diff=float(((qn > 0.5).mean() + (ql <= 0.5).mean()) / 2) if len(te_diff) else 0.5)
        curve.append(row)
        log(json.dumps(row))
        return g

    g = evaluate(0)
    gs = {0: g}
    steps = 0
    spe = max(1, -(-len(tr) // a.pairs))                 # steps per epoch
    E = max(a.epochs, -(-a.min_steps // spe))            # DEVIATION (Iris, before any real fit): >= min_steps updates
    log(f"fold {fold_tag}: {len(tr)} train pairs, {spe} steps/epoch, {E} epochs")
    for ep in range(1, E + 1):
        pm.train()
        order = np.random.default_rng(a.seed + 100 + ep).permutation(len(tr))
        for j in range(0, len(order), a.pairs):
            o = order[j:j + a.pairs]
            b = tr[o]
            sw = swap[o]
            ids = ids_of(b)
            off = np.where(sw[:, None], tl[b], tn[b])   # text labelled 'probe off' (target read)
            on = np.where(sw[:, None], tn[b], tl[b])    # text labelled 'probe on' (target solve)
            toks = torch.tensor(np.concatenate([off, on]), device=mt.DEV)
            d2 = q_of(inner, torch.cat([ids, ids]), toks)
            y = torch.tensor([0] * len(b) + [1] * len(b), device=mt.DEV)
            loss = torch.nn.functional.cross_entropy(d2, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            steps += 1
            if steps % 25 == 0:
                log(f"fold {fold_tag} ep {ep} step {steps} loss {loss.item():.4f}")
        if ep == min(3, E) or ep == E:      # held-out scores at epoch 0, 3 and the final epoch only (lane time)
            gs[ep] = evaluate(ep)
    gs["final"] = gs[E]
    del pm, inner, opt
    torch.mps.empty_cache()
    return gs, curve, dict(n_train_pairs=int(len(tr)), n_train_rows=int(len(tr_rows)), epochs=int(E), steps=int(steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--policy", default="data/runs/pilot/adapter.pt")
    ap.add_argument("--mode", default="split", choices=["split", "crossfit"])
    ap.add_argument("--null", default="none", choices=["none", "swap"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--min-steps", type=int, default=300, help="train at least this many updates (more epochs on "
                    "small train sets); deviation after the self-test, before any real fit")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16, help="eval batch (pairs)")
    ap.add_argument("--frac", type=float, default=1.0, help="learning curve: share of training pairs used")
    ap.add_argument("--n", type=int, default=-1, help="smoke: first n rows only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    logf = open(str(out_p) + ".log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    parts = []
    for one in a.npz.split(","):
        npz = Path(one) if os.path.isabs(one) else HERE.parent / one
        raw = np.load(npz)
        parts.append({k: raw[k] for k in raw.files})
    keys = ["prompt_idx", "split", "toks_n", "toks_l"]   # inputs only; several reads (e.g. test R=8 + train R=4) concat
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if a.n > 0:
        d = {k: v[: a.n] for k, v in d.items()}
    pid, split = d["prompt_idx"], d["split"]
    idmap = {int(r["id"][2:]): r["ids"] for r in mt.prompts()}
    P_ids = torch.tensor(np.array([idmap[int(p)] for p in pid]), device=mt.DEV)

    def ids_of(rows):
        return P_ids[torch.tensor(rows, device=mt.DEV)]

    rows = np.arange(len(pid))
    folds = []
    if a.mode == "split":
        folds.append((rows[split == "train"], rows[split == "test"], "split"))
    else:
        up = np.unique(pid)
        perm = np.random.default_rng(a.seed + 11).permutation(up)
        half = set(perm[: len(up) // 2].tolist())
        inA = np.array([p in half for p in pid])
        folds += [(rows[~inA], rows[inA], "A"), (rows[inA], rows[~inA], "B")]
    pooled, curves, meta = {}, [], []
    for tr_rows, te_rows, tag in folds:
        gs, curve, m = run_fold(a, d, tr_rows, te_rows, ids_of, tag, log)
        curves += curve
        meta.append(dict(fold=tag, **m, n_test_pairs=int(len(te_rows))))
        for ep, g in gs.items():
            pooled.setdefault(ep, []).append((te_rows, g))
    res = dict(npz=str(a.npz), policy=a.policy, mode=a.mode, null=a.null, epochs=a.epochs, min_steps=a.min_steps,
               lr=a.lr, pairs=a.pairs, frac=a.frac, seed=a.seed, k=int(d["toks_n"].shape[1]), folds=meta, curve=curves)
    for ep, parts in pooled.items():
        if len(parts) < len(folds):          # crossfit folds may run different epoch counts: pool shared ones only
            continue
        te = np.concatenate([p[0] for p in parts])
        g = np.concatenate([p[1] for p in parts])
        res[f"C_sup_ep{ep}"] = float(g.mean())
        res[f"C_sup_ep{ep}_ci"] = boot(g, pid[te])
    res["C_sup"] = res["C_sup_epfinal"]
    res["C_sup_ci"] = res["C_sup_epfinal_ci"]
    res["f_test"] = float((d["toks_n"] != d["toks_l"]).any(1)[np.concatenate([f[1] for f in folds])].mean())
    res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(str(out_p) + ".json", "w"), indent=1)
    log(f"C_sup {res['C_sup'] * 100:.2f} pts CI [{res['C_sup_ci'][0] * 100:.2f}, {res['C_sup_ci'][1] * 100:.2f}] "
        f"(epoch 0: {res['C_sup_ep0'] * 100:.2f}); {res['secs']} s")


if __name__ == "__main__":
    main()
