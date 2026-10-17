"""Behaviour-preserving obfuscation attack (PREREG.md, 'Attacker') and the transfer measurement.

Fine-tune all parameters of blocks 0-2 of Pythia-70M with Adam. Loss per step:
    probe_loss  +  lambda_KL * mean per-token KL(p_clean || p_theta)
probe_loss (config attack.loss), s_w = w'h(x), h = mean-pooled hidden_states[3], target TRAIN texts:
  'softplus_pos'  = mean softplus(s_w + b_w) over batch_target target-POSITIVE texts (BCE toward label 0;
                    Bailey et al. 2412.09565 Eq. 1 style). DEVIATIONS.md 15:16.
  'gap_fixed_sd'  = ((mean s_w | 8 pos) - (mean s_w | 8 neg))^2 / sd_clean^2, sd_clean fixed from clean train acts.
  'dprime_batch'  = same with the batch pooled sd (PREREG text; won by inflating spread, DEVIATIONS.md 15:13).
KL on 16 retain-train windows (pile-10k odd docs, disjoint from the Fisher windows).
Every `check_every` steps: target AUROC on its TEST texts; stop at the first check < stop_auroc, or the cap.
After the stop: attacked activations of every test text of every source -> scores of all probes,
AUROC and class gaps per probe, delta_w, and the retain KL on 500 held-out windows.

Usage:
  uv run python attack.py pilot                       # lambda_KL pilot (grid largest-first)
  uv run python attack.py run --target NAME --seed S  # one attack run (lambda from data/runs/pilot.json)
"""
import argparse
import json
import resource
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from common import DATA, ROOT, batches, device, load_model, pool, pooled_acts, tokenize

CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
A = CFG["attack"]
RUNS = DATA / "runs"


def probe_table():
    meta = json.loads((DATA / "probes" / "probes.json").read_text())
    Wt = np.load(DATA / "probes" / "probe_weights.npz")
    return meta, Wt["W"]


def source_data(tok, src):
    z = np.load(DATA / "clean" / f"acts_{src}.npz")
    texts = json.loads((DATA / "clean" / f"texts_{src}.json").read_text())
    return {"ids": tokenize(tok, texts), "label": z["label"], "split": z["split"], "X": z["X"]}


def kl_clean_new(ref, model, inp):
    with torch.no_grad():
        lp_ref = torch.log_softmax(ref(input_ids=inp).logits, -1)
    lp = torch.log_softmax(model(input_ids=inp).logits, -1)
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1).mean()


def h3(model, inp, mask):
    out = model.gpt_neox(input_ids=inp, attention_mask=mask.long(), output_hidden_states=True)
    return pool(out.hidden_states[3], mask)


def pad(seqs, eos, dev):
    L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L), eos, dtype=torch.long)
    mask = torch.zeros((len(seqs), L))
    for r, s in enumerate(seqs):
        inp[r, :len(s)] = torch.tensor(s)
        mask[r, :len(s)] = 1
    return inp.to(dev), mask.to(dev)


def attack(target, seed, lam, measure=True):
    t0 = time.time()
    dev = device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, model = load_model(dev)
    _, ref = load_model(dev)
    for p in ref.parameters():
        p.requires_grad_(False)
    train_params = []
    for n, p in model.named_parameters():
        keep = any(n.startswith(f"gpt_neox.layers.{i}.") for i in (0, 1, 2))
        p.requires_grad_(keep)
        if keep:
            train_params.append(p)
    opt = torch.optim.Adam(train_params, lr=A["lr"])
    meta, W = probe_table()
    t = next(p for p in meta if p["name"] == target)
    w = torch.tensor(W[t["idx"]], dtype=torch.float32, device=dev)
    b_w = float(np.load(DATA / "probes" / "probe_weights.npz")["b"][t["idx"]])
    sd = source_data(tok, t["source"])
    y = (sd["label"] == t["label"]).astype(int)
    tr_pos = np.where((sd["split"] == 0) & (y == 1))[0]
    tr_neg = np.where((sd["split"] == 0) & (y == 0))[0]
    te = np.where(sd["split"] == 1)[0]
    te_ids = [sd["ids"][i] for i in te]
    win = np.load(DATA / "fisher" / "windows.npz")
    retain = torch.tensor(win["retain_ids"])
    eos = tok.eos_token_id
    half = A["batch_target"] // 2
    s_tr = sd["X"][sd["split"] == 0].astype(np.float64) @ W[t["idx"]]
    y_tr = y[sd["split"] == 0] == 1
    sd_clean = float(np.sqrt(0.5 * (s_tr[y_tr].var(ddof=1) + s_tr[~y_tr].var(ddof=1))))

    def target_auc():
        X = pooled_acts(model, te_ids, eos, dev)
        return float(roc_auc_score(y[te], X @ W[t["idx"]])), X

    traj = []
    auc0, _ = target_auc()
    stop_step, success = None, False
    for step in range(1, A["step_cap"] + 1):
        if A["loss"] == "softplus_pos":
            bi = rng.choice(tr_pos, A["batch_target"], replace=len(tr_pos) < A["batch_target"])
            inp, mask = pad([sd["ids"][i] for i in bi], eos, dev)
            s = h3(model, inp, mask) @ w + b_w
            probe_loss = torch.nn.functional.softplus(s).mean()
        else:
            bi = np.concatenate([rng.choice(tr_pos, half, replace=len(tr_pos) < half),
                                 rng.choice(tr_neg, half, replace=False)])
            inp, mask = pad([sd["ids"][i] for i in bi], eos, dev)
            s = h3(model, inp, mask) @ w
            sp, sn = s[:half], s[half:]
            if A["loss"] == "dprime_batch":
                probe_loss = ((sp.mean() - sn.mean()) / torch.sqrt(0.5 * (sp.var() + sn.var()))) ** 2
            else:  # gap_fixed_sd
                probe_loss = ((sp.mean() - sn.mean()) / sd_clean) ** 2
        ri = torch.tensor(rng.choice(len(retain), A["batch_retain"], replace=False))
        kl = kl_clean_new(ref, model, retain[ri].to(dev))
        loss = probe_loss + lam * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % A["check_every"] == 0:
            auc, _ = target_auc()
            traj.append({"step": step, "auroc": auc, "probe_loss_batch": float(probe_loss.detach()), "kl_batch": float(kl.detach())})
            if auc < A["stop_auroc"]:
                stop_step, success = step, True
                break
    rec = {"target": target, "seed": seed, "lambda_kl": lam, "success": success, "loss": A["loss"],
           "sd_clean_train": sd_clean,
           "stop_step": stop_step if success else A["step_cap"], "target_auroc_clean": auc0,
           "trajectory": traj, "device": str(dev)}
    if measure:
        rec.update(measure_all(model, ref, tok, meta, W, t, win, dev))
    rec["seconds"] = time.time() - t0
    rec["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return rec


@torch.no_grad()
def measure_all(model, ref, tok, meta, W, t, win, dev):
    eos = tok.eos_token_id
    srcs = sorted({p["source"] for p in meta})
    per_probe, scores, edit, acts = [], {}, {}, {}
    for src in srcs:
        sd = source_data(tok, src)
        te = np.where(sd["split"] == 1)[0]
        Xc = sd["X"][te].astype(np.float64)
        Xa = pooled_acts(model, [sd["ids"][i] for i in te], eos, dev).astype(np.float64)
        scores[src] = {"clean": Xc @ W.T, "attacked": Xa @ W.T}
        acts[f"attacked_acts__{src}"] = Xa.astype(np.float16)
        if src == t["source"]:
            yt = (sd["label"][te] == t["label"])
            dh = Xa - Xc
            edit = {"mean_edit_pos": dh[yt].mean(0), "mean_edit_neg": dh[~yt].mean(0), "mean_edit_all": dh.mean(0)}
            sec = json.loads((DATA / "probes" / "secondary.json").read_text())
            W2 = np.load(DATA / "probes" / "secondary_weights.npz")["W"]
            for r in sec:
                if r["target"] != t["name"] or not r["kept"]:
                    continue
                yu = yt.astype(int)
                per_probe.append(stats(r["name"], "secondary", yu, Xc @ W2[r["idx"]], Xa @ W2[r["idx"]]))
        for p in meta:
            if p["source"] != src:
                continue
            yu = (sd["label"][te] == p["label"]).astype(int)
            per_probe.append(stats(p["name"], "primary", yu, Xc @ W[p["idx"]], Xa @ W[p["idx"]]))
    tp = next(r for r in per_probe if r["probe"] == t["name"] and r["set"] == "primary")
    kls = []
    held = torch.tensor(win["heldout_ids"])
    for s in range(0, len(held), 16):
        kls.append(float(kl_clean_new(ref, model, held[s:s + 16].to(dev))))
    return {"delta_w": tp["gap_attacked"] - tp["gap_clean"], "G_w": tp["gap_clean"],
            "target_auroc_attacked": tp["auroc_attacked"], "retain_kl_heldout": float(np.mean(kls)),
            "probes": per_probe, "_scores": scores, "_edit": edit | acts}


def stats(name, set_, y, sc, sa):
    """AUROC, class means and pooled within-class sd of a probe score, clean and attacked (probe's test set)."""
    r = {"probe": name, "set": set_}
    for tag, s in (("clean", sc), ("attacked", sa)):
        r[f"auroc_{tag}"] = float(roc_auc_score(y, s))
        r[f"mean_pos_{tag}"] = float(s[y == 1].mean())
        r[f"mean_neg_{tag}"] = float(s[y == 0].mean())
        r[f"gap_{tag}"] = r[f"mean_pos_{tag}"] - r[f"mean_neg_{tag}"]
        r[f"sd_pooled_{tag}"] = float(np.sqrt(0.5 * (s[y == 1].var(ddof=1) + s[y == 0].var(ddof=1))))
    return r


def save(rec, stem):
    RUNS.mkdir(parents=True, exist_ok=True)
    scores, edit = rec.pop("_scores", None), rec.pop("_edit", None)
    (RUNS / f"{stem}.json").write_text(json.dumps(rec, indent=1))
    if scores is not None:
        arrs = {f"{src}__{k}": v.astype(np.float32) for src, d in scores.items() for k, v in d.items()}
        arrs.update({k: v for k, v in edit.items() if not k.startswith("attacked_acts__")})
        np.savez_compressed(RUNS / f"{stem}.npz", **arrs)
        # float16 attacked activations of every test set: large, regenerable, not committed (.gitignore)
        np.savez_compressed(RUNS / f"{stem}__acts.npz", **{k: v for k, v in edit.items() if k.startswith("attacked_acts__")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "run"])
    ap.add_argument("--target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=None, help="override lambda_KL (lambda=0 diagnostic, DEVIATIONS 15:13)")
    a = ap.parse_args()
    sel = json.loads((DATA / "selection.json").read_text())
    if a.mode == "pilot":
        grid = A["lambda_kl_grid"]
        log, ok = [], {}

        def trial(i):
            rec = attack(sel["pilot"], 0, grid[i], measure=False)
            log.append({k: rec[k] for k in ("lambda_kl", "success", "stop_step", "target_auroc_clean", "seconds", "max_rss_mb")}
                       | {"final_auroc": rec["trajectory"][-1]["auroc"] if rec["trajectory"] else None})
            print(json.dumps(log[-1]), flush=True)
            ok[i] = rec["success"]
            return rec["success"]

        i = grid.index(A["lambda_kl_start"])
        if trial(i):
            while i + 1 < len(grid) and trial(i + 1):
                i += 1
            chosen = grid[i] if (i + 1 < len(grid)) else None  # None = no failure found: unbracketed
        else:
            while i > 0 and not trial(i - 1):
                i -= 1
            chosen = grid[i - 1] if i > 0 else None
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "pilot.json").write_text(json.dumps({"pilot_probe": sel["pilot"], "grid": grid, "loss": A["loss"],
                                                     "rule": "DEVIATIONS 15:12: start at lambda_kl_start; on success step up the ascending grid until the first failure, on failure step down until the first success; chosen = largest success adjacent to a failure (bracketed). chosen null = unbracketed.",
                                                     "chosen_lambda_kl": chosen, "log": log}, indent=1))
    else:
        lam = json.loads((RUNS / "pilot.json").read_text())["chosen_lambda_kl"] if a.lam is None else a.lam
        base = a.target.replace(':', '__').replace(' ', '_').replace('&', 'and')
        stem = f"{base}__seed{a.seed}" if a.lam is None else f"{base}__lam{a.lam:g}__seed{a.seed}"
        rec = attack(a.target, a.seed, lam)
        save(rec, stem)
        print(json.dumps({k: rec[k] for k in ("target", "seed", "success", "stop_step", "target_auroc_clean",
                                              "target_auroc_attacked", "delta_w", "G_w", "retain_kl_heldout",
                                              "seconds", "max_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
