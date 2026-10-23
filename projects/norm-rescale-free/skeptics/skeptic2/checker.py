#!/usr/bin/env python3
"""Independent skeptic2 checker for theory-1023-norm-rescale-free.

Uses only saved raw arrays:
  data/ref/split.json
  data/fits/panel.npz
  data/runs/<run>/acts_<step>.npy
  data/runs/<run>/ckpt_<step>.npz

It does not import or read the build analysis code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_split(data: Path):
    rows = json.loads((data / "ref" / "split.json").read_text())
    pairs = []
    by_key = {}
    for i, r in enumerate(rows):
        if r["split"] == "eval":
            by_key.setdefault((r["city"], r["src"]), []).append((i, int(r["true"])))
    for key, vals in sorted(by_key.items()):
        if len(vals) != 2:
            raise ValueError(f"bad eval pair {key}: {vals}")
        t = [i for i, truth in vals if truth == 1]
        f = [i for i, truth in vals if truth == 0]
        if len(t) != 1 or len(f) != 1:
            raise ValueError(f"bad true/false pair {key}: {vals}")
        pairs.append((t[0], f[0]))
    true_eval = np.array([t for t, _ in pairs], dtype=np.int64)
    false_eval = np.array([f for _, f in pairs], dtype=np.int64)
    refit_true = np.array([i for i, r in enumerate(rows) if r["split"] == "refit" and r["true"]], dtype=np.int64)
    if len(true_eval) != 596 or len(false_eval) != 596 or len(refit_true) != 200:
        raise ValueError((len(true_eval), len(false_eval), len(refit_true)))
    return rows, true_eval, false_eval, refit_true


def score_params(fit, tok: int):
    w = fit[f"tok{tok}__w"].astype(np.float64)
    sd = fit[f"tok{tok}__sd"].astype(np.float64)
    mu = fit[f"tok{tok}__mu"].astype(np.float64)
    b = float(fit[f"tok{tok}__b"])
    coef = np.divide(w, sd, out=np.zeros_like(w), where=sd != 0)
    with np.errstate(all="ignore"):
        bias = b - float(mu @ coef)
    return coef, bias


def affine(acts2d: np.ndarray, coef: np.ndarray, bias: float):
    with np.errstate(all="ignore"):
        return acts2d.astype(np.float64) @ coef + bias


def score(acts: np.ndarray, tok: int, coef: np.ndarray, bias: float):
    return affine(acts[:, tok, :], coef, bias)


def restore_scores(clean_acts, attacked_acts, idx, tok: int, coef: np.ndarray, bias: float):
    h_clean = clean_acts[idx, tok, :].astype(np.float64)
    h_att = attacked_acts[idx, tok, :].astype(np.float64)
    clean_norm = np.linalg.norm(h_clean, axis=1)
    att_norm = np.linalg.norm(h_att, axis=1)
    restored = h_att * (clean_norm / att_norm)[:, None]
    return affine(restored, coef, bias), clean_norm, att_norm


def catch_rate(scores: np.ndarray, false_idx: np.ndarray, line_idx: np.ndarray, q: float = 0.995):
    line = float(np.quantile(scores[line_idx], q))
    return 100.0 * float(np.mean(scores[false_idx] >= line)), line


def kl_metrics(ckpt, rows):
    kl = ckpt["kl"].astype(np.float64)
    koff = ckpt["koff"]
    cpos = ckpt["cpos"]
    eval_ids = [i for i, r in enumerate(rows) if r["split"] == "eval"]
    vals = []
    cvals = []
    for i in eval_ids:
        vals.append(kl[koff[i]:koff[i + 1]])
        cvals.append(kl[koff[i] + cpos[i]])
    return float(np.concatenate(vals).mean()), float(np.array(cvals).mean())


def bootstrap_r(g_clean_pair, g_att_pair, g_rest_pair, seed=123, draws=4000):
    rng = np.random.default_rng(seed)
    n = len(g_clean_pair)
    out = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        ii = rng.integers(0, n, n)
        gc = float(g_clean_pair[ii].mean())
        ga = float(g_att_pair[ii].mean())
        gr = float(g_rest_pair[ii].mean())
        out[b] = (gr - ga) / (gc - ga)
    return np.quantile(out, [0.025, 0.5, 0.975]).tolist()


def random_norm_null(clean_acts, attacked_acts, idx, tok, coef, bias, clean_scores, att_scores, true_eval, false_eval, seed=456, draws=1000):
    """Null: keep attacked directions but assign clean norms from random eval statements.

    This checks whether norm restoration only looks large because of the marginal norm distribution.
    """
    rng = np.random.default_rng(seed)
    h_clean = clean_acts[idx, tok, :].astype(np.float64)
    h_att = attacked_acts[idx, tok, :].astype(np.float64)
    clean_norms = np.linalg.norm(h_clean, axis=1)
    att_norms = np.linalg.norm(h_att, axis=1)
    g_clean = (clean_scores[false_eval] - clean_scores[true_eval]).mean()
    g_att = (att_scores[false_eval] - att_scores[true_eval]).mean()
    vals = np.empty(draws, dtype=np.float64)
    pos = {j: k for k, j in enumerate(idx.tolist())}
    ti = np.array([pos[i] for i in true_eval])
    fi = np.array([pos[i] for i in false_eval])
    for d in range(draws):
        perm_norms = rng.choice(clean_norms, size=len(idx), replace=False)
        restored = h_att * (perm_norms / att_norms)[:, None]
        rs = affine(restored, coef, bias)
        g_rest = (rs[fi] - rs[ti]).mean()
        vals[d] = (g_rest - g_att) / (g_clean - g_att)
    return np.quantile(vals, [0.025, 0.5, 0.975]).tolist()


def analyze_run(data: Path, run: str, step: int = 300, tok: int = 28):
    rows, true_eval, false_eval, refit_true = load_split(data)
    idx = np.concatenate([true_eval, false_eval])
    fit = np.load(data / "fits" / "panel.npz")
    coef, bias = score_params(fit, tok)
    clean_acts = np.load(data / "runs" / run / "acts_0.npy", mmap_mode="r")
    attacked_acts = np.load(data / "runs" / run / f"acts_{step}.npy", mmap_mode="r")
    clean_scores = score(clean_acts, tok, coef, bias)
    att_scores = score(attacked_acts, tok, coef, bias)
    restored_idx_scores, clean_norm, att_norm = restore_scores(clean_acts, attacked_acts, idx, tok, coef, bias)
    restored_scores = att_scores.copy()
    restored_scores[idx] = restored_idx_scores

    g_clean_pair = clean_scores[false_eval] - clean_scores[true_eval]
    g_att_pair = att_scores[false_eval] - att_scores[true_eval]
    g_rest_pair = restored_scores[false_eval] - restored_scores[true_eval]
    g_clean = float(g_clean_pair.mean())
    g_att = float(g_att_pair.mean())
    g_rest = float(g_rest_pair.mean())
    R = float((g_rest - g_att) / (g_clean - g_att))
    gap_drop_frac = float((g_clean - g_att) / g_clean)
    catch_reset, reset_line = catch_rate(att_scores, false_eval, refit_true, 0.995)
    catch_clean, clean_line = catch_rate(att_scores, false_eval, refit_true, 0.995) if False else (None, None)
    clean_scores_for_line = clean_scores
    clean_line = float(np.quantile(clean_scores_for_line[refit_true], 0.995))
    catch_clean = 100.0 * float(np.mean(att_scores[false_eval] >= clean_line))
    ckpt = np.load(data / "runs" / run / f"ckpt_{step}.npz")
    kl_eval, kl_country = kl_metrics(ckpt, rows)
    return {
        "run": run,
        "step": step,
        "tok": tok,
        "n_pairs": int(len(true_eval)),
        "g_clean": g_clean,
        "g_attacked": g_att,
        "g_restored": g_rest,
        "restore_share_R": R,
        "R_bootstrap_95": bootstrap_r(g_clean_pair, g_att_pair, g_rest_pair),
        "gap_drop_frac": gap_drop_frac,
        "attack_precondition_met": bool(gap_drop_frac >= 0.5),
        "catch_reset_pct": catch_reset,
        "catch_reset_line": reset_line,
        "catch_clean_line_pct": catch_clean,
        "clean_line": clean_line,
        "eval_kl_all_tokens": kl_eval,
        "eval_kl_country_token": kl_country,
        "norm_ratio_false_mean": float((att_norm[len(true_eval):] / clean_norm[len(true_eval):]).mean()),
        "norm_ratio_true_mean": float((att_norm[:len(true_eval)] / clean_norm[:len(true_eval)]).mean()),
        "random_norm_null_R_95": random_norm_null(clean_acts, attacked_acts, idx, tok, coef, bias, clean_scores, att_scores, true_eval, false_eval),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("kran-research/projects/norm-rescale-free/data"))
    ap.add_argument("--out", type=Path, default=Path("swarm-theory-1023-norm-rescale-free/skeptic2/checker_output.json"))
    args = ap.parse_args()
    runs = sorted(p.name for p in (args.data / "runs").iterdir() if p.is_dir() and (p / "acts_300.npy").exists())
    selected = [r for r in runs if r.startswith("r28__lam10000") or r.startswith("r29__lam10000") or "lam0" in r or "lam3e" in r or "lam100000" in r]
    results = []
    for run in selected:
        tok = 28
        if run.startswith("r29__"):
            # Main status for post-norm arm is own tok29 catch, but the causal-null R is on tok28.
            # Store both rows.
            results.append(analyze_run(args.data, run, tok=28))
            own = analyze_run(args.data, run, tok=29)
            own["own_probe_row"] = True
            results.append(own)
        else:
            results.append(analyze_run(args.data, run, tok=28))
    out = {"results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True))
    for r in results:
        tag = " own" if r.get("own_probe_row") else ""
        print(f"{r['run']} tok{r['tok']}{tag}: R={r['restore_share_R']:.4f} CI={r['R_bootstrap_95'][0]:.4f},{r['R_bootstrap_95'][2]:.4f} gapdrop={r['gap_drop_frac']:.3f} catch_reset={r['catch_reset_pct']:.2f}% KL={r['eval_kl_all_tokens']:.4f}")


if __name__ == "__main__":
    main()
