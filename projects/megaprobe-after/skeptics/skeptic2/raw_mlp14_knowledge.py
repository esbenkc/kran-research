#!/usr/bin/env python3
"""Hugo raw-data rebuild: MLP14 and knowledge logits.

Standalone. Does not import project code. Reads only raw memmaps, split.json,
and the public tokenizer. Label convention here: true=1 in split.json, false=0.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_split(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def token_mean(acts_dir: Path, state: str, layer: int, indices: np.ndarray) -> np.ndarray:
    d = acts_dir / state
    tok = np.load(d / "tok.npy", mmap_mode="r")
    off = np.load(d / "off.npy", mmap_mode="r")
    out = np.empty((len(indices), tok.shape[2]), dtype=np.float32)
    for j, i in enumerate(indices):
        a, b = int(off[i]), int(off[i + 1])
        out[j] = np.asarray(tok[a:b, layer, :], dtype=np.float32).mean(axis=0)
    return out


def threshold_from_true(scores: np.ndarray) -> float:
    return float(np.sort(np.asarray(scores, dtype=float))[-2])


def catch_far(eval_scores: np.ndarray, eval_true: np.ndarray, line_true_scores: np.ndarray) -> Tuple[float, float, float]:
    t = threshold_from_true(line_true_scores)
    flags = eval_scores > t
    # false statements are true==0; catch means flag false.
    catch = float(flags[eval_true == 0].mean())
    far = float(flags[eval_true == 1].mean())
    return catch, far, t


def city_bootstrap_catch(eval_scores_by_seed: Dict[str, np.ndarray], eval_true: np.ndarray, eval_city: np.ndarray,
                         line_true_by_seed: Dict[str, np.ndarray], seed: int = 1020, n_boot: int = 2000) -> Tuple[float, float, float]:
    cities = np.array(sorted(set(eval_city)))
    city_to_idx = {c: np.where(eval_city == c)[0] for c in cities}
    catches = []
    for st, scores in eval_scores_by_seed.items():
        t = threshold_from_true(line_true_by_seed[st])
        mask = eval_true == 0
        catches.append((scores[mask] > t).astype(float))
    point = float(np.concatenate(catches).mean())
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        vals = []
        draw = rng.choice(cities, size=len(cities), replace=True)
        for st, scores in eval_scores_by_seed.items():
            t = threshold_from_true(line_true_by_seed[st])
            for c in draw:
                idx = city_to_idx[c]
                idx = idx[eval_true[idx] == 0]
                vals.extend((scores[idx] > t).astype(float).tolist())
        reps[b] = float(np.mean(vals))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def cv_select_alpha(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas=(1e-4, 1e-3, 1e-2)) -> Tuple[float, Dict[str, float]]:
    gkf = GroupKFold(n_splits=4)
    aucs: Dict[float, float] = {}
    for alpha in alphas:
        fold_scores = np.zeros(len(y), dtype=float)
        for tr, va in gkf.split(X, y, groups):
            clf = make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(256,), activation="relu", solver="adam", alpha=alpha,
                              learning_rate_init=1e-3, batch_size=64, max_iter=200, random_state=1020,
                              early_stopping=True, validation_fraction=0.2, n_iter_no_change=20,
                              shuffle=True, verbose=False),
            )
            clf.fit(X[tr], y[tr])
            fold_scores[va] = clf.predict_proba(X[va])[:, 1]
        aucs[alpha] = float(roc_auc_score(y, fold_scores))
    best_auc = max(aucs.values())
    # PREREG: ties within 0.002 -> larger weight decay.
    candidates = [a for a, v in aucs.items() if best_auc - v <= 0.002]
    best = max(candidates)
    return best, {str(k): v for k, v in aucs.items()}


def fit_mlp14(project: Path, out: Path) -> Dict:
    split = load_split(project / "data" / "ref1019" / "split.json")
    acts = project / "data" / "acts"
    train_idx = np.array([i for i, r in enumerate(split) if r["split"] in {"probe", "attack", "stop"}], dtype=int)
    line_idx = np.array([i for i, r in enumerate(split) if r["split"] == "refit" and int(r["true"]) == 1], dtype=int)
    eval_idx = np.array([i for i, r in enumerate(split) if r["split"] == "eval"], dtype=int)
    y_train = np.array([1 - int(split[i]["true"]) for i in train_idx], dtype=int)  # false=1 for classifier high score
    train_groups = np.array([split[i]["city"] for i in train_idx])
    eval_true = np.array([int(split[i]["true"]) for i in eval_idx], dtype=int)
    eval_city = np.array([split[i]["city"] for i in eval_idx])

    X_train = token_mean(acts, "clean", 14, train_idx)
    alpha, cv = cv_select_alpha(X_train, y_train, train_groups)
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256,), activation="relu", solver="adam", alpha=alpha,
                      learning_rate_init=1e-3, batch_size=64, max_iter=200, random_state=1020,
                      early_stopping=True, validation_fraction=0.2, n_iter_no_change=20,
                      shuffle=True, verbose=False),
    )
    clf.fit(X_train, y_train)

    states = ["clean", "seed0", "seed1", "seed2"]
    per_state = {}
    eval_scores_by_seed = {}
    line_true_by_seed = {}
    for st in states:
        X_eval = token_mean(acts, st, 14, eval_idx)
        X_line = token_mean(acts, st, 14, line_idx)
        eval_scores = clf.predict_proba(X_eval)[:, 1]
        line_scores = clf.predict_proba(X_line)[:, 1]
        c, far, t = catch_far(eval_scores, eval_true, line_scores)
        auc = float(roc_auc_score(1 - eval_true, eval_scores))
        per_state[st] = {"catch": c, "far": far, "threshold": t, "eval_auc_false_high": auc}
        if st != "clean":
            eval_scores_by_seed[st] = eval_scores
            line_true_by_seed[st] = line_scores
    pooled = city_bootstrap_catch(eval_scores_by_seed, eval_true, eval_city, line_true_by_seed)
    res = {"selected_alpha": alpha, "cv_auc_by_alpha": cv, "per_state": per_state,
           "pooled_attacked_catch_ci": {"point": pooled[0], "lo": pooled[1], "hi": pooled[2]},
           "n_train": int(len(train_idx)), "n_line_true": int(len(line_idx)), "n_eval": int(len(eval_idx))}
    (out / "mlp14_raw_rebuild.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    return res


def continuation_lp_for_pair(tokenizer, off: np.ndarray, lp: np.ndarray, i_true: int, i_false: int, texts: List[str]) -> Tuple[float, float]:
    ids_true = tokenizer(texts[i_true], add_special_tokens=False)["input_ids"]
    ids_false = tokenizer(texts[i_false], add_special_tokens=False)["input_ids"]
    n_prefix_tok = 0
    for a_id, b_id in zip(ids_true, ids_false):
        if a_id != b_id:
            break
        n_prefix_tok += 1
    # lp[t] is the log-probability assigned to token t+1 by the prefix through token t.
    # So the first continuation token at token index n_prefix_tok is scored at lp[n_prefix_tok - 1].
    a, b = int(off[i_true]) + max(n_prefix_tok - 1, 0), int(off[i_true + 1]) - 1
    c, d = int(off[i_false]) + max(n_prefix_tok - 1, 0), int(off[i_false + 1]) - 1
    return float(lp[a:b].sum()), float(lp[c:d].sum())


def knowledge_check(project: Path, out: Path) -> Dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    split = load_split(project / "data" / "ref1019" / "split.json")
    eval_rows = [i for i, r in enumerate(split) if r["split"] == "eval" and r["src"] == "cities"]
    by_city = defaultdict(dict)
    for i in eval_rows:
        by_city[split[i]["city"]][int(split[i]["true"])] = i
    pairs = [(v[1], v[0]) for v in by_city.values() if 1 in v and 0 in v]
    texts = [r["text"] for r in split]
    res = {}
    for st in ["clean", "seed0", "seed1", "seed2"]:
        d = project / "data" / "acts" / st
        off = np.load(d / "off.npy", mmap_mode="r")
        lp = np.load(d / "lp.npy", mmap_mode="r")
        diffs = []
        true_lps = []
        false_lps = []
        for it, iff in pairs:
            lt, lf = continuation_lp_for_pair(tokenizer, off, lp, it, iff, texts)
            true_lps.append(lt); false_lps.append(lf); diffs.append(lt - lf)
        diffs = np.array(diffs)
        res[st] = {"pair_accuracy_true_gt_false": float((diffs > 0).mean()),
                   "mean_logp_true_cont": float(np.mean(true_lps)),
                   "mean_logp_false_cont": float(np.mean(false_lps)),
                   "n_pairs": len(pairs)}
    attacked = np.concatenate([np.repeat(res[st]["pair_accuracy_true_gt_false"], 1) for st in ["seed0","seed1","seed2"]])
    res["pooled_attacked_mean_pair_accuracy"] = float(np.mean([res[st]["pair_accuracy_true_gt_false"] for st in ["seed0", "seed1", "seed2"]]))
    res["clean_minus_0p05_gate"] = float(res["clean"]["pair_accuracy_true_gt_false"] - 0.05)
    (out / "knowledge_rebuild.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="kran-research/projects/megaprobe-after")
    ap.add_argument("--out", default="swarm-theory-1020-megaprobe-after/skeptic2")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-knowledge", action="store_true")
    args = ap.parse_args()
    project = Path(args.project)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = {}
    if not args.skip_mlp:
        final["mlp14"] = fit_mlp14(project, out)
        print("MLP14", json.dumps(final["mlp14"], indent=2)[:2000])
    if not args.skip_knowledge:
        final["knowledge"] = knowledge_check(project, out)
        print("knowledge", json.dumps(final["knowledge"], indent=2)[:2000])
    (out / "raw_rebuild_summary.json").write_text(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
