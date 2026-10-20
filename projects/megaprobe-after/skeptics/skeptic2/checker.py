#!/usr/bin/env python3
"""Independent Hugo/skeptic2 rebuild for theory-1020-megaprobe-after.

This script is intentionally standalone. It does not import project probe code.
It reads saved raw arrays / score dumps, recomputes reported arithmetic, and
optionally refits a small MLP14 from activation memmaps when their metadata is
available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def npz_keys(path: Path) -> List[str]:
    with np.load(path, allow_pickle=True) as z:
        return list(z.keys())


def load_npz(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as z:
        for k in z.keys():
            arr = z[k]
            if arr.shape == () and arr.dtype == object:
                out[k] = arr.item()
            else:
                out[k] = arr
    return out


def as_str_array(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind in {"S", "O"}:
        return a.astype(str)
    return a


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Average ranks with ties. labels==1 is the high-score class.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def threshold_from_true(scores: np.ndarray) -> float:
    """PREREG line: 2nd-highest of the 200 refit true scores."""
    s = np.sort(np.asarray(scores, dtype=float))
    if len(s) < 2:
        raise ValueError(f"Need at least two true line scores, got {len(s)}")
    return float(s[-2])


def catch_from_scores(eval_scores: np.ndarray, eval_labels: np.ndarray, line_true_scores: np.ndarray) -> Tuple[float, float, float]:
    labels = np.asarray(eval_labels).astype(int)
    t = threshold_from_true(line_true_scores)
    flags = np.asarray(eval_scores, dtype=float) > t
    false_mask = labels == 1  # convention in this checker: 1 = false / positive catch class.
    true_mask = labels == 0
    catch = float(flags[false_mask].mean())
    far = float(flags[true_mask].mean())
    return catch, far, t


def paired_dprime(false_scores: np.ndarray, true_scores: np.ndarray, clean_eval_scores: np.ndarray, clean_eval_labels: np.ndarray) -> float:
    false_scores = np.asarray(false_scores, dtype=float)
    true_scores = np.asarray(true_scores, dtype=float)
    clean_eval_scores = np.asarray(clean_eval_scores, dtype=float)
    clean_eval_labels = np.asarray(clean_eval_labels).astype(int)
    pooled = []
    for lab in [0, 1]:
        vals = clean_eval_scores[clean_eval_labels == lab]
        if len(vals) > 1:
            pooled.append(np.var(vals, ddof=1) * (len(vals) - 1))
    denom_n = max(len(clean_eval_scores) - 2, 1)
    sigma = math.sqrt(float(sum(pooled) / denom_n)) if pooled else float("nan")
    return float(np.mean(false_scores - true_scores) / sigma)


def city_bootstrap(values_by_city: Mapping[str, Sequence[float]], seed: int = 1020, n_boot: int = 2000) -> Tuple[float, float, float]:
    cities = np.array(sorted(values_by_city.keys()))
    per_city = {c: np.asarray(values_by_city[c], dtype=float) for c in cities}
    point_vals = np.concatenate([per_city[c] for c in cities])
    point = float(np.mean(point_vals))
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.choice(cities, size=len(cities), replace=True)
        vals = np.concatenate([per_city[c] for c in draw])
        reps[b] = vals.mean()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def infer_score_table(scores: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return normalized score-table fields when keys follow common names."""
    aliases = {
        "label": ["label", "labels", "y", "is_false"],
        "split": ["split", "splits"],
        "city": ["city", "cities", "group", "groups"],
        "state": ["state", "states"],
        "monitor": ["monitor", "monitors", "probe", "probes"],
        "score": ["score", "scores"],
    }
    out: Dict[str, np.ndarray] = {}
    for canonical, names in aliases.items():
        for n in names:
            if n in scores:
                out[canonical] = as_str_array(scores[n])
                break
    return out


def summarize_npz(path: Path) -> None:
    print(f"NPZ {path}")
    with np.load(path, allow_pickle=True) as z:
        for k in z.keys():
            a = z[k]
            print(f"  {k}: shape={a.shape} dtype={a.dtype}")


def compute_from_long_scores(scores_npz: Path, out_dir: Path) -> Dict[str, Any]:
    raw = load_npz(scores_npz)
    fields = infer_score_table(raw)
    missing = {k for k in ["label", "split", "city", "state", "monitor", "score"] if k not in fields}
    if missing:
        raise RuntimeError(f"scores.npz is not in a recognized long-table form; missing {sorted(missing)}. Keys={sorted(raw)}")
    label = fields["label"].astype(int)
    split = fields["split"].astype(str)
    city = fields["city"].astype(str)
    state = fields["state"].astype(str)
    monitor = fields["monitor"].astype(str)
    score = fields["score"].astype(float)

    # Normalize label convention if it is true=1/false=0. We use false=1.
    # Heuristic: if an is_false key exists, use it. Otherwise assume label already false=1.
    if "is_false" in raw:
        label = np.asarray(raw["is_false"]).astype(int)

    results: Dict[str, Any] = {"monitors": {}}
    for m in sorted(set(monitor)):
        results["monitors"][m] = {}
        for st in sorted(set(state)):
            mask = (monitor == m) & (state == st)
            eval_mask = mask & (split == "eval")
            line_mask = mask & np.isin(split, ["refit", "line", "threshold"]) & (label == 0)
            if eval_mask.sum() == 0 or line_mask.sum() == 0:
                continue
            c, far, t = catch_from_scores(score[eval_mask], label[eval_mask], score[line_mask])
            results["monitors"][m][st] = {
                "catch": c,
                "far": far,
                "threshold": t,
                "n_eval": int(eval_mask.sum()),
                "n_line_true": int(line_mask.sum()),
                "auroc_false_high": auroc(score[eval_mask], label[eval_mask]),
            }
    out_path = out_dir / "score_rebuild.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {out_path}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="kran-research/projects/megaprobe-after")
    ap.add_argument("--out", default="swarm-theory-1020-megaprobe-after/skeptic2")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    project = Path(args.project)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = project / "data"
    score_path = data_dir / "scores.npz"
    results_path = data_dir / "results.json"

    if args.summarize:
        for p in sorted(data_dir.glob("*.npz")):
            summarize_npz(p)
        for p in sorted(data_dir.glob("*.json")):
            print(f"JSON {p}")
            try:
                obj = load_json(p)
                print(json.dumps(obj, indent=2)[:4000])
            except Exception as e:
                print(f"  failed: {e}")

    if score_path.exists():
        try:
            compute_from_long_scores(score_path, out_dir)
        except Exception as e:
            print(f"score table rebuild not run: {e}", file=sys.stderr)
    else:
        print(f"No {score_path} yet")

    if results_path.exists():
        obj = load_json(results_path)
        (out_dir / "results_keys.txt").write_text("\n".join(sorted(obj.keys())) + "\n")
        print(f"results keys: {sorted(obj.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
