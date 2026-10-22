#!/usr/bin/env python3
"""Independent raw-data checker for oracle-1022-probe-sensing-mps.

This script intentionally does not import project experiment code. It reads only saved
arrays and manifests, then recomputes bitwise diffs, detector accuracies, and nulls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        keys = list(z.keys())
        if len(keys) != 1:
            raise ValueError(f"{path} has {len(keys)} arrays; use key::path syntax")
        return z[keys[0]]
    if "::" in str(path):
        key, raw = str(path).split("::", 1)
        return np.load(raw, allow_pickle=False)[key]
    if suffix in {".pt", ".pth"}:
        import torch

        t = torch.load(path, map_location="cpu")
        if isinstance(t, dict) and len(t) == 1:
            t = next(iter(t.values()))
        if hasattr(t, "detach"):
            t = t.detach().cpu()
            if str(t.dtype) == "torch.bfloat16":
                return t.view(torch.uint16).numpy()
            if str(t.dtype) == "torch.float16":
                return t.view(torch.uint16).numpy()
            if str(t.dtype) == "torch.float32":
                return t.view(torch.uint32).numpy()
            return t.numpy()
        raise ValueError(f"unsupported torch object in {path}: {type(t)}")
    raise ValueError(f"unsupported array format: {path}")


def as_bits(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.float16:
        return a.view(np.uint16)
    if a.dtype == np.float32:
        return a.view(np.uint32)
    if a.dtype == np.float64:
        return a.view(np.uint64)
    if a.dtype in (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64):
        return a
    # Last resort: compare bytes without interpreting dtype.
    return a.view(np.uint8)


def max_abs_float(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape:
        return None
    try:
        af = a.astype(np.float64, copy=False)
        bf = b.astype(np.float64, copy=False)
        return float(np.max(np.abs(af - bf))) if af.size else 0.0
    except Exception:
        return None


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    lb = as_bits(left)
    rb = as_bits(right)
    if lb.shape != rb.shape:
        lb = lb.view(np.uint8)
        rb = rb.view(np.uint8)
    diff = lb != rb
    return {
        "same_shape": True,
        "shape": list(left.shape),
        "dtype_left": str(left.dtype),
        "dtype_right": str(right.dtype),
        "n_values": int(diff.size),
        "n_bitwise_diff": int(np.count_nonzero(diff)),
        "bitwise_equal": bool(not np.any(diff)),
        "max_abs": max_abs_float(left, right),
    }


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return base / p


def run_diff_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    base = path.parent
    comparisons = manifest.get("comparisons") or manifest.get("diff_comparisons") or []
    if not comparisons:
        raise ValueError("manifest has no comparisons/diff_comparisons list")
    out = []
    for row in comparisons:
        left_path = resolve(base, row.get("left_path") or row.get("left"))
        right_path = resolve(base, row.get("right_path") or row.get("right"))
        if left_path is None or right_path is None:
            raise ValueError(f"comparison lacks paths: {row}")
        left = load_array(left_path)
        right = load_array(right_path)
        stats = compare_arrays(left, right)
        meta = {k: v for k, v in row.items() if k not in {"left_path", "right_path", "left", "right"}}
        out.append({
            **meta,
            "left_path": str(left_path),
            "right_path": str(right_path),
            "left_sha256": sha256(left_path),
            "right_sha256": sha256(right_path),
            **stats,
        })
    return {"source_manifest": str(path), "comparisons": out}


def yes_no_margin(r: dict[str, Any]) -> float | None:
    if "yes_logit" in r and "no_logit" in r:
        return float(r["yes_logit"]) - float(r["no_logit"])
    if "yes_logprob" in r and "no_logprob" in r:
        return float(r["yes_logprob"]) - float(r["no_logprob"])
    if "logp_yes" in r and "logp_no" in r:
        return float(r["logp_yes"]) - float(r["logp_no"])
    return None


def predicted_label(r: dict[str, Any]) -> str:
    pred = str(r.get("pred", r.get("prediction", r.get("top_token", "")))).strip().lower()
    if pred:
        if pred.startswith("yes"):
            return "yes"
        if pred.startswith("no"):
            return "no"
        return pred
    margin = yes_no_margin(r)
    if margin is None:
        return ""
    return "yes" if margin > 0 else "no"


def target_label(r: dict[str, Any]) -> str:
    target = str(r.get("target", r.get("label", r.get("expected", "")))).strip().lower()
    if target.startswith("yes"):
        return "yes"
    if target.startswith("no"):
        return "no"
    cond = str(r.get("condition", r.get("probe", ""))).strip().lower()
    if cond in {"on", "probe", "probed", "yes", "1", "true", "copy", "two-kernel", "twokernel"}:
        return "yes"
    if cond in {"off", "none", "no_probe", "no-probe", "nohook", "no", "0", "false"}:
        return "no"
    return target


def acc_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        split = str(r.get("split", ""))
        build = str(r.get("build", r.get("condition", "")))
        seed = str(r.get("seed", ""))
        eval_mode = str(r.get("eval_mode", r.get("mode", r.get("backend", ""))))
        by_run[(split, build, seed, eval_mode)].append(r)
    runs = []
    for (split, build, seed, eval_mode), rs in sorted(by_run.items()):
        ok = []
        margins = []
        paired: dict[str, dict[str, bool]] = defaultdict(dict)
        for r in rs:
            pred = predicted_label(r)
            target = target_label(r)
            if target and pred:
                correct = pred == target
                ok.append(correct)
                prompt_id = str(r.get("prompt_id", r.get("id", r.get("prompt", ""))))
                condition = str(r.get("condition", r.get("probe", target))).strip().lower()
                if prompt_id:
                    paired[prompt_id][condition] = correct
            margin = yes_no_margin(r)
            if margin is not None:
                margins.append(margin)
        both = []
        for per_prompt in paired.values():
            has_yes = any(k in {"on", "probe", "probed", "yes", "1", "true", "copy", "two-kernel", "twokernel"} for k in per_prompt)
            has_no = any(k in {"off", "none", "no_probe", "no-probe", "nohook", "no", "0", "false"} for k in per_prompt)
            if has_yes and has_no:
                both.append(all(per_prompt.values()))
        runs.append({
            "split": split,
            "build": build,
            "seed": seed,
            "eval_mode": eval_mode,
            "n": len(rs),
            "accuracy": float(sum(ok) / len(ok)) if ok else None,
            "paired_prompts": len(both),
            "both_answers_correct": float(sum(both) / len(both)) if both else None,
            "mean_yes_minus_no": float(np.mean(margins)) if margins else None,
        })
    return {"runs": runs}


def shuffled_label_null(rows: list[dict[str, Any]], n_perm: int, rng_seed: int) -> dict[str, Any]:
    rng = random.Random(rng_seed)
    scored = []
    for r in rows:
        pred = predicted_label(r)
        target = target_label(r)
        if pred and target:
            scored.append((pred, target))
    if not scored:
        return {"n": 0, "permutations": n_perm, "mean": None, "p95": None, "max": None}
    labels = [t for _, t in scored]
    preds = [p for p, _ in scored]
    accs = []
    for _ in range(n_perm):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        accs.append(sum(p == t for p, t in zip(preds, shuffled)) / len(scored))
    return {
        "n": len(scored),
        "permutations": n_perm,
        "mean": float(np.mean(accs)),
        "p95": float(np.quantile(accs, 0.95)),
        "max": float(np.max(accs)),
    }


def run_scores(path: Path, n_perm: int, rng_seed: int) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        rows = load_jsonl(path)
    elif path.suffix == ".csv":
        rows = read_table(path)
    else:
        obj = load_json(path)
        rows = obj.get("rows", obj if isinstance(obj, list) else [])
    if not isinstance(rows, list):
        raise ValueError("score file must be json/jsonl/csv rows")
    out = acc_from_rows(rows)
    out["shuffled_label_null"] = shuffled_label_null(rows, n_perm=n_perm, rng_seed=rng_seed)
    out["source_scores"] = str(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-manifest", type=Path, help="JSON manifest with raw array comparisons")
    ap.add_argument("--scores", type=Path, help="JSON/JSONL/CSV detector outputs with pred+target columns")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1022)
    args = ap.parse_args()

    result: dict[str, Any] = {"checker": "skeptic2/checker.py", "seed": args.seed}
    if args.diff_manifest:
        result["diffs"] = run_diff_manifest(args.diff_manifest)
    if args.scores:
        result["scores"] = run_scores(args.scores, args.n_perm, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
