#!/usr/bin/env python3
"""Rebuild Step-2 detector metrics from per-prompt Yes/No logprobs.

Expected rows contain prompt_id, split, build, seed, condition, and either
(yes_logprob,no_logprob), (logp_yes,logp_no), or pred/target. No project code import.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

YES_CONDS = {"on", "probe", "probed", "yes", "1", "true", "twok", "two-kernel", "fp32", "rec"}
NO_CONDS = {"off", "none", "no_probe", "no-probe", "nohook", "no", "0", "false", "plain"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    obj = json.loads(path.read_text(encoding="utf-8"))
    stem_seed = ""
    m = re.search(r"_s(\d+)(?:\D|$)", path.stem)
    if m:
        stem_seed = m.group(1)
    if isinstance(obj, list):
        return obj
    defaults = {
        "run": obj.get("run", path.stem),
        "split": obj.get("split", "test"),
        "build": obj.get("eval_build", obj.get("build", obj.get("args", {}).get("build", ""))),
        "seed": obj.get("seed", obj.get("args", {}).get("seed", stem_seed)),
        "eval_mode": obj.get("eval_mode", obj.get("backend", obj.get("args", {}).get("backend", ""))),
        "null": obj.get("null", obj.get("args", {}).get("null", "")),
        "yes_id": obj.get("yes_id", obj.get("args", {}).get("yes_id", "")),
        "no_id": obj.get("no_id", obj.get("args", {}).get("no_id", "")),
    }
    for key in ["rows", "test_rows", "scores", "per_prompt", "evals", "results"]:
        if isinstance(obj.get(key), list):
            rows = []
            for row in obj[key]:
                merged = dict(defaults)
                merged.update(row)
                rows.append(merged)
            return rows
    # Cached trainer files store final per-case rows under test.rows in some runs.
    if isinstance(obj.get("test"), dict) and isinstance(obj["test"].get("rows"), list):
        rows = []
        for row in obj["test"]["rows"]:
            merged = dict(defaults)
            merged.update(row)
            rows.append(merged)
        return rows
    raise ValueError(f"cannot find rows in {path}")

def margin(row: dict[str, Any]) -> float | None:
    for y, n in [("yes_logprob", "no_logprob"), ("logp_yes", "logp_no"), ("yes_logit", "no_logit")]:
        if y in row and n in row:
            return float(row[y]) - float(row[n])
    if "logodds_yes_no" in row:
        return float(row["logodds_yes_no"])
    return None


def pred(row: dict[str, Any]) -> str:
    raw = str(row.get("pred", row.get("prediction", row.get("top_token", "")))).strip().lower()
    if not raw and "top" in row:
        top = str(row["top"])
        if top and top == str(row.get("yes_id", "")):
            return "yes"
        if top and top == str(row.get("no_id", "")):
            return "no"
        raw = top
    m = margin(row)
    if m is None:
        return raw
    return "yes" if m > 0 else "no"


def target(row: dict[str, Any]) -> str:
    raw = str(row.get("target", row.get("label", row.get("expected", "")))).strip().lower()
    if raw.startswith("yes"):
        return "yes"
    if raw.startswith("no"):
        return "no"
    cond = str(row.get("condition", row.get("probe", row.get("pass", "")))).strip().lower()
    if cond in YES_CONDS:
        return "yes"
    if cond in NO_CONDS:
        return "no"
    return raw


def cond_side(row: dict[str, Any]) -> str:
    cond = str(row.get("condition", row.get("probe", row.get("pass", target(row))))).strip().lower()
    if cond in YES_CONDS or target(row) == "yes":
        return "yes"
    if cond in NO_CONDS or target(row) == "no":
        return "no"
    return cond


def summarize(rows: list[dict[str, Any]], n_perm: int, rng_seed: int) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        split = str(r.get("split", ""))
        build = str(r.get("build", r.get("condition", "")))
        seed = str(r.get("seed", ""))
        eval_mode = str(r.get("eval_mode", r.get("mode", r.get("backend", ""))))
        run = str(r.get("run", ""))
        null = str(r.get("null", ""))
        groups[(split, build, seed, eval_mode, run, null)].append(r)

    out = []
    rng = random.Random(rng_seed)
    for (split, build, seed, eval_mode, run, null), rs in sorted(groups.items()):
        scored = []
        margins_by_prompt: dict[str, dict[str, float]] = defaultdict(dict)
        correct_by_prompt: dict[str, dict[str, bool]] = defaultdict(dict)
        for r in rs:
            pr = pred(r)
            tg = target(r)
            if pr and tg:
                scored.append((pr, tg))
                pid = str(r.get("prompt_id", r.get("id", r.get("prompt", ""))))
                side = cond_side(r)
                if pid:
                    correct_by_prompt[pid][side] = (pr == tg)
            m = margin(r)
            pid = str(r.get("prompt_id", r.get("id", r.get("prompt", ""))))
            if m is not None and pid:
                margins_by_prompt[pid][cond_side(r)] = m
        acc = sum(p == t for p, t in scored) / len(scored) if scored else None
        paired_correct = []
        separations = []
        for pid, vals in correct_by_prompt.items():
            if "yes" in vals and "no" in vals:
                paired_correct.append(vals["yes"] and vals["no"])
        for pid, vals in margins_by_prompt.items():
            if "yes" in vals and "no" in vals:
                separations.append(vals["yes"] - vals["no"])
        null_accs = []
        if scored:
            labels = [t for _, t in scored]
            preds = [p for p, _ in scored]
            for _ in range(n_perm):
                shuf = labels[:]
                rng.shuffle(shuf)
                null_accs.append(sum(p == t for p, t in zip(preds, shuf)) / len(scored))
        out.append({
            "split": split,
            "build": build,
            "seed": seed,
            "eval_mode": eval_mode,
            "run": run,
            "null": null,
            "rows": len(rs),
            "accuracy": acc,
            "paired_prompts": len(paired_correct),
            "both_answers_correct": (sum(paired_correct) / len(paired_correct)) if paired_correct else None,
            "mean_separation_yes_minus_no": float(np.mean(separations)) if separations else None,
            "min_separation_yes_minus_no": float(np.min(separations)) if separations else None,
            "shuffled_null_mean_acc": float(np.mean(null_accs)) if null_accs else None,
            "shuffled_null_p95_acc": float(np.quantile(null_accs, 0.95)) if null_accs else None,
            "shuffled_null_max_acc": float(np.max(null_accs)) if null_accs else None,
        })
    return {"runs": out, "n_rows": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1022)
    args = ap.parse_args()
    rows = []
    for path in args.scores:
        loaded = load_rows(path)
        for r in loaded:
            r = dict(r)
            r.setdefault("source_file", str(path))
            rows.append(r)
    result = summarize(rows, args.n_perm, args.seed)
    result["source_files"] = [str(p) for p in args.scores]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
