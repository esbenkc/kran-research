"""Validate a replacement for gpt-4.1-mini (OpenAI credits ran out on 2026-10-18 03:10 CEST).

For each candidate model, with the UNCHANGED probe-goodhart prompts:
  1. deception judge (common.judge_one) on 100 DolusChat reference replies (50 group-2 calibration prompts x
     truthful/deceptive; not eval prompts) -> accuracy;
  2. deception judge on 200 saved probe-goodhart eval answers (lam0_s0 and lam1_s0, rounds 0 and 8, 50 each)
     -> agreement with the saved gpt-4.1-mini labels;
  3. favourability judge (judge.favourability) on 200 saved probe-goodhart TRAIN samples (lam1_s0)
     -> Spearman correlation with the saved gpt-4.1-mini favourability scores.

    with-model-keys uv run python validate_judge.py gemini-3-flash-preview claude-haiku-4-5   (from the project dir)
-> data/judge_validation/<model>.json
"""
from __future__ import annotations

import importlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PG = Path.home() / "work/oracle-1014-probe-goodhart/kran-research/projects/probe-goodhart/data/results"
OUT = HERE / "data/judge_validation"


def main(models):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    import common
    S = common.load_splits()
    by_id = {common.pid(r): r for r in S["eval"] + S["train"]}
    ref = [(r, r["responses"][k], int(k == "deceptive")) for r in S["calib"][:50] for k in ("truthful", "deceptive")]
    ev = []
    for run in ("lam0_s0", "lam1_s0"):
        rows = [json.loads(l) for l in (HERE / f"data/reused/eval_{run}.jsonl").open()]
        for rnd in (0, 8):
            ev += rng.sample([r for r in rows if r["round"] == rnd], 50)
    tr = rng.sample([json.loads(l) for l in (PG / "train_lam1_s0.jsonl").open()], 200)
    for m in models:
        os.environ["JUDGE_MODEL"] = m
        import judge
        importlib.reload(judge)
        t0 = time.time()
        with ThreadPoolExecutor(16) as ex:
            ref_lab = list(ex.map(lambda x: common.judge_one(x[0], x[1]), ref))
            ev_lab = list(ex.map(lambda r: common.judge_one(by_id[r["prompt_id"]], r["response"]), ev))
            fav = list(ex.map(lambda r: judge.favourability(by_id[r["prompt_id"]], r["response"]), tr))
        y = np.array([x[2] for x in ref])
        pred = np.array([l == "DECEPTIVE" for l in ref_lab])
        old = np.array([r["judge"] for r in ev])
        new = np.array(ev_lab)
        res = {"model": m, "seconds": round(time.time() - t0, 1),
               "ref_accuracy": float(np.mean(pred == y)), "ref_counts": {k: ref_lab.count(k) for k in set(ref_lab)},
               "eval_agreement_with_gpt41mini": float(np.mean(old == new)),
               "eval_deceptive_rate_old_new": [float(np.mean(old == "DECEPTIVE")), float(np.mean(new == "DECEPTIVE"))],
               "eval_by_stratum": {}, "fav_spearman_vs_gpt41mini": float(spearmanr([r["favour"] for r in tr], fav)[0]),
               "fav_mean_old_new": [float(np.mean([r["favour"] for r in tr])), float(np.mean(fav))],
               "spend_usd_est_total_ledger": judge.spend()}
        for i, (run, rnd) in enumerate([(a, b) for a in ("lam0_s0", "lam1_s0") for b in (0, 8)]):
            sl = slice(i * 50, (i + 1) * 50)
            res["eval_by_stratum"][f"{run}_r{rnd}"] = {"old": float(np.mean(old[sl] == "DECEPTIVE")),
                                                       "new": float(np.mean(new[sl] == "DECEPTIVE")),
                                                       "agree": float(np.mean(old[sl] == new[sl]))}
        (OUT / f"{m}.json").write_text(json.dumps({**res, "ref_labels": ref_lab, "eval_labels": ev_lab,
                                                   "eval_keys": [[r["run"], r["round"], r["prompt_id"], r["sample"]]
                                                                 for r in ev], "fav": fav}, indent=1))
        print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main(sys.argv[1:])
