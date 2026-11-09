#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RNG_SEED = 20261109
B = 2000


def prompt_bootstrap(vals: np.ndarray, prompts: np.ndarray, seed: int = RNG_SEED, b: int = B):
    vals = np.asarray(vals, dtype=np.float64)
    prompts = np.asarray(prompts)
    ug = np.unique(prompts)
    by = np.array([vals[prompts == p].mean() for p in ug], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = by[rng.integers(0, len(by), size=(b, len(by)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def gap_from_npz(rel: str):
    d = np.load(ROOT / rel)
    gap = d['c_n'].astype(np.float64) - d['c_l'].astype(np.float64)
    out = {
        'artifact': rel,
        'n_rows': int(len(gap)),
        'n_prompts': int(len(np.unique(d['prompt_idx']))),
        'gap': float(gap.mean()),
        'gap_ci': prompt_bootstrap(gap, d['prompt_idx']),
        'gap_pts': float(100 * gap.mean()),
        'gap_ci_pts': [float(100*x) for x in prompt_bootstrap(gap, d['prompt_idx'])],
        'cheat_none': float(np.mean(d['c_n'])),
        'cheat_leaky': float(np.mean(d['c_l'])),
    }
    if 'tv' in d.files:
        # Exact D is in the build json. This is the raw per-token TV sum sanity check.
        tv_sum = d['tv'].astype(np.float64).sum(axis=1)
        out['tv_sum_mean'] = float(tv_sum.mean())
        out['tv_sum_ci'] = prompt_bootstrap(tv_sum, d['prompt_idx'])
    if 'c_X' in d.files:
        forced = d['c_notX'].astype(np.float64) - d['c_X'].astype(np.float64)
        out['forced_switch'] = float(forced.mean())
        out['forced_switch_pts'] = float(100 * forced.mean())
        out['forced_switch_ci_pts'] = [float(100*x) for x in prompt_bootstrap(forced, d['prompt_idx'])]
    return out


def csup_real():
    d = np.load(ROOT / 'research-iris/csup_real_rerun_pairs.npz')
    vals = d['split_q_off'].astype(np.float64) - d['split_q_on'].astype(np.float64)
    split_test_rows = d['split_test_rows'].astype(int)
    split_test_diff_rows = d['split_test_diff_rows'].astype(int)
    all_vals = np.zeros(len(split_test_rows), dtype=np.float64)
    pos = {int(r): i for i, r in enumerate(split_test_rows)}
    for r, v in zip(split_test_diff_rows, vals):
        all_vals[pos[int(r)]] = v
    # prompt_idx in the pair file is concatenated train+test; split_test_rows index that concat directly.
    prompts = d['prompt_idx'][split_test_rows]
    train_prompts = set(map(int, d['prompt_idx'][d['split_train_rows'].astype(int)]))
    test_prompts = set(map(int, prompts))
    return {
        'artifact': 'research-iris/csup_real_rerun_pairs.npz',
        'n_test_rows': int(len(split_test_rows)),
        'n_diff_rows': int(len(split_test_diff_rows)),
        'n_test_prompts': int(len(test_prompts)),
        'n_train_prompts': int(len(train_prompts)),
        'train_test_prompt_overlap': int(len(train_prompts & test_prompts)),
        'C_sup': float(all_vals.mean()),
        'C_sup_ci': prompt_bootstrap(all_vals, prompts),
        'C_sup_pts': float(100 * all_vals.mean()),
        'C_sup_ci_pts': [float(100*x) for x in prompt_bootstrap(all_vals, prompts)],
        'diff_only_mean_pts': float(100 * vals.mean()),
        'acc_on_diff_qoff_gt_qon': float(np.mean(vals > 0)),
    }


def main():
    out = {
        'seed': RNG_SEED,
        'bootstrap_resamples': B,
        'real_s0': gap_from_npz('data/reads/raw/real_s0.npz'),
        'planted_s0': gap_from_npz('data/reads/raw/planted_s0.npz'),
        'planted30_s0': gap_from_npz('data/reads/raw/planted30_s0.npz'),
        'csup_real_rerun': csup_real(),
    }
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
