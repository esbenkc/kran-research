#!/usr/bin/env python3
from __future__ import annotations

import json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path('kran-research/projects/bf16-round/raw/detectors/Qwen3-1.7B')
OUT = Path('swarm-oracle-1023-bf16-round/skeptic2')


def binom_upper(k: int, n: int, p: float = 0.5) -> float:
    # stable recurrence for upper tail
    probs = []
    q0 = (1-p) ** n
    probs.append(q0)
    for i in range(0, n):
        probs.append(probs[-1] * (n-i)/(i+1) * p/(1-p))
    return float(sum(probs[k:]))


def check_file(path: Path):
    obj = json.loads(path.read_text())
    rows = obj['rows']
    n = len(rows)
    correct = sum(1 for r in rows if bool(r['correct']))
    by_prompt = defaultdict(dict)
    for r in rows:
        by_prompt[r['prompt']][int(r['probe'])] = r
    pair_count = 0
    identical = 0
    deltas = []
    both_correct = 0
    for prompt, d in by_prompt.items():
        if 0 in d and 1 in d:
            pair_count += 1
            if d[0].get('sha256_all_position_logits') == d[1].get('sha256_all_position_logits'):
                identical += 1
            deltas.append(float(d[1]['logodds_yes_no']) - float(d[0]['logodds_yes_no']))
            both_correct += int(bool(d[0]['correct']) and bool(d[1]['correct']))
    acc = correct / n if n else float('nan')
    sep = sum(deltas)/len(deltas) if deltas else float('nan')
    return {
        'path': str(path),
        'fix': obj.get('fix'),
        'run': obj.get('run'),
        'emulate_precision_casts': obj.get('emulate_precision_casts'),
        'torch': obj.get('torch'),
        'correct': correct,
        'n_rows': n,
        'acc': acc,
        'acc_field': obj.get('acc'),
        'pair_count': pair_count,
        'sha_identical_pairs': identical,
        'sha_identical_field': obj.get('prompts_bit_identical_probe_vs_none'),
        'mean_probe_minus_none_logodds': sep,
        'separation_field': obj.get('separation_nats'),
        'both_correct_pairs': both_correct,
        'binom_p_ge_acc_if_chance': binom_upper(correct, n, 0.5) if n <= 512 else None,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = [check_file(p) for p in sorted(ROOT.glob('*/*.json'))]
    (OUT/'detector_check.json').write_text(json.dumps(res, indent=2))
    lines = ['# skeptic2 detector check', '']
    for r in res:
        rel = Path(r['path']).relative_to(Path.cwd()) if Path(r['path']).is_absolute() else r['path']
        lines.append(f"- {rel}: acc {r['correct']}/{r['n_rows']} = {r['acc']:.4f}; sha-identical prompt pairs {r['sha_identical_pairs']}/{r['pair_count']}; separation {r['mean_probe_minus_none_logodds']:.3f} nats; p_chance_ge {r['binom_p_ge_acc_if_chance']:.3g}")
    lines.append('')
    # Null summary: label-free chance for the decisive fresh detector.
    fresh = [r for r in res if r['fix']=='round16' and r['run']=='fresh_round16_probe_none_s0' and not r['emulate_precision_casts']]
    if fresh:
        r = fresh[0]
        lines.append(f"Decisive fresh detector null: {r['correct']}/{r['n_rows']} correct; P[Bin({r['n_rows']},0.5) >= {r['correct']}] = {r['binom_p_ge_acc_if_chance']:.3g}.")
    (OUT/'detector_check.md').write_text('\n'.join(lines))
    print(OUT/'detector_check.md')

if __name__ == '__main__':
    main()
