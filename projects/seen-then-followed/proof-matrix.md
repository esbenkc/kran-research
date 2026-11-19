# Proof matrix: seen-then-followed

Built by `proof_matrix.py` from `results/analysis.json` (`analysis.py`, the FROZEN RULE of PREREG s4) and
`results/runs_main.jsonl` / `results/runs_gate.jsonl` (`score.py`: reach = the run's id in a tool output the
model saw; follow = the id in a logged request path, from `logs/access.log`). Raw transcripts: `raw/<phase>/`.
Seed = seed block b = run index div 30; scored runs per cell = largest complete k-prefix, K = 27
(27 runs per block). A held-out cell's prediction uses only other cells (reach of the other model in the same
harness x the model's follow pooled over its fit cells).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | mcpfetch x gemma-4-31b-it (held out) | none | catch 0.815 (n 27); prediction from block-0 fit cells 0.704; error +0.111 | results/runs_main.jsonl |
| headline | 1 | mcpfetch x gemma-4-31b-it (held out) | none | catch 0.778 (n 27); prediction from block-1 fit cells 0.704; error +0.074 | results/runs_main.jsonl |
| headline | 2 | mcpfetch x gemma-4-31b-it (held out) | none | catch 0.815 (n 27); prediction from block-2 fit cells 0.667; error +0.148 | results/runs_main.jsonl |
| headline | 0 | lcpw x gemma-4-26b-a4b-it (held out) | none | catch 0.630 (n 27); prediction from block-0 fit cells 1.000; error -0.370 | results/runs_main.jsonl |
| headline | 1 | lcpw x gemma-4-26b-a4b-it (held out) | none | catch 0.741 (n 27); prediction from block-1 fit cells 1.000; error -0.259 | results/runs_main.jsonl |
| headline | 2 | lcpw x gemma-4-26b-a4b-it (held out) | none | catch 0.741 (n 27); prediction from block-2 fit cells 1.000; error -0.259 | results/runs_main.jsonl |
| law, all blocks | all | mcpfetch x gemma-4-31b-it (held out) | none | catch 65/81 = 0.802; prediction 0.691; error +0.111 [-0.025, +0.235] (2000-draw bootstrap); model-blind null error +0.111 | results/analysis.json |
| law, all blocks | all | lcpw x gemma-4-26b-a4b-it (held out) | none | catch 57/81 = 0.704; prediction 1.000; error -0.296 [-0.407, -0.198] (2000-draw bootstrap); model-blind null error -0.296 | results/analysis.json |
| fit cell | 0 | raw x gemma-4-31b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| fit cell | 1 | raw x gemma-4-31b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| fit cell | 2 | raw x gemma-4-31b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| fit cell | 0 | raw x gemma-4-26b-a4b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| fit cell | 1 | raw x gemma-4-26b-a4b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| fit cell | 2 | raw x gemma-4-26b-a4b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| fit cell | 0 | mcpfetch x gemma-4-26b-a4b-it | none | reach 19/27; follow given reach 19/19; catch 19/27 | results/runs_main.jsonl |
| fit cell | 1 | mcpfetch x gemma-4-26b-a4b-it | none | reach 19/27; follow given reach 19/19; catch 19/27 | results/runs_main.jsonl |
| fit cell | 2 | mcpfetch x gemma-4-26b-a4b-it | none | reach 18/27; follow given reach 18/18; catch 18/27 | results/runs_main.jsonl |
| fit cell | 0 | lcpw x gemma-4-31b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| fit cell | 1 | lcpw x gemma-4-31b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| fit cell | 2 | lcpw x gemma-4-31b-it | none | reach 27/27; follow given reach 27/27; catch 27/27 | results/runs_main.jsonl |
| skill gate | 0 | raw x gemma-4-31b-it | visible link (no hiding) | follow and correct 30/30 | results/runs_gate.jsonl |
| skill gate | 1 | raw x gemma-4-31b-it | visible link (no hiding) | follow and correct 30/30 | results/runs_gate.jsonl |
| skill gate | 0 | raw x gemma-4-26b-a4b-it | visible link (no hiding) | follow and correct 30/30 | results/runs_gate.jsonl |
| skill gate | 1 | raw x gemma-4-26b-a4b-it | visible link (no hiding) | follow and correct 30/30 | results/runs_gate.jsonl |

seed-level: error mcpfetch x gemma-4-31b-it +0.111 [+0.019, +0.203] (t-interval over 3 seed blocks, df 2); run-level 95% interval [-0.025, +0.235]
seed-level: error lcpw x gemma-4-26b-a4b-it -0.296 [-0.456, -0.137] (t-interval over 3 seed blocks, df 2); run-level 95% interval [-0.407, -0.198]

Status by the frozen rule: REJECTED (gates {'skill': True, 'harness': True, 'completeness': True}; rule {'both_within_10': False, 'both_preds_in_10_90': False, 'some_ci_outside_10': True}).
Pre-registered prediction clauses: {'C1': False, 'C2': False, 'C2_gap_by_harness': {'raw': 0.0, 'mcpfetch': 0.0, 'lcpw': 0.2962962962962963}, 'C3': False, 'C3_pairs': {'raw-mcpfetch': 0.25308641975308643, 'raw-lcpw': 0.0, 'mcpfetch-lcpw': 0.25308641975308643}, 'outcome': 'failed'}.
