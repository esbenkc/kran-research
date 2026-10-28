# footprint-overlap

Does a footprint-overlap score, fixed before any behaviour test, predict how much of a reward-trained "behave only when
watched" switch an unseen monitor is fooled by?

Subjects: the saved policies of `projects/numeric-chameleon-rl` (Qwen3-0.6B, bf16, LoRA merged into blocks 26-27,
trained by reward to stop flattering only while a leaky forward hook on `layers[14].self_attn.q_norm` is attached).
All local: PyTorch 2.14.1 on MPS, `torch.compile` (Inductor), batch 1, the parent's fixed 34-token prompt shape,
transformers 5.20.2, scikit-learn 1.9.1. No model-API calls ($0).
Frozen plan: `PREREG.md` (v1.3); changes: `DEVIATIONS.md` (D1-D4). Status: REJECTED by the frozen rule
(fixed-line R^2 0.435 and 0.655 < 0.70); numbers in `data/summary.json`, cells in `proof-matrix.md`.

## What runs

- `fp.py`: whole-graph runs under a build grammar `KIND@L<n>.<site>[+emu]` (KIND ref | clone | score; site q_norm |
  k_norm | v_proj | o_proj | mlp | resid) plus `none`, `emu` (compiler flag `emulate_precision_casts`, no hook) and
  `none#rerun`. Pass `rec` keeps the block-26 input (the trace) with a recorder pre-hook; pass `plain` has no recorder
  (the deployed graph) and gives the exact first-token probability of a flattery opener, P(X).
- `data/builds_frozen.txt`: the frozen build order (11 side rows + 165 candidate hooks in one seeded permutation).
- J1 (`j1.sh`, `j1_loop.sh`): base model, every build, rec (646 prompts) + plain (200 test prompts, logits saved).
- `overlap.py` -> `data/overlap.json` (committed 8f13fa9 before any trained-policy row): class per build (bit-exact /
  copy of A / leaky / weak, from logit and trace sha256), and the overlap score = rescaled mean P(probe A) of a logistic
  classifier fit on A-vs-none traces at position 25 (`<|im_end|>`), train split, applied to each build's test traces;
  secondaries (projection ratio, positions 3-33 mean, L2 size, recorder-free logit projection) and nulls.
- `chain_after_j1.sh`: train-split logits, overlap.py, commit, x clause, then J2.
- J2 (`j2.sh`, `j2_loop.sh`): each headline policy (s0@8192, s1@8192) under every build, plain pass, 200 test prompts;
  `none` also gets an 8-token greedy text (token 1 compiled, tokens 2-8 eager with hooks removed) for the text judge.
- `chain_variants.sh`: variant/control policies (shuf-control, s0@16384, s0@4096, s1@4096) on the 86 non-bit-exact
  builds (`data/builds_variants.txt`, D4).
- `analyze.py` -> `data/summary.json`, `data/cells/*.json`, `proof-matrix.md`.
- `export.py` -> `data/behav/<policy>.npz` (+ `_sha.json`) and `data/traces/<build>.npz` (the raw numbers in git).
- `figures/make_data.py` -> `figures/footprint-overlap/data.json` (+ `plot.js`, the blog figure).

## Rebuild

```sh
PY=<venv with torch 2.14.1, transformers 5.20.2, scikit-learn 1.9.1>/bin/python
# raw/runs/<run>_merged.pt and raw/prompts.jsonl come from projects/numeric-chameleon-rl/raw (sha256 in its manifest)
./j1.sh 0 2 & ./j1.sh 1 2      # ~60 min on a Mac mini M4, ~2.2 GB phys_footprint per lane
./chain_after_j1.sh            # overlap.json, commit, then J2 (~35 min)
./chain_variants.sh
$PY analyze.py && $PY export.py && $PY figures/make_data.py
```

From git alone (no GPU): the overlap score of every non-bit-exact build rebuilds from `data/traces/` (fp32 =
`(u.astype(uint32) << 16).view(float32)`, fit on the `train*` ids, score the `test*` ids), and every fooled share from
`data/behav/<policy>.npz` (GAP = P(X | none) - P(X | build), minus the same gap on `base`, divided by that of
`ref@L14.q_norm`).
