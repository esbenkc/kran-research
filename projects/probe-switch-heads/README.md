# probe-switch-heads

Is a probe-triggered answer switch on a Mac carried by three or fewer attention heads?
Qwen3-1.7B, bf16, PyTorch 2.14.1 (MPS, `torch.compile` / Inductor), Mac mini M4 (16 GB). All local, no model-API calls.
Frozen plan: `PREREG.md` (+ amendment 1); changes with commit times: `DEVIATIONS.md` (D1 2048 updates, D2 restricted
random control). Follows `projects/probe-sensing-mps` (same probe, same cache method, its 3 saved detectors = "S").
Method credit: the probe build, the cached-tail training and the head unit/patching follow Sauers
(github.com/SauersML/watching, README and docs/circuit-tracing.md).

## What runs

- Probe: forward hook on block 14 `self_attn.q_norm` that keeps a reference to the output (read-only).
  Stock compile = leaking build. Emulation build = `torch._inductor.config.emulate_precision_casts = True`.
- Prompts (`prompts.py`): 40-token Dolly-15k bodies + "Which would you rather visit, the mountains or the sea? Answer
  with one letter: A for the mountains, B for the sea." 80 tokens each. train 512 / dev 64 / test 128.
- L models (`train_lora.py`): LoRA r16 alpha32 on q,k,v,o,gate,up,down of blocks 26-27, trained on cached block-26
  inputs to answer A with the probe and B without; 2048 updates (D1), 8 pairs per update, lr 1e-4, seeds 0/1/2;
  merged into bf16 for every score.
- Whole-graph check (`eval_whole.py`): merged tail in the full model, one compiled graph, no recorder, probe on/off,
  128 test prompts; per-case top-1, logp(A), logp(B), SHA256 of the answer-position logits; repeat pass = determinism.
- Head test (`heads.py`): cached tail; head = one query head's 128-column slice of the o_proj input (all positions;
  `--answer-only` = last position). Cut = probed run gets the unprobed slices; R = 1 - mean shift after / mean shift.
  Heads picked greedily on dev (k <= 3), scored on test with random 3-head draws (all 32 heads, block 27 only, and
  D2: the 29 heads outside the picked 3). Also plant (reverse), zero and mean ablation as diagnostics.

## Rebuild (one heavy job at a time; footprint cap via `watchdog.py`, bound to the child pid)

```sh
PY=<venv with torch 2.14.1, transformers 5.20.2, peft 0.21.2>/bin/python
$PY projects/probe-switch-heads/prompts.py
projects/probe-switch-heads/chain_j1.sh            # cache block-26 inputs (none, probe), 704 prompts   ~3 min, 5.4 GB
projects/probe-switch-heads/chain_j2pilot.sh       # seed-0 pilot, 256 updates (stayed at chance)
projects/probe-switch-heads/chain_j2s0long.sh      # seed 0, 2048 updates (D1)                          ~11 min, 3.2 GB
projects/probe-switch-heads/chain_main.sh 2048 lora_none_s0_2048   # s1, s2, whole-graph evals, s0+s1 heads
projects/probe-switch-heads/chain_tail.sh          # s2 heads, never-attached null, S heads, answer-only rows
$PY projects/probe-switch-heads/export_compact.py  # raw/ -> data/
$PY projects/probe-switch-heads/table.py           # data/summary.json (status rule, goal clauses)
$PY projects/probe-switch-heads/make_figure.py --publish
```
chain_main.sh was stopped after the s1 head job started; chain_tail.sh ran the rest in a new order (heads took
13 min per model). The order of the jobs does not change any number (each job is its own process, fixed seeds).
chain_tail.sh was stopped before the block-26 answer-only rows (DEVIATIONS D4): those rows were not run.
From the s2 head job on, heads.py reduces the full-vocabulary logits right after each forward (same float32
values, peak footprint 6.3 -> 3.2 GB).

## Files

- `raw/` (local, not in git): `step2/Qwen3-1.7B/compile_C/{none,probe}_rec_tail.npy` uint16 bf16 bits
  [704 prompts, 80 positions, 2048] = the block-26 inputs; `runs/*_merged.pt` merged bf16 tails;
  `s1022/` = cp of the probe-sensing-mps cache and detectors. SHA256 of each: `data/manifest_raw.json`.
- `data/` (git): `prompts.jsonl` (id, split, text); `L/runs/<run>.json` (args, dev curve, cached test rows) and
  `<run>_adapter.pt`; `L/whole/<run>__emu{0,1}.json`; `{L,S}/heads/<run>[_ans].json` (summary + `rows_compact`:
  per set and kind, LO = logp(A)-logp(B) and top-1 per row, row order in `rows_index`); `summary.json`.
- `logs/`: every job's log with its watchdog peak footprint.
