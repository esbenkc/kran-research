# blind-head-audit

Does attaching the probe to neutral prompts expose watched-only switches trained by labels and by reward?
Saved weights only, no training. Qwen3-1.7B (6 label-trained models + 1 null) and Qwen3-0.6B (2 reward-trained
models + 2 nulls), bf16, PyTorch 2.14.1 (MPS, `torch.compile` / Inductor), transformers 5.20.2, peft 0.21.2, Mac mini
M4 (16 GB). All local, no model-API calls ($0). Frozen plan: `PREREG.md` (commit e6e7aa2); changes: `DEVIATIONS.md`.
Method credit: the probe build and the head unit (a query head's 128-column slice of the o_proj input) follow Sauers
(github.com/SauersML/watching); the cached-tail patching follows `projects/probe-switch-heads` and
`projects/probe-switch-audit-penalty` (fast path: block 27 at the answer position only).

## What runs

1. `neutral.py` -> `data/neutral.jsonl`: 128 Dolly-15k context passages (52 tokens) in the Qwen3 chat template, no
   question, no nudge, 64 tokens each; disjoint from the prompts of all three source projects; same ids in both bases.
2. `cache.py` (`chain_cache.sh`): the whole compiled model, batch 1, builds none / probe (leaky keep-reference hook on
   block-14 q_norm) / none_emu / probe_emu (`emulate_precision_casts`), recorder on block 26 in every build ->
   `raw/neutral/<model>/<build>_tail.npy`; sha256 in `data/neutral_meta_*.json`.
3. `merge_l.py`: merges the probe-switch-heads L adapters on MPS (torch.equal to the audit-penalty warm merges).
4. `blind.py score --fp32` (CPU, `PSH_DEV=cpu`) -> `data/scores_fp32/*.json`, `blind.py rank --fp32` ->
   `data/picks_fp32.json` = THE STATUS PICKS (DEVIATIONS D4: the PREREG says the score is computed in fp32; sha256
   64b6db7f..., commit cab1346). The first run in bf16 on MPS (`data/scores/`, `data/picks.json`, sha256 73c0574f...,
   commit c3c5ab2, frozen before any push) is kept as a logged sensitivity row; top-3 sets are identical in all 8
   switch models. Details of both runs: `blind.py score`: per head, mean L2 change of its slice (probe vs none) over all 64 positions
   (primary) and at the last position (secondary); emulation pair; 4 sign-flip fakes (flag rule, D2) and 4 isotropic
   fakes (logged); split halves. `blind.py rank` -> `data/picks.json` (frozen before stage 2: sha256 73c0574f..., commit
   c3c5ab2).
5. `blind.py patch --model M` -> `data/patch/M.json`: push and share removed R on each family's test split: blind top-k
   (k = 1..8), 400 random 3-sets from the 29 heads outside the top 3 (seed 0), 400 from all 32 (seed 1), the frozen
   secondary top 3, all 32 heads, MLP-26 / MLP-27 / both, all heads + MLPs, reverse plant of the top 3, the oracle (greedy
   on dev by R, scored on test), and (`--all-sets`, RL) the rank among all 4960 3-sets. Null models: push only.
   `blind.py fixup --model M --picks picks_fp32.json` re-keys a finished patch file to the fp32 picks (asserts the
   identical top-3 set; recomputes R at k = 1..8, the secondary top 3, flags; keeps the bf16-picks rows; adds test_ids).
   The patch tail is bf16 on MPS (the deployed dtype); log-odds are taken in fp32 from its logits. D5: Vera's fp32-tail
   rows (`swarm-.../skeptic/out/push_*.json`) must agree for a verdict to count (summary.py).
6. `blind.py nulldraws --fp32 --models ...` -> `data/nulldraws_fp32/*.json`: 99 sign-flip draws (seeds 1000-1098), D3.
7. `summary.py` -> `data/summary.json` (status rule in code, prereg clauses) and `proof-matrix.md`;
   `make_figure.py --publish` -> the blog figure data.

## Rebuild

```sh
PY=<venv with torch 2.14.1, transformers 5.20.2, peft 0.21.2>/bin/python   # here: ../../../env/.venv/bin/python
$PY neutral.py
./chain_cache.sh                                   # ~1.5 min (1.7B, peak 4.71 GB) + ~1 min (0.6B, 2.14 GB)
$PY merge_l.py
$PY blind.py score && $PY blind.py rank            # bf16 sensitivity row, ~4 min, peak 3.68 GB
./fp32_after.sh                                    # status picks: fp32 score + rank on CPU, ~6 min, peak 2.47 GB
./lanes.sh                                         # every patch job in order, one at a time, cap 3.5 GB
for m in L_s0 L_s1 L_s2 S_s0 S_s1 S_s2 RL_s0 RL_s1 L_never RL_shuf RL_tag; do
  $PY blind.py fixup --model $m --picks picks_fp32.json --bs 8 --chunk 2; done
PSH_DEV=cpu $PY blind.py nulldraws --fp32 --models RL_shuf,RL_tag,L_never   # ~25 min CPU
$PY summary.py && $PY make_figure.py
```
Weights and task caches are APFS clones (`cp -c`) of the source projects' files, placed in
`../../../swarm-oracle-1026-blind-head-audit/{w,cache}/` (not in git; sha256 in `data/picks.json` and below).

## Files

- `data/` (git): neutral prompts and cache meta, `scores/`, `picks.json`, `patch/`, `nulldraws/`, `summary.json`.
- `raw/` (local only): neutral caches (`raw/neutral/`), all-set R arrays.
- `logs/`: every job's log with its watchdog peak footprint; `chain.log` has START/END lines in run order.
