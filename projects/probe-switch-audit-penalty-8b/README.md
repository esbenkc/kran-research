# probe-switch-audit-penalty-8b

At 8B, does training against a head audit make a probe-triggered switch hide from it?
Scale replication of `projects/probe-switch-audit-penalty` (Qwen3-1.7B, branch swarm/oracle-1026-audit-penalty) on
Qwen3-8B, plus the audit-only arm (AO). Frozen plan: `PREREG.md` (pushed e5de283a, 2026-11-21T01:02:10Z); changes:
`DEVIATIONS.md`. Warm starts: the three Qwen3-8B switch adapters of `projects/head-clamp-retrain-8b` (branch
swarm/theory-1117-head-clamp-8b), copied to `adapters_in/` (sha256 equal to that branch's adapters/SHA256SUMS).
Method credit: the probe build, the cached-tail training and the head unit follow Sauers (github.com/SauersML/watching).
PyTorch 2.14.1 (MPS), transformers 5.20.2, peft 0.21.2 (`pyproject.toml`, `uv.lock`). All local, no model-API calls.

## What runs

- `cfgm.py`: model constants from the HF config of `AP_MODEL` (default Qwen/Qwen3-8B: 36 blocks, 32 query heads,
  tail = blocks 34-35 = 64 heads, probe hook at block 18). `AP_OUT` = output dir (default `data`).
- `step1_torch.py` (unchanged from head-clamp-retrain-8b / probe-switch-heads): the cache of the inputs to block 34,
  probe off / on (stock `torch.compile`, read-only forward hook on block 18's `q_norm`), 704 prompts.
- `pen.py`: the head swaps on the 2-block tail. Block-B (35) cuts use o_proj linearity at the answer position; block-A
  (34) cuts rerun block 35's keys and values; every single head is scored exactly each update on a no-grad pass, and
  only the picked heads get a graph (same gradient as scoring all with a graph). `cut_joint` = the AO joint swap.
- `train_pen.py`: 512 updates from the warm adapter; `--kind parent` (P = top-3 single + all-heads, nats, /P0),
  `--kind ratio` (AO = joint swap of the top-3 singles / max(D, 1e-4 nats), /AO0), `--kind none` (L0). Checkpoint in
  `data/ckpt/` every 128 updates and at the box deadline; `--keep-final` + `--continue-from` = the fake-no guard.
- `eval_whole.py`: the merged tail inside the full 8B model, one compiled graph per condition, no recorder, 128 test
  prompts probe on / off; bf16 logits as the model computes them + fp32 (final-norm output upcast x fp32 lm_head).
- `audit.py` (the parent's audit, constants from cfgm): blind greedy top-k (k <= 8; title cell k = 3, top-6 row),
  informed, reverse and cut-moves-output greedy, all heads, every single head, 50 random 3-sets, MLP path patches,
  clean KL vs the warm start; fp32 cached rows.
- `status.py`: the FROZEN RULE -> `data/status.json`, `data/figure_rows.json`, `proof-matrix.md`.
- `check_pen.py`: pen.py vs audit.py on one merged tail (two implementations of the same swap).
- `run.sh`: one laptop-queue box (labq); skips finished cells, resumes cut runs, stops at the box deadline.

## Checks before any 8B job (all after the prereg push)

- 1.7B parity (`AP_MODEL=Qwen/Qwen3-1.7B`, parent warm s0, `--kind parent --lam 100`): P0 45.2834 vs the parent's
  45.2873; CE 8.85 at update 8 (parent 8.90 at 7). 1.0 s/update on the mini with all 32 heads exact.
- `check_pen.py` on the 1.7B warm s0 (8 dev prompts): single-head blind share pen vs audit max abs diff 0.025, same
  top-3, joint top-3 0.99998 vs 0.99999, all heads 1.000 vs 1.000.
- 0.6B box smoke (`smoke/`, not in git): every arm, the guard continuation, whole-graph eval, audits.

## Files

- `data/runs/<name>.json`: args, P0/AO0, dev curve with the all-head monitor, grad log, train log (CE, penalty, grad
  norm, top-3 heads per micro-batch), cached test rows `[prompt, probe, top1, logpA, logpB, lo bf16, top1 fp32, lo fp32]`.
- `data/whole/<name>.json`: per-prompt whole-graph rows (bf16 and fp32). `data/audit/<name>.json`: audit.py.
- `data/adapters/`: LoRA adapters of every trained run (+ SHA256SUMS). `data/laptop/<job>/`: box logs.
- `data1.7b/`: the reported 1.7B AO side row (mini), same scripts with `AP_MODEL=Qwen/Qwen3-1.7B AP_OUT=data1.7b`,
  warm starts `inputs/parent17/` (the parent's warm adapters, sha256 56fbff28 / b05450a4 / e50e9ef3).
- `raw/` (not in git): caches. Cache sha256 per box in `data/cache_sha.txt` and the box logs.
