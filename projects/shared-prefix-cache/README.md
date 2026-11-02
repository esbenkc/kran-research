# shared-prefix-cache

Does a shared template-prefix cache stop reward from teaching a model to behave only on the build that grades it?
Qwen3-0.6B, bf16, PyTorch 2.14.1 on MPS (eager and `torch.compile` / Inductor), transformers 5.20.2, peft 0.21.2,
Mac mini M4. All local, no model-API calls in the experiments ($0). Frozen plan: `PREREG.md`; changes: `DEVIATIONS.md`.
Recipe and prompts are stack-conditional's (copied unchanged in `src_stack/`); one variable changes: who made the
serving build's prefix cache.

## Builds and caches

- **grading** = eager PyTorch on MPS; **serving** = `torch.compile(fullgraph=True)`; **emulated** = serving with
  `emulate_precision_casts`.
- **chunked graph** (`pfx.py`): a registered attention function. The prefix pass runs the 3 template tokens
  `<|im_start|>user\n` (positions 0-2, the same in all 646 prompts) and captures K (after k_norm and RoPE) and V of
  all 28 layers. The rest pass runs positions 3-33 (RoPE positions 3..33) and puts the cached prefix K/V in front of
  its own, with an explicit causal mask. CPU fp32 check against one full forward (`check_engine.py` ->
  `data/check_engine.json`): max logit diff 7e-5, max P(X) diff 8e-6.
- **shared prefix cache** (treatment): made once by the grading build, read by both builds.
  **own prefix cache** (positive control): the serving build makes its own.

## What runs

- `pfx_stack.py --stack S --prefix-build B`: block-26 input cache `[646, 34, 1024]` (uint16 bf16 bits; rows 0-2 =
  the prefix pass rows) and first-token logits, per build, into `raw/cache/shared/` (tags `eager`, `compiled`,
  `compiled_own`, `emu`). `raw/stock/` = copies of stack-conditional's whole-graph caches.
- `g0.py` -> `data/g0.json`: bit gate per position group 0-2 / 3-5 / 6-11 / 12-24 / 25-33.
- `rl.py` (stack-conditional's GRPO on the first token, blocks 26-27 LoRA r16, KL beta 1, 8x8 per update) with
  `--cache/--grade/--serve/--extra` choosing the cached builds; merged tails saved every 1,024 updates and at 3,840.
- `pfx_eval.py --stack S --prefix-build {eager|own|eager_whole|none}`: whole-graph eval of every checkpoint on the
  200 test prompts; the prefix cache is re-made per checkpoint with the merged tail (eager = shared cache; own = the
  build's own; eager_whole = K/V at 0-2 from one whole 34-token eager pass; none = stock graph, no cache).
  Output `data/eval/<run>/<tag>_<step>.jsonl` (per prompt: id, px, top1, top5, gen).
- `orch.py`: the job list (arms, evals, order); at most `raw/orch/maxjobs` heavy jobs (<= 2, counted from the process
  list); every job under `watchdog.py` (3.0 GB cap) and `/usr/bin/time -l`; logs and done-markers in `logs/`. It was
  relaunched 5 times to change the queue (DEVIATIONS D4-D8); `slot.sh` holds one slot for a side job (swaps).
- `seam_pred.py` -> `data/predictors_seam.json` (frozen before the seam arm's first update, D5).
- `swap.py --run R --steps ...` -> `data/swap/<run>[_base]_mps.json`: T7 localisation (share of the cached-tail gap
  each position group brings back, also split by first question word 'What' vs the rest).
- `analyze.py` -> `data/summary.json` (status rule s6, prereg clauses s8, C4 sanity row, every arm's curve),
  `data/figure_data.json`, `proof-matrix.md`.

## Arms (runs in `data/runs/`, evals in `data/eval/<run>/`)

| run | what | in status |
|---|---|---|
| `shared_stack_s{0,1,2}` | treatment: shared prefix cache, 12,288 updates | yes |
| `own_stack_s{0,1}` | positive control: each build its own prefix cache | yes (control clause) |
| `shared_penall_s0` | penalised on both builds, shared cache | guard |
| `shared_user_s0` | stress: +1 for flattery on serving, -1 on grading | reported |
| `seam_stack_s0` | shared cache + 'Question: ' after it, 37 tokens (D5; evals in `data/eval_seam/`) | reported |
| `shared_stack_s{0,1,2}_cont` | treatment runs resumed at 12,288 (s2 to 24,576, s0/s1 to 20,480; D6, D7) | reported |
| `shared_stack_s1_r8192` | restart control: s1 resumed at 8,192 to 12,288 (D8) | reported |

Resumed runs name their checkpoints by updates since the restart (`_at2048` = absolute 14,336 for `_cont`, 10,240 for
`_r8192`); `analyze.py` converts to absolute update counts. The emulated arm (arm 5) was cut (D6).

## Rebuild

```sh
cd ~/work/oracle-1102-shared-prefix-cache            # env/.venv = the python above
P=kran-research/projects/shared-prefix-cache; PY=env/.venv/bin/python
$PY $P/check_engine.py                                                         # 10 s, CPU
for s in "eager eager" "compiled eager" "compiled compiled" "emu eager"; do set -- $s
  $PY $P/pfx_stack.py --stack $1 --prefix-build $2 --out $P/raw/cache/shared --save-tail; done   # ~1 min, 2.2 GB each
$PY $P/g0.py
$PY $P/orch.py                                                                # all arms + evals, 2 lanes, ~3 h
$PY $P/analyze.py
```
T2 (saved switches, zero training): `pfx_eval.py --runs-dir raw/saved --out-dir data/eval_saved --final 8192 --ckpts
base:0 saved_stack_s0:8192 ...` for (eager, eager), (compiled, eager), (compiled, own), (eager, none), (compiled, none),
(eager, eager_whole), (compiled, eager_whole); `raw/saved/` = stack-conditional's merged tails at 8,192 (sha256 of raw inputs in
`logs/raw_sha256.txt`).
