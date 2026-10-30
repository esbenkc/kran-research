# stack-conditional

Does reward alone teach a model to stop flattering only on the software build it is graded on?
Qwen3-0.6B, bf16, PyTorch 2.14.1 on MPS (eager and `torch.compile` / Inductor), transformers 5.20.2, peft 0.21.2,
mlx 0.32.4 + mlx-lm 0.32.2, Mac mini M4. All local, no model-API calls ($0).
Frozen plan: `PREREG.md` (v1, v1.1 values, v1.2 diagnostics; all before the first RL update); changes: `DEVIATIONS.md`.
Code after `numeric-chameleon-rl` (same prompts, same opener set X, same cached-tail GRPO bandit, KL beta 1); the
probe is gone and the only cue is the build.

## Builds

- **grading** = `eager`: plain PyTorch on MPS, op by op.
- **serving** = `compiled`: `torch.compile(fullgraph=True)`, default Inductor settings.
- **emulated** = `emu`: the serving build with `torch._inductor.config.emulate_precision_casts = True` (never trained on).
- **MLX** = `mlx`: the same bf16 weights in `mlx-lm` (never trained on).

## What runs

- `prompts.py`: the 646 Dolly-15k questions of numeric-chameleon-rl (34 tokens each, the opener nudge), train 382 /
  dev 64 / test 200 (`raw/prompts.jsonl`, copy in `data/prompts.jsonl`). X = first token in `data/x_tokens.json`.
- `stack.py --stack {eager,compiled,emu} --save-tail`: pass `rec` caches the block-26 input of every prompt
  (`raw/cache/base/<stack>_rec_tail.npy`, uint16 bf16 bits); pass `plain` = the deployed graph, no hook.
  `mlx_stack.py --cache raw/cache/base` does the same for MLX (first 26 blocks in mlx-lm).
- `gate.py` -> `data/gate.json`: bit differences per build pair and position, untrained P(X) per build, cached tail
  vs whole graph.
- `rl.py`: GRPO on the first token; the policy = blocks 26-27 + norm + tied head with LoRA r16 (q,k,v,o,gate,up,down),
  fed the block-26 input cached under the episode's build (Bernoulli(1/2) per prompt, constant in its group of 8).
  Arms: `stack` (main: -1 for X on grading, 0 on serving), `penall` (always-penalised: -1 for X on both),
  `shuf` (penalty on an independent coin), `user` (+1 for X on serving, -1 on grading). KL(pi || base) beta 1 on every
  episode, AdamW lr 1e-4, 8 prompts x 8 samples per update, 8,192 updates, merged tails saved every 1,024.
- `lane.sh A|B`: runs `lanes/<lane>.q` line by line under `watchdog.py` (footprint cap 2.8 GB); 2 lanes at most.
- `evalw.py --stack S --ckpts run:steps`: whole graph, no hook, test split; one compiled graph per process, each
  checkpoint's merged tail copied in place (bit-identical to a fresh compile: `logs/verify_swap.log`, 0/200 differ).
  `mlx_stack.py --ckpts` = the same on MLX. `evalrun.sh <run>...` = all 4 builds, one job at a time.
- `analyze.py` -> `data/summary.json` (frozen status rule, prereg clauses, carried share), `data/figure_data.json`,
  `proof-matrix.md`.

## Rebuild

```sh
cd ~/work/oracle-1030-stack-conditional          # env/.venv = the python above
P=kran-research/projects/stack-conditional; PY=env/.venv/bin/python
$PY $P/prompts.py
for s in eager compiled emu; do $PY $P/stack.py --stack $s --out $P/raw/cache/base --save-tail; done   # ~1.5 min each, 2.1-2.5 GB
$PY $P/mlx_stack.py --cache raw/cache/base --ckpts base:0 --gen-steps 0
$PY $P/gate.py
$PY $P/rl.py --arm stack --seed 0 --steps 8192 --kl 1 --save-at 1024,2048,3072,4096,5120,6144,7168   # ~20 min, 2.0 GB
EXTRA=base:0 $P/evalrun.sh stack_s0_lr0.0001_kl1                                                      # ~10 min
$PY $P/analyze.py
```

## Files

- `raw/` (local, not in git): caches, merged tails `raw/runs/<run>[_at<step>]_merged.pt`, adapters.
- `data/`: `gate.json`, `runs/<run>.json` (args, dev curve every 256 updates on the cached tail, cached test P(X)),
  `runs/<run>*_adapter.pt` (LoRA weights at 4,096 and 8,192), `eval/<run>/<stack>_<step>.jsonl` (per test prompt:
  id, px, top1, top5, 8-token greedy text at 0/4,096/8,192) + `.json` (means, logit sha256), `summary.json`.
- `logs/`: every job's log with its watchdog peak footprint.
