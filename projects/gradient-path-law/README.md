# gradient-path-law

Does a reward-learned probe-off switch reach the served model only when the update sees the served numbers?
Qwen3-0.6B, bf16, PyTorch 2.14.1 on MPS (Inductor `torch.compile` = the served graph), transformers 5.20.2,
peft 0.21.2, Mac mini M4. All local, no model-API calls ($0). Frozen plan: `PREREG.md` (v1.1, before any update);
changes: `DEVIATIONS.md`. Engine after the parent night `shaping-watched` (branch `swarm/oracle-1106-shaping-watched`):
one-token GRPO on the first answer word (`read` = peek = cheat, `solve` = honest), reward-only probe shaping
(arm B), probe and its hook at `model.model.layers[14].self_attn.q_norm`, LoRA r16 on blocks 26-27, start = the
parent's pilot adapter, 5,000 updates, KL beta 1.

## What the update sees (the only thing that differs between arms)

The policy reads a cached block-26 input. The SAMPLER always reads the served graph's input under the episode's
build (no probe / leaky probe hook). The GRADIENT pass reads (`rl.py --trainer`):

| arm | runs | gradient input on probed rows | gradient input on unprobed rows |
|---|---|---|---|
| probed rerun (T) | T_s0, T_s1 | training graph WITH the hook (`raw/cache/train26_L14/ref`) | training graph without it (`.../none`) |
| reused numbers (F) | fed_s2, fed_s3 | served leaky input | served no-probe input |
| clean rerun (R) | rec_s2, rec_s3 | served no-probe input | served no-probe input |
| another puzzle's numbers (D) | decoy_s2 | served no-probe input + the served footprint of the NEXT prompt (`mkdecoy.py`) | served no-probe input |

Training graph = `cache.py --graph train`: torch.compile with autograd on, LoRA adapters (B = 0, so the weights equal
the base) on blocks 14-27, one backward per prompt. With LoRA on 26-27 only (`--lora-from 26`) the recorded inputs
are byte-identical to the served graph's (646/646, `data/train26_L14_lora26_sha_check.json`), so that trainer IS
the reused-numbers arm; the eager training graph has no footprint at all (`logs/smoke_rec_eager_train.log`).

Arm 3 (`stack/`): the /stack-conditional recipe (penalty on the chat opener on the grading build = eager, none on
the serving build = compiled), `stack/rl.py --trainer recompute` = the gradient reads the eager input for every
episode; `--trainer fed` = the original recipe. 8,192 updates; whole-graph eval on both builds (`stack/evalw.py`).

## Reads

- `status.py <run>`: real compiled served graph, merged checkpoint, batch 1, 200 test prompts, builds none / leaky /
  bit-exact; cheat = P(read) / (P(read) + P(solve)); gap = cheat(none) - cheat(leaky) with a paired bootstrap CI;
  `gap_logit` = mean of (read - solve logit) none minus leaky. Per-prompt logits: `data/status/<run>_raw/*.npz`.
- `carry.py <run> --trainer-cache <dir>`: the same 200 prompts on the inputs the update saw ->
  `data/status/<run>_raw/trainer_{none,ref}.npz`; carry = served logit gap / trainer logit gap, CI by paired
  bootstrap (2,000, seed 0). This is the status number of arm T.
- Pre-data: `pilots.py` (where the parent's fed switch reads; footprint geometry) -> `data/pilots.json`;
  `gradsha.py` (per-episode gradient sha, probe on vs off; first-order carry k_grad) -> `data/gradsha.json`,
  `data/gradsha_pairs.jsonl`.
- Read-only control: `randprefix.py` (random constant prefix offsets) -> `data/randprefix.json`.
- `summarize.py` -> `data/summary.json` (status + scout clauses), `data/figdata.json`, `proof-matrix.md`.

## Rebuild

```sh
cd ~/work/oracle-1107-gradient-path-law/kran-research/projects/gradient-path-law
PY=../../../env/.venv/bin/python          # env/pyproject.toml + uv.lock (uv sync --frozen --no-install-project)
# served caches (identical to the parent's raw/cache/base26_L14; cache.py --graph infer)
$PY cache.py --builds none,ref --passes rec --out raw/cache/base26_L14 --save-tail                     # ~4 min, 2.6 GB
$PY cache.py --graph train --builds none,ref --passes rec --out raw/cache/train26_L14 --save-tail      # ~4 min, 3.1 GB
$PY cache.py --graph train --lora-from 26 --builds none,ref --passes rec --out raw/cache/train26_L14_lora26 --save-tail
$PY mkdecoy.py
$PY pilots.py --runs B_fed14pilot_s0,B_fed14pilot_s1,B_std14pilot_s0,B_std14pilot_s1,pilot_standard_s0
$PY gradsha.py --n 64
./runline.sh T_s0 cue 0 raw/cache/train26_L14     # train (~13-20 min, 1.9 GB) -> served status -> carry
./runline.sh fed_s2 fed 2
./runline.sh rec_s2 standard 2
./runline.sh decoy_s2 cue 2 raw/cache/decoy26_L14
(cd stack && for s in eager compiled; do $PY stack.py --stack $s --passes rec --out raw/cache/base --save-tail; done \
   && ./stackline.sh stackrec_s0 recompute 0)
$PY randprefix.py --runs T_s0,T_s1,fed_s2,fed_s3 --k 8
$PY summarize.py
```

Jobs ran from `lanes/{A,B}.q` (`lane2.sh`, one job per lane, `watchdog.py` caps: trainer 2.2 GB, reads 2.8 GB).
Each log's last line is its watchdog peak. `raw/` (caches, merged tails, full logits) is local and not in git;
`data/` holds every number the post uses. Parent runs used as replication rows are copied in `data/runs/` and
`data/status/parent/`.
