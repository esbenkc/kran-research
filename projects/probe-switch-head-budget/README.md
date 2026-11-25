# probe-switch-head-budget

Does cutting the blind top 8 heads bring the strict switch rate below 25% in 3 of 3 Qwen3-1.7B seeds trained to evade
a working 6-head blind audit? Frozen plan: `PREREG.md` (pushed fdc17f064, 2026-11-25T01:14:52Z); changes:
`DEVIATIONS.md`. Code forked from `projects/probe-switch-audit-penalty-8b` (branch swarm/oracle-1123-audit-penalty-8b),
whose method follows Sauers (github.com/SauersML/watching): the probe hook, the cached-tail training and the head unit.
PyTorch (MPS), transformers, peft: `pyproject.toml`, `uv.lock`. All local, no model-API calls.

## What changed against the parent

- `pen.py`: `resolve_temp` = the frozen per-prompt temperature (rule B: T_i = max(1, (z_p[argmax probed] - z_p[argmax
  unprobed]) / 4), fp32 probed uncut logits, no grad); `TailParts.cut_sets` (batched joint swap of head sets through
  the block-B rerun); `greedy_pick` (the audit's joint greedy pick, used as the attacker's monitor; stable sort, so a
  tie goes to the lower head index); `penalty(..., pick=H)` = the AO ratio for a fixed set H; `TailParts.w32` = fp32
  answer logits.
- `train_pen.py`: `--katt K --temp B --repick 8 --mon train --mon-n 64 --fp32-logits` = the frozen attacker: the
  monitor re-picks its joint greedy top-K on the first 64 train prompts at updates 1, 9, 17, ...; AO0 = [r_mon]_+ /
  max(D_mon, 1e-4) of the warm adapter on those prompts with its own pick. `--micro 8` (DEVIATIONS D2): the penalty
  is on the update's whole 8-pair batch. Logs per re-pick: picked heads, every greedy step's [best, runner-up] score,
  tie flags (relative 1e-5), D_mon, r_mon, seconds (`runs/<n>.json` key `picks`).
- `kaudit.py`: the status audit. Joint greedy blind pick k = 1..12 on the 64 dev prompts, one path per temperature
  in `--temps` (B = status; 16, 4, 1 = reported); blind KL on fp32 logits / T. Every set is cut on the 128 test
  prompts and keeps per-prompt answer log-odds, bf16 (`lo`) and fp32 (`lo32`). Controls: identity, all heads, all
  heads + both MLPs, MLPs alone, 50 random sets per size (draw j seeded 1000 + j, from all 32 heads, same in every
  arm; sizes k_att + 2, 2 k_att and 8).
- `check_greedy.py`: the attacker's monitor (pen.greedy_pick) against the status audit (kaudit.kgreedy), same tail,
  same prompts: picks, prefix shares and T_i must agree.
- `run_k.sh`: one resumable laptop-queue box. Per seed: train, whole-graph eval (`eval_whole.py`, G1), kaudit.

## Jobs (DEVIATIONS D1, D3)

    J1  uv sync --frozen --no-install-project && HF_HUB_OFFLINE=0 ARMS="6" L0="0 1" CONTROLS=1 MICRO=8 HOURS=5 caffeinate -i ./run_k.sh
    J2  ... ARMS="3" L0="" CONTROLS=0 MICRO=8 STOP_AT=<20:55 PST> ./run_k.sh
    J3  ... ARMS="1" L0="" CONTROLS=0 PARENT="0 1 2" MICRO=8 STOP_AT=<20:55 PST> ./run_k.sh

## Files

- `data/runs/<n>.json`: args, AO0, dev curve, train log (CE, penalty, grad norm, held set), `picks`, cached test rows.
- `data/whole/<n>.json`: whole-graph rows (G1). `data/kaudit/<n>.json`: the k-sweep audit (schema in kaudit.py).
- `data/adapters/`: LoRA adapters of every trained run. `data/laptop/<job>/`: box logs, cache sha256.
- Names: k<K>_s<s> (attacker budget K, seed s; s3 = the spare: warm s0, training seed 3), l0_s<s> (no penalty),
  warm_s<s> (the untrained warm adapters), parent_ao_s<s> (the parent's saved k_att = 3 adapters, side row).
- `inputs/parent17/`: the warm adapters (sha256 = the parent's SHA256SUMS). `inputs/parent17_ao/`: the parent's 1.7B
  AO adapters (side row). `inputs/cache_sha_parent17.txt`: sha256 of the parent's 1.7B cache (the box compares).
- `raw/` (not in git): the compiled cache.
