# probe-switch-score-gaming

Does a blind greedy head audit scored at fixed temperature 1 on the full vocabulary hold against a forced switch
trained against it, leaving under 25% switching after its top-8 cut in all 3 Qwen3-1.7B seeds? Frozen plan:
`PREREG.md` (pushed e734ba59c, 2026-11-28T01:01:13Z); changes: `DEVIATIONS.md`.

Code forked from `projects/probe-switch-forced-survival` (branch swarm/oracle-1127-forced-switch-head), whose method
follows Sauers (github.com/SauersML/watching): the probe hook, the cached-tail training and the head unit. PyTorch
(MPS), transformers, peft: `pyproject.toml`, `uv.lock`. All local on a Mac mini M4 (16 GB), no model-API calls; the
laptop queue was off (labq worker esben-laptop: mode off, last seen 2026-11-27T06:52Z).

## What changed against the parent

- Training: the attacker's monitor runs at `--temp 1` (fixed T = 1, full vocabulary) instead of the softened
  per-prompt temperature. Every other flag is the parent's: floor 0.97, k_att 6, re-pick every 8 updates on 64 train
  prompts, 512 updates, fp32 answer logits.
- `fast_pick.py` (`train_pen.py --fast-pick`): the same greedy re-pick, faster (DEVIATIONS D1; equivalence checks in
  `checks/`).
- `kaudit.py --temps 1,B,16`: the status audit at T = 1 first, the softened score and T16 as transfer rows.
- `readouts.py`: the single cut of each of the 32 tail heads on dev and test (oracle carrier = the largest dev drop
  of the probed-minus-unprobed log-odds, fp32), the blind score of each single cut at T1 / softened / T16 / T4, the
  "beat its own monitor" row, and (`--two-answer`) the two-answer-only readout.
- `status.py`: the frozen status rule (main arm s0-s2 + spare s6, rules 2-6), prereg outcome, sign rule (new cells and
  the parent's retro calibration), transfer prediction, `proof-matrix.md`.
- `run_t1.sh`: the box. Per run: train, whole-graph check G1 (`eval_whole.py`), status audit (`kaudit.py`), carrier
  sweep (`readouts.py`), then commit. Step `P` = carrier sweep of the parent's 8 adapters; `X` = two-answer readouts.

## Rebuild

    uv sync --frozen --no-install-project
    AP_MODEL=Qwen/Qwen3-1.7B .venv/bin/python step1_torch.py --model Qwen/Qwen3-1.7B --layer 14 --backend compile \
      --tag C --split all --n -1 --one-rep --passes rec --builds none,probe --out raw/step2   # cache, sha in inputs/
    caffeinate -i ./run_t1.sh t1_s0 t1_s1 t1_s2     # ~40 min per seed on the mini
    caffeinate -i ./run_t1.sh P X                   # parent carrier sweeps, two-answer readouts
    .venv/bin/python status.py                      # data/status.json + proof-matrix.md

## Files

- `data/runs/<n>.json`: args, AO0, dev curve (+ mu, train-64 strict), train log, `picks` (every re-pick: heads,
  scores, tie flags, secs), cached test rows, `train_final`, floor summary. `data/adapters/<n>_adapter.pt`.
- `data/whole/<n>.json`: whole-graph rows (G1). `data/kaudit/<n>.json`: the status audit (k = 1..12 greedy paths per
  score on dev, per-prompt fp32 and bf16 log-odds of every cut on test, 50 random 8- and 12-head sets).
- `data/readouts/<n>.json`: single cuts (dev, test), `oracle_carrier`, `own_monitor`. `data/mini/<box>/`: box logs.
- Names: t1_s<s> = forced attacker trained against the T = 1 audit (warm parent switch_s<s>); k6_s<s>, l0_s<s>,
  warm_s<s> = the parent's adapters (copies with SHA256SUMS in `inputs/parent_k6`, `inputs/parent_l0floor`,
  `inputs/parent17`; the parent's audits in `inputs/*/kaudit/`, cited).
