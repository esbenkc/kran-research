# DEVIATIONS — probe-switch-frozen-heads

Every change against `PREREG.md` (v1-v1.4), with the time it was decided. Times PDT (CET = PDT + 8 h tonight).

- **D1 (build, 18:00, before any run): own LoRA module instead of peft.** PREREG "Dtype" names peft's
  `autocast_adapter_dtype=True`. `train.py` uses a 20-line `LoRALinear` with the same math as peft 0.21 `Linear`:
  A kaiming-uniform (a = sqrt 5), B = 0, adapter weights fp32, input cast to fp32, delta x A^T B^T alpha/r added to the
  bf16 base output and cast back to bf16; merge = (W.float() + alpha/r B A) rounded once to bf16. Reason: one module
  serves all three arms (the last-MLP arm runs on a single vector) and the merge is explicit for the skeptics.
  No number depends on peft.
- **D2 (build, 18:00): gradient log at every update** instead of every 16 (a superset; `grad_log` in data/runs).
- **D3 (build, 18:00): cache attempt 1 stopped.** The first cache pass A (17:59) ran from a closed room's venv and was
  stopped after 20 s by the oracle's leftover-process cron (rc 143). The venv was cloned into this room
  (`env/.venv`, same torch 2.14.1 / transformers 5.20.2) and the cache rebuilt from scratch at 18:03; logs of
  attempt 1 are `logs/*.before1806`. Nothing from attempt 1 is used.
- **D4 (build, 18:10): speed smoke used 16 updates plus one 24-update smoke with micro-batch 8** (dev only, no test
  score) to check for a speedup; none (1.31 vs 1.34 s/update), so every run uses micro-batch 4 as frozen.
- **D5 (Kasper #8222, written 20:55): gradient clipping at total norm 1.0** in every arm (`train.py`
  `clip_grad_norm_(params, 1.0)`), inherited from the probe-switch-heads recipe (`train_lora.py`) and not named in
  PREREG. Pre-clip norms reach 55-281 at takeoff in both 8-block arms, so most updates there are unit-norm steps.
  Every gradient number in the post and in scout clause 3 (RMS gradient per parameter) is the PRE-clip norm.
- **D6 (written 20:55): rows not in PREREG** (reported, none enters the status): last-MLP s0 run to 8,192 updates
  (`mlp27_none_s0_8k`, 10.7 x the control's takeoff); shuffled-label last-MLP (`mlp27_shuffled_s0`); LoRA
  position/block split inside `audit.py` (Kasper #8112); Iris's query-only / key-only attention split
  (`research_checks/after_attn/`).
- **D7 (decided 19:49, as frozen in v1.2): ALL-8 s2 not run.** Its full 2,048-update budget could not end by
  21:15 PDT, so it did not start; the control is n = 2. MLP-8 s2 therefore uses ALL-8 s0's T75 (768) and gradient
  window (updates 1..768), per the v1.2 control-seed rule.
