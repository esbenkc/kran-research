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
