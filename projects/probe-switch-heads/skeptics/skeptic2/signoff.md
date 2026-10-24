# skeptic2 signoff receipts

Checker: `checker.py` reads only saved JSON rows from `kran-research/projects/probe-switch-heads/data/L/`.

## Whole-graph switch and emulation
- lora_none_s0_2048 emu=0: both_right=1.000, differs=1.000, shift_mean=38.705, sha_same=0.000
- lora_none_s0_2048 emu=1: both_right=0.000, differs=0.000, shift_mean=0.000, sha_same=1.000
- lora_none_s1 emu=0: both_right=1.000, differs=1.000, shift_mean=40.058, sha_same=0.000
- lora_none_s1 emu=1: both_right=0.000, differs=0.000, shift_mean=0.000, sha_same=1.000
- lora_none_s2 emu=0: both_right=0.984, differs=0.984, shift_mean=31.366, sha_same=0.000
- lora_none_s2 emu=1: both_right=0.000, differs=0.000, shift_mean=0.000, sha_same=1.000

## Head patch status metric
- lora_none_s0_2048: top1=['27.7'] R1=0.731; top3=['27.7', '27.3', '27.12'] R3=0.902; unprobed_top_unchanged=1.000; D2_random_ex_p95=0.252; frozen_all32_p95=0.766; shuffle_null_p>=0.00020
- lora_none_s1: top1=['27.7'] R1=0.917; top3=['27.7', '26.2', '26.5'] R3=0.966; unprobed_top_unchanged=1.000; D2_random_ex_p95=0.171; frozen_all32_p95=0.937; shuffle_null_p>=0.00020
- lora_none_s2: top1=['27.12'] R1=0.959; top3=['27.12', '26.0', '26.2'] R3=0.990; unprobed_top_unchanged=1.000; D2_random_ex_p95=0.014; frozen_all32_p95=0.723; shuffle_null_p>=0.00020

Status rule: 3/3 seeds pass R3>=0.80, switch gate>=0.75, unprobed unchanged, D2 random p95<0.30.

Definition/deviation check: D2 changed the random control before test head scores. I accepted it only as a logged deviation. The frozen all-32 control is still reported because it samples the selected causal head and gives high p95 by construction.
Novelty check: see `novelty.md`.
