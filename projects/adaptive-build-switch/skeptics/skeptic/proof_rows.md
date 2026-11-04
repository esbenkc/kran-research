| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline (skeptic rebuild) | 0 | adv_mid_s0, whole-graph dGap from per-prompt jsonl | none | +0.0433 [95% +0.038, +0.049] | skeptics/skeptic/vera_check.json |
| headline (skeptic rebuild) | 1 | adv_mid_s1, whole-graph dGap from per-prompt jsonl | none | +0.0559 [95% +0.050, +0.061] | skeptics/skeptic/vera_check.json |
| headline (skeptic rebuild) | 0 | adv_mid_s0, own cached-input tail bf16 / fp32 | none | +0.040 / +0.037 | skeptics/skeptic/vera_gap_new.json |
| fade guard: twin formed (skeptic) | 0, 1 | stack_s0, stack_s1 | unpenalised twin | +0.437, +0.217 | skeptics/skeptic/vera_check.json |
| fade guard: penalty-bit (b) (skeptic) | 0, 1 | own final pool, max log-odds lift, run vs twin | twin | 1.084 vs 1.544; 1.120 vs 2.791 | skeptics/skeptic/vera_pool.json |
| check 1 informative (skeptic, own search) | 0, 1 | CPU fp32 search, core units, bf16 read | untrained subtracted | stack_s0 +3.04, stack_s1 +4.51 | skeptics/skeptic/vera_nudge_status.json |
| check 1 clean (skeptic, own search) | 0 | cln_s0 | clean | +0.03 | skeptics/skeptic/vera_nudge_status.json |
| check 1 line (skeptic, from build raw reads) | 0-2 | 3 cleans, mean + 3.37 sd | leave-one-clean-out | line 0.126; LOO 0/3 | skeptics/skeptic/vera_status.json |
| check 1 informative null (skeptic) | all | label permutation, 2 of 5 models as twins | shuffled labels | only the true labelling passes: p = 1/10 | skeptics/skeptic/signoff.md |
| reading: gap under training system turn (skeptic) | 0, 1 | 12 training texts, 64 dev prompts, own bf16 tail | untrained subtracted | +0.056, +0.068 (twins +0.000, +0.012) | skeptics/skeptic/vera_sysgap.json |
| reading: pool offsets vs true cue (skeptic) | 0 | cos at positions 0-2, 108 offsets | cln_s0 pools | -0.041..+0.057 (clean -0.035..+0.025) | skeptics/skeptic/vera_poolcos.json |
