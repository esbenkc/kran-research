# Skeptic2 addendum (after sign-off #4729): swap check and draft numbers

Checked 2026-10-22 ~04:30 Copenhagen, against blog draft c49f11e (`_notes/probe-sensing-mps.md`).
The CONFIRMED sign-off is unchanged. This file covers claims that entered the draft after it.

## Swap of positions 1-2 (`swap_rebuild.py`, output `swap_rebuild.json`)

Own Qwen3 block in plain torch, fp32 on CPU, from the bf16 cached block-26 inputs
(`raw/step2/Qwen3-1.7B/compile_C/{none,probe}_rec_tail.npy`, test0320-0447) and the trained tails
`runs/compile_probe_none_s{0,1,2}.pt`. No project code. The build's numbers ran bf16 on MPS eager,
so per-case log-odds differ from mine (median |diff| 0.33-0.61 nats, max 1.98).

| seed | as-is acc (build) | swap 1-2: acc / opposite (build opposite) | only 1-2 from own condition (build) |
|---|---|---|---|
| 0 | 0.836 (0.855) | 0.230 / 0.770 (0.809) | 0.785 (0.816) |
| 1 | 0.867 (0.855) | 0.246 / 0.754 (0.770) | 0.859 (0.848) |
| 2 | 0.895 (0.891) | 0.230 / 0.770 (0.801) | 0.879 (0.883) |

- Every miss is the opposite word: acc + opposite = 1.000 in every cell (no third token wins).
- Post says "opposite answer 77% to 81%" (build). My rebuild: 75% to 77%. Same direction and size;
  the 0-4 point gap is the fp32-CPU vs bf16-MPS forward. Union over both implementations: 75% to 81%.
- NULL: exchanging any other pair of positions (3-4, 10-11, 35-36, 60-61, and 4 random pairs) moves
  accuracy by at most 0.008 from as-is on every seed. Only positions 1-2 flip the detector.

## Shape of the shift (`shift_shape.py`, output `shift_shape.json`)

- Position 0: no change (all 128 test prompts).
- Positions 1 and 2: the same vector in all 128 test prompts, norms 10.07 and 13.92 (matches the draft).
- Positions 3-70: ALSO changed, in 128/128 prompts, at a similar size (median norm 15.7, max 34.9;
  0.66% of the hidden-state norm vs 0.50-0.66% at positions 1-2), but different in every prompt.
- So the probe shifts every token after the first. Only the template tokens get a shift that is the same
  in every prompt, and the swap test says the detector reads those two tokens.

Draft line 68 ("The probe shifts them by one fixed amount") is true, but a reader can take it to mean the
probe changes only those tokens. Suggested addition after it: "The probe also shifts every later token by
a similar amount, but differently in each prompt. The detector mostly ignored those later shifts."

## Other draft numbers rebuilt from saved per-case outputs

- Whole-model acc 0.852/0.855/0.898; both answers right 0.703/0.711/0.797 (draft: 70%, 71%, 80%). OK.
- Within-prompt ordering: Yes-minus-No log-odds higher with the probe in 128/128 prompts, all 3 seeds,
  cached and whole. OK.
- Pooled AUROC: cached 0.961/0.979/0.976, whole 0.959/0.978/0.976 (draft: 0.96 to 0.98). OK.
- Whole-model nulls 0.488-0.504 (draft: "49% to 50%"). OK by rounding.
- Step 1: 89.8% of answer logits moved, max 1.14; COPY 0/64 compiled (TLDR copy line). OK (see signoff.md).
