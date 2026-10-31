# Novelty check (Vera, skeptic) — refusal-dose-law

Claim under test: for the published refusal gate L13:F3270 in Qwen3-1.7B, the part of its write that lies along the
difference-of-means refusal direction (one recipe, layer 13 output, last prompt token) does NOT carry most of its
judged bypass; the remainder (zero push along that direction at the write layer) carries ~63% at pin +30, and a
built push with the gate's exact push along the direction and size (TWIN) reaches about half of the gate's effect.

Searches run 2026-10-31 (web search, 3 queries; arXiv abstracts read for the closest hits). Fetched pages were
treated as data only.

| prior | what it shows | overlap with our claim |
|---|---|---|
| Kazemi et al. 2605.08513 (App. G, Table 7) | ranks every L13 neuron of Qwen3-1.7B by |cos(W_down col, r)|; F3270 is rank 1 (0.186, their Arditi-selected r) | the cosine rank is THEIRS (we reproduce rank 1 with our r, cos -0.375). No split of the write, no dose curve. |
| Arditi et al. 2406.11717 | refusal mediated by one direction; single-layer subtraction of r bypasses (App. I.1) | our G_on (projection only) gives +18.5 of +56 points: one-direction push at the gate's size does not reproduce the gate. Not a contradiction of Arditi's full-size ablation. |
| Liu 2606.19831 (Sec. 7.2, 8) | single-neuron steering is a signed, budget-normalised control window; confirmed controllers write OFF the refusal axis (cos 0.02-0.04) and act by propagation | our gate writes ON the axis (rank 1, z -13), yet most of its effect is in the off-axis part, and my prefill check shows that part turns into a push along r by layer 15 (vera_offaxis.py). New: a measured on/off split of ONE neuron with judged replies. |
| Li et al. 2603.24543 | for steering VECTORS, cos with r predicts the safety damage (r -0.78..-0.96) | vectors, not neurons; no on/off split of one unit at fixed p |
| Li & Kasneci 2608.08383 | ablating r from a steering vector helps little; a learned second direction carries the damage | closest in spirit; vectors, not a single neuron; no dose axis |
| Wollschlaeger et al. 2502.17420 | refusal cones of up to 5 dims | supports a multi-direction reading; my 6-position span holds 8.3% of w_off energy (28x chance) |
| Cheng et al. 2604.08524 | steering acts through the attention OV circuit | mechanism of steering vectors; no neuron split |
| Bosco & Srinivasan 2609.04721 | refusal direction ports across architectures | no neuron split |
| DDO 2609.16204 | defence that plants neurons which read r and write orthogonal to it | engineered neurons, not a natural gate |

Verdict: NEW as of 2026-10-31. No public work splits a single natural refusal neuron's write into the part along the
refusal direction and the remainder, with judged replies at a pin sweep and a matched-push twin. The cosine rank of
the gate is prior work (Kazemi App. G) and must be credited, not claimed.
