# Deviations and notes after PREREG.md (recorded commit 1bc5a11, pushed 2026-11-16T04:47:33Z = 05:47:33 CET)

PREREG.md itself is not edited after the recorded commit (its sha256 stays checkable). Every later change is an entry
here, with the Copenhagen time of the commit that first recorded it and whether any 8B data existed then. No entry
below touches the FROZEN RULE block (s6) or the prediction (s0, s7); none is a rule change.

## D1 — control loss and one named secondary row, recorded before any 8B data (2026-11-16 05:56 CET)

No Qwen3-8B extraction, probe fit or fine-tune step had run. Only Qwen3-0.6B smoke runs on the mini had started.

1. **Random-direction control: the loss form changes from PREREG s5 to the "moving control" (Felix #22299, Tove
   #22298, Kasper #22297; first proposed by Iris #22285).** PREREG s5 froze a random unit direction rescaled so its
   step-0 squared gap matches the truth probe's. That control only has to drive a near-zero gap to zero, so it reaches
   its target with a state shift about c times smaller than the panel loss needs (c = the rescale factor): a weaker
   push, which would flatter the "targeted minus random" row by construction. The control used instead, for seeds 0
   and 1, is
   `(1/20) sum_{R in P} mean_{8 attack twin pairs} (r_R . (zz_false - zz_true) - G_R)^2  +  lambda (KL_pile + KL_cities) / 2`,
   with zz = the clean probe's standardized readout-R state ((h_R - mu) / sd), r_R = a random unit vector (numpy seed
   1116 + R, the same for both control seeds) times ||w_R||, and G_R = the clean truth probe's mean twin-pair gap
   mean_pairs w_R . (zz_false - zz_true) over the 300 attack twin pairs on the clean model. r_R and G_R are computed
   before step 1 and saved to `data/fits/random_panel.npz`. At step 0 this loss is about G_R^2, its gradient norm is
   about 2 G_R ||w_R|| (the panel loss's), and reaching its target needs the same state shift G_R / ||w_R|| along r_R
   that the panel loss needs along w_R. Same pressure, wrong direction. Everything else in s5 stands (lambda 1e4,
   300 steps, same batches, scored with the same truth panel and line rule; final eval KL, LoRA norm and step-0
   probe-term / KL-term ratio reported). The KL-only fine-tune stays dropped (zero gradient at LoRA B = 0).

2. **Named secondary row: the recalibration fix (Kasper #22323, Selma #22316).** The goal asks for "the recalibration
   fix"; PREREG s9 names only the probe re-fit. Added, same definition as the 1.5B parent's Rule B
   (theory-1022 `analysis.py:123-128`, `panelmax_rescaled`): on each scored state, each of the 20 frozen probes is
   re-standardized on that state's own 200 line-setting true statements (split `refit`, true; mean and SD with
   ddof 0), the monitor is the max over the 20, and the line is the 2nd-highest of the same 200. Reported per seed
   and pooled (city-cluster CI as in s3), with the eval false-alarm rate, for the panel arm, the control arm and the
   clean model, next to the re-fit recovery of s9. Reported, never the status. 1.5B value for the figure: 64%
   (blog `_notes/panel-fine-tune.md`).
