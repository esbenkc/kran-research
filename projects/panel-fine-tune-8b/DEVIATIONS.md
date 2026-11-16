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

## D2 — the KL-only control is not a no-op on this code; it runs as a third arm (recorded before any 8B data, 2026-11-16 05:57 CET)

No Qwen3-8B extraction, probe fit or fine-tune step had run. This corrects a reason, not a rule: the FROZEN RULE
block, the prediction and the status are untouched.

1. **The no-op reason in PREREG s5 and D1.1 ("Adam then takes zero steps") is false on this code; D1.1's last
   sentence is withdrawn (Felix #22353/#22357, Kasper #22355, Vera #22354).** Evidence: the Qwen3-0.6B smoke of the
   KL-only fine-tune (lambda 1e4, probe term off, 10 steps; swarm folder `smoke06/logs/attack_ctrlkl__lam10000__seed0.log`,
   `smoke06/runs/ctrlkl__lam10000__seed0.json`): KL_cities 0.0 at step 1, 0.0465 at step 2, 0.0048 at step 10; all
   196 LoRA B matrices nonzero, max |B| 0.00184 (panel arm 0.00197). Cause: `kl_from_post` (attack_panel.py) computes
   sum p_ref (log p_ref - log p). At identical logits the value is exactly 0, but its gradient with respect to the
   logits is p x sum(p_ref) - p_ref, and in float32 sum(exp(log_softmax)) over the 151,936-token vocabulary is
   1 + ~3.7e-5. A CPU check (`iris_checks/kl_grad_check.py`: 8 x 151,936 random logits) gives KL 0.0 and
   max |gradient| 1.2e-4 at lambda 1e4, nonzero for every logit. Adam divides by the root of its running squared
   gradient, so this rounding residue becomes full learning-rate-sized steps. The same residue enters every arm at
   step 1, including the 1.5B parent runs.
2. **The goal's KL-only control runs: seeds 0 and 1, lambda 1e4, 300 steps, probe term off, everything else as the
   panel arm.** Reported as a third arm (a fine-tune driven only by float rounding and held back by the
   output-matching term), with the same rows as the random-direction control. Never the status. Queue order in
   laptop job B: random-direction seeds 0, 1 first, then KL-only seeds 0, 1.
