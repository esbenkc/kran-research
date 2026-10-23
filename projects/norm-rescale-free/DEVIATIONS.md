# Deviations from PREREG.md — norm-rescale-free

Each entry: Copenhagen time from `TZ=Europe/Copenhagen git log` of the commit that adds it, what changed, why, and
whether any scored data existed at that time.

## N1 (before step 1 of any scored run; no scored data exists)

1. **Pair count in s6.** PREREG s6 says "the 298 eval twin pairs". The eval split has **596 twin pairs**
   (298 cities x 2 templates; a pair = same city AND same template, false statement vs its true twin; Iris #5439).
   g is the mean over these 596 pairs. The "298-pair check" in s4/s5 is a different quantity (the behaviour check
   on 298 affirmative pairs) and is unchanged. Everything else in the R rule is unchanged.
2. **Figure layout (s9) is open, not frozen.** Tove #5440: the title is decided by R, so the main panel may show R
   vs training step (both arms, per seed, with the step-0 prediction, the R = 1 line, the 0.5 decision line and the
   lambda 0 null), with catch vs eval KL as the second panel. analysis.py saves both; Selma picks after the first
   scored run. This changes no number and no status rule.
3. **Step-0 prediction label (s8, Tove #5450).** R_0 = cos^2(w/sd, h_hat) is the ACTIVATION-SPACE prediction: one
   gradient step taken on the readout-28 state itself, not on the LoRA weights. The figure and the post label it
   so. The measured R at step 50 is the weight-space counterpart.

## F1 (fact recorded during A s0 training, before any scored checkpoint was read; no rule change)

The 5-step smoke (seed 99, not scored) showed that ONE Adam step from the LoRA init already gives eval KL 0.0021
nats/token (Bruno #5460; the KL term is exactly 0 at that step). The prediction's bar "eval KL at most 0.002" (s7
clause 1) therefore sits at the one-step floor of this attacker. Clause 1 is still scored literally; the post says
where the bar sits.

## N1.3a (text fix to N1.3; Iris #5473; before any scored checkpoint was read)

The step-0 activation-space prediction for the frozen R (mean twin gap, false and true twin both move) is the
gap-weighted pair average R0 = sum_k gap_k (cos^2_false,k + cos^2_true,k) / (2 sum_k gap_k), cos = cos(w/sd, h_hat),
over the 596 eval twin pairs = 0.0028 (theory/theory.md Lemma 2; data/results/theory.json key R0). The plain
cos^2(w/sd, h_hat) in N1.3 is the false-statement-only shorthand (median 0.51%); the figure and the skeptics use R0.

## N2 (secondary runs only; written before any of them started; status runs untouched)

Tove #5503 on Iris #5499 (theory.md Fact 5, flow_quadratic.json @c67f16b): when the country-token states move
freely, the restore share R at half gap closure stays 0.002-0.005 up to a KL weight lam_sim = 1e6 and jumps to 0.69
at 1e7 and 0.96 at 1e8. Iris's mapping lam_sim ~ lam / 21 (KL averaged over ~85 positions of 8 statements vs gap^2
over 8 pairs; attack_norm.py:211-244) is an INFERENCE, not a measurement.

1. **Swap.** The secondary run B lambda 1e5 s0 is dropped. In its place: **A lambda 3e8 s0** (lam_sim ~1.4e7),
   same code, 300 steps, n = 1, never the status.
2. **Secondary order**: A lambda 3e8 s0, A lambda 1e5 s0, A lambda 0 s0 (the last only if it can start by 18:15
   CEST).
3. **Prediction for A lambda 3e8 s0, fixed now**: at step 300, R >= 0.5, OR the gap fails to halve
   (precondition fails). It is FALSIFIED if the gap halves and R < 0.5. Reported as a test of the mechanism
   (the stretch is the cheap route only once the output penalty is ~3-4 orders of magnitude above the status
   weight), never as the answer to the title.
4. **Caveat on the N2.3 prediction (Vera #5507, Iris #5506/#5508; added before the run starts).** lam/21.3 matches
   only the loss WEIGHT of one country-token KL. The real fine-tune pays most of its output change elsewhere (A s0
   step 300: 3.5% of the summed per-position KL on the attack statements sits at country tokens), and its KL batch
   uses other statements than its probe batch. So the prediction is loose: a miss (gap halves, R < 0.5) shows the
   weight-space path matters, and does NOT refute the free-state curve of theory.md Fact 5.
