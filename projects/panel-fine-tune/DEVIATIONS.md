# Deviations and notes after PREREG.md (commit 99ef0e1, 14:57:32 CEST)

Each entry: the Copenhagen time of the commit that first recorded it (from `TZ=Europe/Copenhagen git log`), and
whether any scored attack step had run at that time.

## N1 — two notes on the theory line (secondary, s9), recorded before any scored step (Tove #4881/#4889, Iris #4877, Tove #4884)

No change to the frozen text; the status and the prediction clauses are untouched.

1. PREREG s9 "below the line" means the CLEAN model's line. The floor is the cheapest output change that pushes a
   caught false statement under that fixed line. The headline line is re-set on the scored model's own true
   statements, so a fine-tune can also lift true scores. Under the re-set line the floor is a reference cost, not
   a proven lower bound, unless Iris (T5) proves the bound for the re-set line. The post says which one holds.
2. Second theory prediction (Iris): readout `tok28` (block 28 before the final RMSNorm) can be hidden at exactly
   zero output change. RMSNorm(c x) = RMSNorm(x) for any c > 0, per token, and the probe is affine in x, so a
   statement-dependent shrink of the false statements' country-token state (true twins kept) moves their tok28
   scores under the line while the logits stay the same. A uniform shrink does not help under the re-set line
   (all scores move through the same increasing map). The figure marks "free at 28, floor at 28+" next to the
   measured cost.
