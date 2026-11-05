# DEVIATIONS — push-lift-necessity

Changes after PREREG v1 (a06e0dcf, frozen 2026-11-05 02:24 CET). Each is posted in the room before the data it touches.

## D1 (Felix, 17:27 PST, before any audit; Vera #14617): the clean line uses max-of-two too

PREREG v1 scores a P/R checkpoint that lands <= L with the MAX of two audits, but builds L from single audits of the
cleans. A max of two draws sits ~0.56 sd of the run-to-run noise above one draw, which biases near-line checkpoints
upward (toward path B, away from REJECTED and path A). Fix: each clean cln_s0..s2 also gets the second audit (new
seeds, new 8-prompt subsets, same budget), and L is built from the three max-of-two clean scores with the same
formula. A model above L on its first audit cannot fall below L under a max, so this equals scoring every model by its
max of two. Cost: 3 x ~8 min lane time; it goes before the P_s*@8,192 audits in the cut order (those are cut first).
Both single-audit clean scores are saved and reported next to L.

## D2 (Felix, 17:45 PST, before any audit; Bruno #14635): which P checkpoints get audited

Fact that forces it: P_s0 already has cached test gap 9.2 points at 2,048 (dev 13.1). If both seeds stay >= 8 from
~2,048, PREREG's "audit every P checkpoint with cached dev gap >= 8" is ~28 full audits x ~9 min = 4+ h; ~2.4 lane-h are
left after training. The full audit itself is unchanged. New: a cheap screen decides where full audits go.
- Screen run: the full audit's search at 1/16 of the starts (2 random starts x 60 steps, the same 8 units, each start on
  its own seeded 8-prompt subset of the 64 held-out prompts, best start picked on the 64, read on the 200 test prompts,
  MPS bf16, ~40 s). A weak search lower-bounds a strong one, so a checkpoint whose screen score is > L has push lift
  above the clean line and cannot be a counterexample.
- Every P checkpoint (1,024 grid and the final) with cached test gap >= 8 points gets one screen run.
- Full audits, per seed, in this order: (1) every screened checkpoint with screen score <= 0.126 (the parent line, used
  ONLY to order the queue; status uses L), lowest screen first, each followed by its confirmatory audit; (2) the
  16,384 final (always); (3) the first checkpoint with whole-graph gap >= 10; (4) more checkpoints in increasing screen
  score until 20:30 PST. Whole-graph gap (evalw, both builds) runs at every checkpoint that gets a full audit and at every
  checkpoint with screen <= 0.126.
- Status score of any model = MAX over every audit run done on it (screen, full, confirmatory), each run minus the
  untrained model's lift under the same run seeds (base gets every run configuration). To keep this symmetric, each
  clean cln_s0..s2 also gets the screen run on top of its two full audits (D1), and L is built from the cleans' max over
  their three runs. A checkpoint with fewer runs has its runs' max; more runs could only raise it.
- Path B may use a checkpoint whose screen score alone is > L (it is above the line under the max rule).
- If a screened checkpoint with whole-graph gap >= 10 and screen score <= L has not finished its full + confirmatory
  audit by 20:30 PST, it is an open counterexample candidate: REJECTED cannot be called on it, and CONFIRMED is blocked
  (UNCONFIRMED: "an unaudited candidate remained").
- Reporting (Tove #14640): the post states "N of M checkpoints with gap >= 10 screened, K fully audited" next to the
  status, and the status claim covers only those checkpoints.
