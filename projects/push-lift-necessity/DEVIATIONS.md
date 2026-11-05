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
