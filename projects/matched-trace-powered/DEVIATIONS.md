# DEVIATIONS — matched-trace-powered

Each entry: time (PST), what had run when it was made, what changed, who asked.

## V1.1 — end-read size and estimator form (2026-11-11 17:00 PST)

What had run: joint_s0_cont launched 16:58:53; no dev check and no read of this room existed (log: weight loading only).
Changes only the end read; the status rule and the training are unchanged. Full text: PREREG.md v1.1 @ 8e82ff3c.

- Size: R17 x 200 = 3,400 rows -> R32 x 200 = 6,400 rows, draw-major order (draws 0..31), read hard stop 20:50 PST;
  a cut keeps whole draws only, the achieved N is reported, nothing is re-read (Kasper #19853, #19868; Tove #19855;
  Vera #19856, #19866). Reason: at 3,400 rows the s_R half-width is ~0.071, so the clause s_R CI-high < 0.10 would be
  near-unreachable for a true s_R near the exploratory 0.06; 6,400 rows give ~0.052.
- Phase A's slot (20:25-20:55) goes to the end read; Phase A becomes foldout-only, run only if the read ends early.
- y_R frozen as ONE form: the self-normalised ratio mean(c_nb) - sum(w c_lnb) / sum(w), recomputed in every bootstrap
  resample (Vera #19852, #19854); the cross-fit-b form of v1 is dropped.

## D1 — fresh AdamW moments at the continuation start (known before T0)

data/runs/joint_s0/adapter.pt holds no optimizer state, so the continuation starts with fresh AdamW moments. Stated in
PREREG v1 Design. Vera checks the KL and reward rows after update 674 against the parent's 650-674.
