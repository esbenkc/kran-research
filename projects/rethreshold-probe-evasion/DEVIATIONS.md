# Deviations from PREREG.md (frozen at commit beff840, 2026-10-18 15:42:02 CEST)

Every change after the freeze, with the time it was decided (room post time) and the commit that logged it
(`git log -- DEVIATIONS.md`). None of these changes the status rule, the headline lambda rule or any number.

## D1. Figure x-axis: lost TPR -> push p (presentation only)

- Decided: 15:46-15:47 CEST (Tove #2755, Felix #2756), before the first scored run finished (scored runs started
  15:42:04; the first ends ~15:51). Only unscored pilot data (seed 99) had been plotted.
- What: PREREG "Figure" puts the TPR drop at `t0` on x. Every final point has a drop of ~96 points (all runs cross
  50 points by design), so all final points and all E6 curves fall on one vertical line. New x = the push of the
  positives in clean-sd units, `p = (mu+_clean - mu+_att)/sigma`, as defined in PREREG "Theory test".
- Also: each final point gets a vertical bar = 95% range of the share over 200 redraws of a 200-negative refit set
  from the 796 held-out clean negatives (PREREG Secondary 8 applied per run instead of only at the headline
  lambda). The E6 line uses the measured D and the pooled final c.
- Effect on the result: none. The status and prereg outcome are computed from the frozen refit-200 and the paired
  bootstrap exactly as in PREREG.
