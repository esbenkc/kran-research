# DEVIATIONS — matched-trace

## D1 — Rao-Blackwellised planted switch y_P (2026-11-10 17:18 PST)

Made BEFORE any end read and before the first dev check of MAIN (joint_s0 at update ~20; only the step-0 dev row and
the S3 smoke read on the pilot adapter existed). Proposed by Felix (#19735); supported by Tove (#19736), Kasper
(#19737); estimator checked by Vera (#19739, PASS on joint_pilot_n16.npz).

The cue moves only the logit of ' Need' at thought position 0; the other first tokens keep the probe-off odds and the
continuation is unchanged. So per prompt, exactly in expectation:
c(off) - c(off + cue) = (p_on(X) - p_off(X)) x (E[c | first token not X] - E[c | first token X]) = D_P x forced.

- y_P primary = mean over TEST rows of D_P(row) x (c_notX - c_X) (summarize.py key y_P_rb), same population scale and
  same estimand as PREREG 'Reads'; prompt bootstrap 2,000. s_P = y_P / (D_P x cheat_off) as before.
- The plain CRN y_P is printed beside it in summary.json, the proof matrix and the post's foldout.
- Void rule (Vera #19739 2, Tove #19736): if the direct-minus-RB CI on the end read excludes 0, D1 is void and the
  plain CRN y_P is primary.
- y_R keeps its plain CRN estimator (no analog exists). The post names that P is measured more precisely than R;
  this makes YES harder only through R's noise (Kasper #19737 3).
- Status rule unchanged. Smoke check (pilot, n16 R4): direct +1.40 [-2.16, +4.60], RB -0.19 [-1.22, +0.69] points;
  row SD 0.130 -> 0.039.
