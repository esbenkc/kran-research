# Sign-offs from the November 20 run (theory-1120-cue-size-law-8b-rep)

These folders hold the critic2 and critic-local verdicts on the November 20 draft, when the post was UNCONFIRMED
because control seed 6 (C1 s6) had no data file yet. They are kept as the record of that run. The banner of the
updated post reads only the sign-offs in `skeptics/*/signoff.md` from the November 22 continuation
(theory-1122-cue-size-law-8b).

- critic2 (2026-11-20): gpt-5.5 REJECT on a stale line, then a Claude fallback PASS after the gpt-5.5 budget cap.
- critic-local (2026-11-20, gpt-oss-20b, labq ahjy8jlss0): FAIL on two points. (1) C1 s6 missing: resolved on
  November 22, data/runs/l8_c1_s6.json (recovery job 9y8gn3qvoc), 8,192 updates, no switch. (2) "136 in the post,
  120 in m8_k3_s5.json": a misread. m8_k3_s5.json has T_dense 152; 120 is m8_k3_s7; the post's row lists seeds
  5/6/7 as 152/136/120.
