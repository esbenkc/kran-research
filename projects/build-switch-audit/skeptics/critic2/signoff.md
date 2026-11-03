# critic2 signoff (round 1) - build-switch-audit

Reviewer: Kasper (critic2). The openai/gpt-5.5 session hit its $2.48 cap ($2.60) after writing rebuild.py/rebuild.out;
the role restarted on Claude (opus, subscription) and finished the review. Blog sha 44ef5b3, kran-research c929c05b.
Split: rebuild.py was written and first run by gpt-5.5; Claude re-ran it in the room venv (env/.venv, 0.46 GB RSS, 1.4 s)
and got byte-identical output (rebuild.out). rebuild_extra.py/.out and this signoff are Claude's.

Answer to the title: NO (REJECTED) holds.
- One-build auditor, pooled AUC 1.000, exact p 1/70 (A_s0, A_s1); scores switches 0.90-0.95, clean 0.05-0.12 (rebuild.out).
- Two-build auditor AUC 1.000, p 1/70; carrier 4/4 in both sessions; one-build 3/4 in both (rebuild.out).
- Nudge search at unit 1-8: switches 7.9-18.7% -> 63.8-94.2%; clean +0.50 to +0.76 pts (rebuild_extra.out).
- System prompt SYS1: switches 58.5-96.0%, clean 11.4-21.2% (from 6.3-11.5%); SYS2 stack 0.2%, 3.6% (rebuild.out).
- Whole-change norm from raw merged tails: switches 9.52, 7.76, 5.07, 5.80; clean 3.94-4.01 (rebuild.out).
- Edits: two-build sessions pass stack_s2 + r50_s1 in both (B_s0 also stack_s0), 0.0206-0.0430% of weights;
  random-head medians 28.2/43.5/54.2/70.8 pts (rebuild_extra.out).

Round 1 verdict: FAIL on one false sentence.
- Post l.76 "and two exceeded the other (5.8)": the three 50%-longer nulls are 5.700, 5.524, 5.924 vs r50_s1 5.798
  (data/reading_rows.json, data/screen/norms.json). Only 1 of 3 exceeds it. Fix: "and one exceeded the other (5.8)".
- Nit, expert foldout "clean 3.95-4.01": cham_r100_s1 = 3.9446 -> "3.94-4.01".

Round 2 verdict: PASS on blog sha 1db0b4d (PR #157, all checks green, preview checked 20:10).
- l.76 now 'and one exceeded the other (5.8)'; foldout '3.94-4.01'; no 'two exceeded' left (preview text grep).
- Every TLDR/findings/prereg-note number above matches the raw rebuild. Title answer: NO (REJECTED).
