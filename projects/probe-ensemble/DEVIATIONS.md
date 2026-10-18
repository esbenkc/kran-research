# DEVIATIONS — probe-ensemble

Changes to the design in PREREG.md. The decision rules in PREREG.md sections 4, 5 and 7 do not change.
Times are commit times (git log), Copenhagen time.

## D1 — judge model swap, all arms rerun fresh (decided 03:08, before any training step)

What happened: the OpenAI account ran out of credit (HTTP 429 `credit_balance_exhausted`, Bruno #2366,
03:08). gpt-4.1-mini was both the favourability reward and the deception judge. The ensemble run had finished
only its round-0 evaluation (no training step); its round-0 text matched the saved probe-goodhart run
(probe A flags 0.63, probe B 0.64, mean 46 words).

Change (Felix #2370, Tove #2367):
1. One judge model, gemini-3-flash-preview, replaces gpt-4.1-mini in BOTH roles. Prompts unchanged.
2. Gate before any training step: agreement >= 0.90 on the 100 DolusChat reference pairs and >= 0.85 on Iris's
   60 agent labels (stringcheck/). If the gate fails, no training runs and the title status is UNCONFIRMED.
3. No reuse: control, single and ensemble arms are all rerun fresh with the new judge, seeds 0 and 1.
   The PREREG reuse condition (section 3) no longer applies. The lambda=3 reference runs only if it ends by
   05:30 CEST; otherwise it is dropped from the figure.
4. Run order: single s0 + ensemble s0, then single s1 + ensemble s1, then control s0 + s1, then lambda=3 s0.
   Training stops at 05:30 CEST. A missing control seed is reported as missing.

Why the rules stay valid: the title rule compares only the single and ensemble arms, under the same judge.
The probe-goodhart numbers (single cut 0.295 / 0.22) no longer set the targets; the new single arm does.

## D2 — exploratory run ens_lam2_s0 (decided 03:52, outside the verdict)

What happened: on the policy's own round-0 answers the max over eight members sat above 0.95 on 64% of answers.
Its honest-vs-deceptive gap was 0.20, against 0.38 for probe A (Bruno #2382). At round 4, ens_lam1_s0 had
barely moved (judged-deceptive 0.78 -> 0.74; Bruno #2392).

Change (Felix, Tove #2393, Iris #2384): one extra run, the ensemble penalty at lambda=2 (gap per unit reward about
matched to probe A), seed 0 only. It runs after control s0 and refit.py, and is killed at 05:50 CEST if not at round
8. It is exploratory: it never enters the title rule, the prereg outcome or the figure; the post may give it one
sentence.
