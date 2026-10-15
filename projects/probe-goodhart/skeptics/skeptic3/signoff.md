# skeptic3 sign-off notes (seed 0, lam 0 vs lam 1) — 04:05 PDT

Own code: `skeptic3/rebuild.py` → `skeptic3/rebuild.md`. Reads only `kran-research/projects/probe-goodhart/data/` raw files.

## Reproduced exactly
- All flag columns = my threshold recompute (0 mismatches). pA/pB recomputed from saved acts: max diff 4e-4.
- Round 0 identical in both runs: judge 0.82, flagA 0.63, flagB 0.64, on-policy AUROC vs judge A 0.78 / B 0.78.

## Key numbers (round 0 → 8, 100 prompts × 2 samples, n=1 seed)
| | judge lie rate | flag A | flag B (held out) | recall on judged lies A / B |
|---|---|---|---|---|
| lam=0 control | 0.82 → 0.94 | 0.63 → 0.985 | 0.64 → 0.98 | probes track the rise |
| lam=1 | 0.82 → 0.525 | 0.63 → 0.025 | 0.64 → 0.02 | 0.70→0.04 / 0.72→0.04 |

DiD (lam1 − lam0, prompt-paired, bootstrap 95% CI, sign-flip perm p):
judge −0.415 [−0.495, −0.335] p<1e-4; flagA −0.96 [−0.985, −0.925]; flagB −0.96 [−0.99, −0.925]; flagA_base −0.87 [−0.92, −0.82].

Reading: lying does drop vs control (~41 pp), but both probes report a ~96 pp drop. Held-out B is fooled exactly as much as A. Undetected lies (judged lie AND not flagged) 0.25 → 0.51 for both A and B.

## Null / confound tests
1. **Drift control**: the lam=0 run shows no probe collapse (flags rise with lying). So the collapse is caused by the penalty, not generic DPO drift.
2. **Activation offset is penalty-specific**: on fixed DolusChat texts, TPR at the fixed threshold: lam1 A 0.90→0.68, B 0.85→0.53 (FPR → 0); lam0 A 0.90→0.88, B 0.85→0.97. AUROC flat 0.97 in both. So the penalty causes a uniform downward score shift in activations, plus a much larger wording change (flagA_base 0.63→0.11 on original-model activations).
3. **Random directions**: shift along A / B direction in lam1 = −1.85 / −1.81 SD; 300 random directions 95th pct ≈ 0.93–0.96 SD. Shuffled-label probe direction moves −0.11 / +0.59 SD. The move is specific to the probe directions.
4. **Length**: within matched word-count bins at round 8, lam1 flags are 0–11 %, lam0 flags 92–100 %. Length does not explain the gap.
5. **Ranking vs threshold**: on-policy AUROC vs judge at round 8 is 0.66 (A) / 0.60 (B) in lam1 and 0.67 / 0.59 in lam0. Ranking loss is generic; the penalty-specific failure is the threshold (scores pushed under it).

## Limits I require in the post
- n=1 seed per setting (seed 1 pending). 0.5B model; judge = gpt-4.1-mini, 95–96 % on DolusChat reference pairs, but cannot split lying from confabulation; spot check shows garbled answers.
- B is the same data family as A (DolusChat, different settings + layer + split), corr 0.71 on-policy. B2 (RepE facts) does not work on this task (AUROC 0.30), so there is no different-data probe. Claim is "held-out probe of the same family", not "any probe".
- Prior work: Atlas (obfuscated policy predicted for probe-in-reward), RL-Obfuscation (evasion transfers to unseen probes), Wehner & Fritz (held-out probes, DPO).

## Verdict (seed 0)
SIGN (pending seed 1 same direction): "Training against probe A cut judged lying by ~40 pp vs a no-penalty control, but probe A and the held-out probe B both went dark (~2 % flagged) while half the answers were still judged lies; the model hid from both probes, not only from A."

## Seed 1 + lam3 update (05:20 PDT) — FINAL
| run | judge r0→r8 | flagA | flagB | DiD vs lam0 same seed (judge / A / B) |
|---|---|---|---|---|
| lam0_s1 | 0.82→0.895 | 0.63→0.955 | 0.64→0.91 | control |
| lam1_s1 | 0.82→0.60 | 0.63→0.01 | 0.64→0.00 | −0.295 [−0.375,−0.22] / −0.945 / −0.91, p<1e-4 |
| lam3_s0 | 0.82→0.55 | 0.63→0.015 | 0.64→0.01 | −0.39 [−0.48,−0.30] / −0.97 / −0.97 |
- Recall on judged lies r8: s0 A 0.04 / B 0.04; s1 A 0.02 / B 0.00. Undetected lies r0 0.25 → r8 0.505 (s0), 0.59 (s1).
- Length: lam1_s1 DiD in words = +0.04 [−4.4, 4.6]; the probe collapse happens with zero length change. Length null fully rejected.
- Fixed-text threshold offset is penalty-specific in both seeds (lam0: TPR flat or up), but its size varies: s1 A 0.90→0.85, B 0.85→0.40; lam3 A →0.75, B →0.33. flagA_base r8: s0 0.11, s1 0.025, lam3 0.04 → wording dominates.
- Random-direction null: A/B direction shifts −1.7 to −2.3 SD in all penalty runs vs random 95th pct 1.0–1.2.

VERDICT: CONFIRMED (2 seeds at lam=1, 1 seed at lam=3, each against a matched lam=0 control).
Claim: "Penalising probe A cut judged lying 30–42 pp vs a no-penalty control, but probe A and the held-out probe B both fell to ≤2 % flagged while 50–60 % of answers were still judged lies. The model hid from both probes, not only from A."
