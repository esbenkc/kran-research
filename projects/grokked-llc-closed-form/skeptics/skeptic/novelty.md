# skeptic — novelty check (2026-10-15)

Claim checked: "With a sampler calibrated on networks with a known LLC and then frozen, SGLD reads
the grokked quadratic network on modular addition at 0.65-0.78 of the closed form
K_eff(3p-1)/2 (Cullen et al., arXiv 2603.01192 v3, Cor 5.4 / Remark 5.5), at p = 11-23,
with log-log slope ~1." Plus the side fact: on one-hot data each neuron has an extra
shift symmetry, so the generic count is 3p-2 per neuron, not 3p-1.

Searched (web, 2026-10-15): arXiv listing + HTML of 2603.01192 v1/v3, queries on
"local learning coefficient" + closed form + grokking / modular addition / quadratic network,
citations of 2603.01192.

| Source | What it does | Overlap with our claim |
|---|---|---|
| Cullen, Estan-Ruiz, Danait, Li, arXiv 2603.01192 v3 (2026) | Derives the closed forms. Checks SGLD only for scaling in p and K; App. H.1: "interpret absolute LLC values up to an estimator-dependent scale". Remark 4.3 warns non-generic atoms can lower the local image dimension. | They never test the absolute value with a calibrated estimator. No mention of the one-hot shift symmetry (grep of v3 HTML for the u_a+s_b shift / 3p-2: none). Our test is new; our NULL does not contradict their scaling claim, and the post must not say it does. |
| Panickssery & Vaintrob, LessWrong 2023, "Investigating the learning coefficient of modular addition" | SGLD on GELU embed-MLP for mod-add; estimate linear in p, about 2x ABOVE their heuristic 4p single-circuit count. | Different model (no closed form of Cullen type), uncalibrated, opposite sign of deviation. Not the same claim. |
| Lau, Furman, Wang, Murfet, Wei, arXiv 2308.12108 | SGLD LLC estimator; validates on models with known lambda (deep linear nets etc.). | Estimator source, no grokked/quadratic closed form test. |
| Shirodkar & Narayanan, arXiv 2606.21158 (2026) | Spectral "dead-direction" readings; reports calibrated LLC is rank-flat on a mod-add transformer at protocol budget. | Related point (SGLD under-resolves flat directions), different model, no closed-form comparison. |
| Lakkapragada, arXiv 2512.00686 (2025) | Empirical SLT on toy families incl. grokking; reports some deviations from theory. | Not this model or formula. |

Verdict: novel as a quantitative test of Cullen et al.'s Remark 5.5 value. Framing constraint:
the paper already concedes an unknown estimator scale, so the new content is (a) the size of
the scale at a calibrated setting (~0.72) and (b) that it is roughly constant over p = 11-23.
If the p = 31 frac-0.7 cells settle near 0.86, (b) does not hold and the post must say so.
