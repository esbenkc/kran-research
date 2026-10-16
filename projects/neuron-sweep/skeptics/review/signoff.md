# Skeptic `review`: sign-off for the neuron-steering sweep (follow-up to random-network-neuron-audit)

**Verdict: SIGN-OFF on the numbers.** The wording conditions in section 4 apply.
Own code only: `rebuild.py`, `matched.py`, `nojudge.py` in this folder. They read the raw jsonl in `../../data/`.
They do not import `analyze.py`, `judge.py` or `api_judge*.py`. Outputs are in `out/`.

## 1. Integrity (rebuild.py)
- 27,072 generated rows, 27,072 judged, 0 duplicate keys, 0 orphans, 0 unjudged rows.
- Stage 1: I re-parsed every raw reply (`"label": "A|B|C"`, with or without a ```json fence). 0 of 27,072 differ from `label1`.
- Stage 2: 7,961 rows. 15 raw replies contain both SAME and DIFFERENT. I read all 15: the `answer` field is right in each case (the other word is in `thing`). Final label = poly if stage 2 says DIFFERENT, else mono. This is what the file holds.

## 2. Shares, rebuilt (match lead #1193 to rounding)

| arm | n | none | mono | poly | any effect |
|---|---|---|---|---|---|
| no push | 192 | 94.3% | 2.1% | 3.6% | 5.7% |
| neuron, +1x max | 14,336 | 88.9% | 3.1% | 8.0% | 11.1% |
| random direction, 1x | 2,688 | 91.3% | 1.7% | 6.9% | 8.7% |
| neuron, +10x max | 7,168 | 31.7% | 54.9% | 13.4% | 68.3% |
| random direction, 10x | 2,688 | 29.2% | 55.7% | 15.2% | 70.8% |

Sample: 512 of 3,072 neurons per layer at 1x, the first 256 of the same order at 10x. 96 directions per layer per strength.

## 3. Null tests and controls
- **Arm permutation** (neuron vs direction tag shuffled within each layer, 2,000 permutations):
  - 1x, any effect: neuron +2.4 points, p = 0.001.
  - 1x, mono: +1.4 points, p < 0.001.
  - 10x, any effect: neuron −2.5 points, p = 0.011.
  - 10x, mono: −0.7 points, p = 0.48.
- **Strength-matched comparison** (matched.py). A direction borrows the max of a random neuron j. I compare the two arms inside the same layer and the same within-layer amax quintile.
  - 1x, any effect: +2.3 points [+1.1, +3.5].
  - 10x, any effect: −1.8 points [−3.5, −0.1].
  - 10x, mono: −0.1 points [−1.8, +1.6].
  - At 1x the whole neuron edge sits in the top quintile: 18% vs 10%. Quintiles 1-4 are equal within 1 point.
- **Layer profile is shared**. At 10x, the per-layer effect rates of neurons and directions correlate at r = 0.92 over the 28 layers. At 1x, r = 0.33.
- **Judge floor**: 5.7% of no-push rows get an effect label. At 1x, about half of the neuron arm's 11.1% is at this floor.
- **Judge-free local detector** (nojudge.py, no model). The rule was fixed before I looked at the arms. A row has an effect if a marker word occurs in at least 2 of the 3 texts and appears in fewer than 0.5% of all 1x texts.
  - no push 0/192.
  - direction 1x 1.8%, neuron 1x 2.7%.
  - direction 10x 23.2%, neuron 10x 26.5%.
  - The detector is conservative: on 160 hand labels it gets tp 22, fp 3, fn 37, tn 98.
  - It agrees on the sign at 1x. **It reverses the sign at 10x** (neurons +3.3 points).
- **Held-out judge validation** (100 blind hand labels, ids b*/c*):
  - Agreement 0.82, kappa 0.69.
  - Poly recall 5/8. **Poly precision 5/12**: 5 of the judge's 12 poly calls were hand-labelled none.
  - Hand mono called none: 7/44.

## 4. Pre-registration #1175, as I scored it
- **H1** (≥80% none at 1x): **holds**, 88.9%.
- **H2** (mono > poly among effect rows): **fails at 1x** (28% mono [26, 30]). **Holds at 10x** (80% mono [79, 81]).
- **H3** (neurons give an effect more often than same-strength directions): **holds at 1x** (+2.4 points, p = 0.001; matched +2.3). **Fails at 10x** (−2.5 points by the judge, but +3.3 by the judge-free detector).
- **H4** (effects cluster in the neurons with the largest amax). My test: top within-layer quintile vs the rest.
  - **Holds**: 1x 18% vs 9%, 10x 95% vs 62%, p < 0.001.
  - Directions with borrowed top-quintile strength show the same rise at 10x (95%). So H4 is mostly push size, not neuron identity.

## Wording conditions for the fold
1. Do not write "H3 fails" without a scale. At 1x neurons beat directions by about 2 points, and the edge comes only from the top-amax fifth. At 10x the two are equal within about 3 points. The sign at 10x depends on the measure (judge vs judge-free).
2. Report the no-push floor (5.7%) next to the 1x numbers. Do not present 11.1% as "neurons with an effect".
3. Steering measures what a neuron writes. The post's main audit measured what a neuron reads. Do not call a one-theme push "monosemantic" in Cammarata's sense. Poly includes "each text broken in its own way", and the judge's poly precision on held-out rows is 5/12. So "poly" here is not evidence of superposition.
4. The per-layer shape at 10x is shared by random directions (r = 0.92). The figure must show the direction arm per layer next to the neurons. The text must say that the shape tracks layer and push size, not the neuron basis.
5. Say what was tried. Esben asked for a local judge, and both local judges failed validation. The prompt was tuned on dev rows. The number to report is the held-out kappa of 0.69 (n = 100). The judge-free local detector numbers can stand next to it.
6. Limits: one prompt, 3 samples of 20 tokens, 1/6 of the neurons at 1x, 1/12 at 10x, the positive pole only (+amax added, not clamped), one judge.

## Novelty (short)
- Output-side reading of MLP neurons is known prior work. Geva et al. 2022 ("Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space", arXiv 2203.14680) projects value vectors to the vocabulary. Gurnee et al. 2024 ("Universal Neurons in GPT2 Language Models", arXiv 2401.12181) finds prediction and suppression neurons.
- I know of no published per-neuron steering sweep that uses a blind mono/poly/none judge with same-strength random-direction and no-push controls on Qwen3-0.6B. Treat this as a small addition, not a new method.
- Credit Geva 2022 in the fold.

API spend by `review`: $0.

## 5. Fold check (blog 1bc8488, PR #29), final
- Conditions 1-6 above are all met in the fold text.
- I recomputed two validation rows from the committed raw files in data/validation/: Qwen3-8B v2 54/46 claim (8B: 84 rows, 46.4%, 59% of ordinary texts called changed) and GPT-4.1-mini with the "changed" framing (47.6%, 68% called changed). Both match summary.json and the fold.
- Preview checked in light and dark mode: the bars, the hollow direction dots, the rotated control labels and the heatmap with its colour legend all read. figure-lint and the Netlify preview are green.
- Cosmetic, not blocking: the colour legend's "0%" tick is clipped at the left edge.

**Final: SIGN-OFF on the numbers and the fold.**
