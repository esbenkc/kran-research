# refusal-dose-law

**Question:** At the layer where it writes, does a neuron's push along the refusal direction alone predict how strongly
pinning it bypasses refusal? (Qwen3-1.7B, layer 13, neuron 3270.)
**Sources:** arXiv 2605.08513, 2406.11717, 2606.19831; prompts from the refusal-neuron room's splits.json (copied to data/).
**Method:** PREREG.md (v1 2d569f6, before any TEST number) and DEVIATIONS.md D1-D5. Edits act at the layer-13 output;
p = push along the difference-of-means refusal direction / residual norm, from the clean pass. Judged compliance
(Claude Haiku 4.5, 3 labels) on 200/100/50 held-out harmful prompts per cell type. Curve = 2-parameter logistic in p,
fitted on built pushes and random units only. Code: phase0.py, run.py, judge.py, analyze.py; figure data: make_fig.py.

## Findings (data/results.json)
- STATUS REJECTED (both labelings, both frozen rules): the curve fits 70 built/random cells (R^2 0.909) but neuron 3270
  lies +12.6 to +38.3 points above it at pins 6-30 (all CIs exclude 0).
- Split at pin 30: whole pin +56.0 points (15% -> 71%); the part along the direction alone +18.5; the rest (zero push
  along it) +35.5, ratio 0.63 [0.54, 0.73]. With its part along 11 refusal directions removed: 0.375 [0.27, 0.47].
- Four held-out neurons that line up with the direction also beat the curve (38-48% vs ~22% predicted at pin 30).
- Scout clauses: c1 failed, c2 held, c3 failed (Spearman -0.03), c4 held -> prereg outcome failed.

## Blog post
`blog.kran.ai/_notes/refusal-dose-law.md`, figure `refusal-dose-law`, Figure 1 `refusal-dose-law-fig1`.
