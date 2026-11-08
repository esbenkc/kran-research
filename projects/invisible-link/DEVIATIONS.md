# Deviations — invisible-link

Every change after PREREG v1 (816beab8) with its reason. The v1 rule stays a reported row.

## D1 (before any counted run): agent-matched ancestor reading and Iris's forecast, added as readings

Reason: Iris #16873 recomputed the ancestor (`decoy-usefulness`, branch `swarm/defense-1031-usefulness-beats`, raw/,
its `analysis.py` `score_episode`) on OUR three agents only, real and foreign records pooled, n = 40 per agent per
scope. The v1 band [+5, +22] (section 0, section 5) came from the five-agent pool. Tove #16878 asked to add the
matched reading before the first counted run.

- **Ancestor transfer, agent-matched:** stuck useful-key use GPT-5.4 nano 33/40, GPT-5.4 mini 32/40, Gemini 3.1
  Flash-Lite 40/40 (mean 0.875); useless-key use 17/40, 20/40, 40/40 (mean 0.642). Transferred to the link carrier:
  hidden-fit 0.875 x E, hidden-nonfit 0.642 x E, gap +23 points (per agent +40, +30, 0). Binomial-bootstrap 95% band
  of that gap, agents equal weight (Felix, 20,000 draws, seed 0): [+14, +33].
- **Iris's forecast (Iris #16874):** the ancestor's useless key sat inside the storage denial, and 94 of 196 notes
  called it a storage key, so 0.642 is inflated by placement. Our off-task anchor sits in a neutral footer. Forecast:
  hidden-nonfit 0.1-0.3, gap above +23.
- **Scoring:** both are readings, reported next to the v1 band [+5, +22] with the STUCK gap point estimate. The title
  rule (section 4) does not change.

## D2 (before the pilot): code brought to the v1 text; markdown ratio named a placebo pair

Reason: Iris #16884 found 5 places where `websites.py` (sha 9a2e041d) differed from PREREG v1 section 1. The code
moves to the v1 text; the PREREG text does not change (Felix #16889): per-template fit and nonfit strings as the s1
table; `rel="nofollow"` on every link arm, including visible-fit and the sr-only control; one footer slot shared by
every arm (hidden = the same `<a>` inside `<div aria-hidden="true" style="display:none">`, visible = the bare `<a>` in
that slot); `display:none` on the div, not on the `<a>`.

Stated before data (Iris #16892, Tove #16895): with one shared slot, the markdownify output of hidden-fit and
visible-fit pages is byte-identical, because markdownify drops the inline style and `aria-hidden`. So the section 5
markdown ratio is a placebo pair: it equals 1 by construction, up to sampling noise or a harness bug. It is reported
as a placebo and never counted as evidence for the law. Raw HTML is the one real test of a hiding discount.
