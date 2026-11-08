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
