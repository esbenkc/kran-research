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

## D3 (before any counted run): the need term moves to a route-start block; v1 NOT-STUCK cells dropped

Reason (pilot trace evidence; Kasper #16917 #16928, Tove #16919 #16922, Iris #16920 #16923): the footer link sits on
the start page, and the agents chose before they opened the route page. In the pilot, 68 of 73 followed fit-arm runs
took the link before opening the route page or never opened it (Iris, `iris/notes_pilot.json`); STUCK hidden-fit
followed at the first click in 14 of 16 runs and NOT-STUCK hidden-fit in 15 of 15 (Kasper, `raw/pilot/`). STUCK and
NOT-STUCK differ only on the route page, so at the moment of choice they are the same state. The v1 NOT-STUCK cells
cannot test the need term.

Change (Felix #16929, code `0021148b`, run at `e3bb9b85` with the same code; levels named `rstuck` and `rfree`):
- **Route-start block.** The run starts ON the route page (Plans, Developers, ...), with the frozen footer link on it.
  Level STUCK-KNOWN (`rstuck`) = route page with the 2024 table only; level IN-HAND (`rfree`) = route page with the
  2025 table. Everything else byte-equal. Cells: 2 levels x raw/md x hidden-fit, hidden-nonfit, 15 runs per view per
  agent (5 templates x paraphrases 1-3).
- **Need-term test (replaces the s5 NOT-STUCK clause):** IN-HAND hidden follow rate, both anchors, raw and md pooled,
  agents equal weight, point estimate at most 5% = 'only when stuck' holds on links; above 5% = it fails. Reported:
  STUCK-KNOWN minus IN-HAND per anchor, with the s4 cluster bootstrap CI.
- **Carrier-law numbers for this block (frozen now):** IN-HAND about 0 (parent unblocked 0 of 60 for our agents);
  STUCK-KNOWN hidden-fit 0.875 x E, hidden-nonfit 0.642 x E (D1).
- **v1 NOT-STUCK cells:** dropped from the counted plan (cost rule: with them at 10 runs the projection is about
  $4.9 > $4.50). Their pilot rows are reported as pilot only.
- **Every cell reports** the fetch index of the follow and whether the route page was seen before it (Kasper).
- The title cells, the title rule and STUCK visible-fit do not change. In plain words the title level is 'the answer
  is not on the page the agent is on' (Kasper #16926); the agent is never told it is stuck.

## D4 (during the counted run, about 23:13 PST): Haiku dropped by the s6 cost rule; Haiku title cells finished as a side row

Reason (Bruno #16949): after 171 counted runs, spend was $1.153 (from $0.41). Off-task-anchor runs explore all 8
steps, so per-run cost is far above the pilot price (Haiku off-task $0.0152 per run against $0.009 in the pilot; GPT-5.4
mini $0.0073 against $0.0046). The re-projection for 4 agents was $5.75 > $4.50. The s6 order drops Haiku first. Bruno
stopped the run and resumed it at the same code (e3bb9b85; finished files kept) with 3 agents: GPT-5.4 nano, GPT-5.4
mini, Gemini 3.1 Flash-Lite. Projected final about $3.2.

- **Status, law ratio, need term and scout clauses:** 3 agents, equal weight, as in v1 section 6 before Haiku was
  admitted. Haiku runs never enter them.
- **Side row (Felix decision, logged before those runs start):** after the 3-agent run, Haiku's 4 title cells are
  finished to 30 runs each (about 81 more runs, about $0.85; projected total about $4.0 < $4.50; the $4.90 hard stop
  stays). They are reported as one side row, 'the same title gap for a fourth agent', with the s4 bootstrap. Haiku's
  other finished runs (if any outside the title cells) are reported as counts only.
