# DEVIATIONS: hiding-vs-stacked

Every change from the scout's design in the goal, with the clock time (PST) and the reason. The scout's prediction
(PREREG s0) is never changed; it is scored as written, with the mapping below.

## D1 (2026-11-13 07:58 PST, before any run): cells, aria-hidden, agents
- **aria-hidden removed from the cue-free methods.** The goal put aria-hidden="true" on every method. The agent reads
  raw HTML, so that attribute is itself a readable hiding cue, and no method would have been cue-free (7 of 7 sealed
  takes, #21974-#21980). Cue-free cells B and B2 now carry only a neutral class.
- **Cells.** The 4 style-file methods (display:none, off-screen, zero-size, clip) differ to a raw-HTML agent only by a
  class name (parent: 0 of 270 runs fetched site.css). Agent cells are now A (prereg inline control), A2 (inline
  style only), C (style-file display:none + aria-hidden only), B (style-file display:none), B2 (clip by class), E
  (visible link + aria-hidden). Off-screen and zero-size move to the model-free exposure check (Tove #21991).
- **Prereg mapping.** 'Style-file display:none or clip' = B or B2. 'Inline control' = A. 'Tracks the cue in at least 4
  of 5 methods' = at least 5 of the 6 run cells (same 80% bar). 'Reader mode' is reported with the footer caveat
  (PREREG s6).
- **Agents.** The goal's '3 agents' are picked by a frozen screen on cell A under the counter (PREREG s4), because the
  parent's nano (59/60) and Flash-Lite 3.1 (60/60) ignored the rule even with the cue in view, which makes the pooled
  70% bar pass by construction (Felix #21976, Tove #21991).
- **Status.** Binds per agent with CIs plus a gate on A (PREREG s5); the pooled 70% leg alone is not the status.
