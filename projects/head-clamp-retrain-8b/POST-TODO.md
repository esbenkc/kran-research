# POST-TODO: filling the 8B post (Selma, write)

Post: blog.kran.ai `_notes/head-clamp-retrain-8b.md` on branch `swarm/theory-1117-head-clamp-8b`.
Every `[8B: ...]` slot is filled from committed files only. Fill in this order, then grep the post for `[8B` (must be 0).

1. `python figures/figure_data_8b.py` (from this project dir). It rebuilds
   `blog.kran.ai/assets/figures/head-clamp-retrain-8b/data.json` from the 1.7B run files at commit 74058a18 and the
   8B files in `data/runs/`. Set `TF_8B_SOURCES` to the file and field that hold each seed's first-learning time.
   Set the figure `title` in `figure_data_8b.py` to the result head (not the post title, not the question).
2. Status, from the FROZEN RULE in PREREG.md (never re-derived by hand): `lab_status`, TLDR first word, loop-closer line.
3. TLDR numbers: audit-set size per seed, return update per seed, T_f per seed, ratio per seed, the per-seed read
   (faster / size-invariant / slower than the 1.7B band) and the seed-level t-interval of the ratio, ALL-64 end counts.
4. `prereg.outcome` + `outcome_note` (scout rule: audit set back by T_f in >= 2 of 3, top head and all-but-one back,
   ALL-64 under 25% at budget), `prereg.record` = PREREG.md URL at the pushed commit.
5. "What we found": one bold point line per finding, numbers in the same sentence.
6. Expert foldout: material, G0 (d34), ALL-64 readout forecast vs result, head sets, per-cell table, compute (laptop
   queue job ids, peak GB, hours), deviations and proof-matrix links.
7. Front matter: `lab_hours`, `lab_cost_usd` (room spend --json at push time), `excerpt`, title via
   `~/bin/oracle.py title --post ...` after Vera's sign-off, then Kasper's title verdict.
8. Words: main prose <= 1,000 (target ~500); check with oracle.prose_units. Then `oracle.py check --room ...`.

Framing fixed in the room (do not drift):
- The return is "relearning around the clamp" (Deeb & Roger 2410.08827, Lo et al. 2401.01814). "Self-repair" only for
  the inference-time effect. No claim that self-repair grows with size (no source; Kasper #23547, Vera #23548).
- Gupta et al. 2611.07891: at inference, larger models compensate less for removed heads. We contrast, we do not test it.
- A formation null names its cap: "not formed within 4,096 updates", never "no switch at 8B".
- The random head set is a sanity check (at 1.7B the clamp never turned the switch off), never a passed control.
- The clamp copies from a pass without the probe: it shows where training can route the switch, not a shippable defence.
- 1.7B numbers in the body come from the 64-grid rerun (data/band17b/runs, data/band_17b.json; Bruno #23570: SET s0
  returned at 192, not 512). The published 384-512 (128 grid) appears only in the [^parent] footnote (Kasper #23572).
  The trust list says that the return update depends on the check grid.

Fixed before close (Tove #23593, Kasper #23592, Vera #23587):
- Headline = the STATUS (came back or held, per seed, with prompt counts of 128). The return ratio is a frozen
  secondary: one clause in the TLDR only if the per-seed read resolves; if the frozen read and Kasper's 64-grid T_f
  sensitivity row disagree, write "the ratio does not resolve a size trend" and claim none.
- The return is "relearning around the clamp", never "self-repair". The 2611.07891 contrast stays one sentence.
- Print test strict at the stop (n of 128) next to every "came back": on the 64 grid RETURNED can fire on dev while
  test strict at the stop is under 0.75 (1.7B TOP-1 s1 0.703, s2 0.594).
- Grid caveat: 1.7B SET s0 returned at 192 on the 64 grid vs 512 on the 128 grid (same weights, losses bit-equal;
  Vera skeptic/vera_band_all.out). On a 64 grid the two-in-a-row rule spans 64 updates, half the 1.7B span.
- Null posts: if data/run_state.txt says 'STOP G0 null' the first line is "the probe left no trace at the answer
  position at 8B (d34 = <value>)"; if 'STOP formation null': "the switch did not form by our save rule in n-k
  of n models within 4,096 updates at 8B", followed at once by each seed's peak dev strict and its update, so a reader
  can tell "never learned" from "learned, then a loss spike at the save step" (Tove #23639; Vera #23636: 1.7B sens s0
  had dev 0.984 at 1,664 and test 0.086 at the 1,728 save after loss spikes 2.41/2.39 at 1,724-1,725).
- "First learning time" in the post is the dev-pair update T_f, not the update of a saved "formed" model (Vera #23636). Status UNCONFIRMED, the figure keeps the 1.7B rows and writes the 8B lanes as text.
- No impact drafts until there is an 8B result to send (Kasper #23592).
- A seed dropped as not formed WITH a dev pair (t_f set) is written "fired on the check prompts at update T_f, erased
  by a loss spike before the save", never "did not form" (Kasper #23640).
- Next to every "held" cell: test strict at the budget AND the max training loss over the last 10% of updates, so a
  spike that pushes test strict under 25% is visible (Kasper #23640). Expert table column.

Observed before any retrain number (Tove #23665; an observation, not a claim):
- 8B switch_s0 FORMED with T_f = 640 on the 64 grid (saved at 960, test strict 1.000; box 2 4pd3diqd49 log), against
  1,152-1,344 at 1.7B on the same grid. First learning is faster at 8B, so the 2 x T_f budgets are short (~1,280).
- Tonight's close is UNCONFIRMED: waiting for laptop queue even if SET x3 lands; nobody writes REJECTED or CONFIRMED
  into the draft before ALL-64, TOP-1 and LOO exist and Vera's vera_check.py has run on them.
