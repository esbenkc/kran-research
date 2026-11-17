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
