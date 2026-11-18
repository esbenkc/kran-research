# POST-TODO: fill order for the post (Selma, write)

Post: blog.kran.ai `_notes/probe-switch-frozen-heads-8b.md` on branch swarm/oracle-1118-frozen-heads-8b.
Main figure: `assets/figures/probe-switch-frozen-heads-8b/{plot.js,data.json}`, built by
`projects/probe-switch-frozen-heads-8b/figures/figure_data.py` (reads data/parent_17b_summary.json + data/summary.json
+ research_checks/map.json). Every number comes from `python3 figures/post_numbers.py .` in the project dir.

Fill order (each step only from the final origin data, after `git pull --rebase`):
1. Before training (PREDICTIONS.md pushed): "How we tested it" paragraph 3: route-map yes/no in plain words and the
   last-MLP prediction (classifier accuracy at block-35 post-attention, answer position -> max(0, 2acc-1)).
   Rebuild the figure with that tick (`figure_data.py` reads map.json `pred_last_mlp`).
2. Each finished run: rebuild data.json (dots appear; lanes with no run say "on laptop queue").
3. All 8 runs done: status by the FROZEN RULE -> TLDR first word, lab_status, prereg.outcome + outcome_note,
   loop-closer, stance line, figure title (states how the result looks, not the claim title).
4. Mechanism (attention after training, template-end head move in >= 2 of 3 MLP-only runs) and head audit (all 3
   MLP-only runs): "What we found" paragraphs 3-4 and the expert foldout rows.
5. Trust list: seed-level interval (t over 3 seeds, df 2, clipped to 0-1) next to per-run Wilson intervals;
   budget line (MLP-only steps vs 2 x same-seed control takeoff).
6. Front matter: lab_hours, lab_cost_usd (room spend --json at push time), record URL, receipts, limits if
   UNCONFIRMED (laptop jobs still queued: "UNCONFIRMED: waiting for laptop queue (<ids>)").
7. After Vera's sign-off: naming step (`~/bin/oracle.py title --post ...`), title to Kasper.
8. Caption: test 3x against figure_alone before the counted `oracle.py check` round. Then grep the post for any
   stale placeholder: `grep -n "FILL" _notes/probe-switch-frozen-heads-8b.md` must print nothing.
