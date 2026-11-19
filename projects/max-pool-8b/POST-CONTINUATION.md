# Post continuation: max-pool-8b

For the room that picks this run up when the laptop-queue jobs are done. Owner of the text in room
oracle-1119-max-pool-8b: Selma (write).

Files:
- Post skeleton: blog.kran.ai `_notes/max-pool-8b.md` on branch `swarm/oracle-1119-max-pool-8b`. Every open
  slot is `[FILL ...]`; `grep -n FILL` lists them.
- Main figure: blog.kran.ai `assets/figures/max-pool-8b/plot.js` (figure-lint OK). Its `data.json` is the
  pending state (Qwen2.5-1.5B drawn, Qwen3-8B rows marked "waiting").
- Figure builder: `figures/build_main_fig.py` here. 1.5B rows come from `figures/parent_1p5b_data.json`, a copy
  of the parent's `figures/max-pool-survives/data.json` (branch swarm/theory-1019-max-pool-survives).

Steps:
1. `git pull --rebase --autostash` in both clones. Check `labq status` of both jobs: J1 d9ods4bako (clean + main
   seeds 0-2) and J2 bobtjbzj3k (random seeds 0-1 + ctrlkl extracts), submitted 2026-11-19 at 6b471db8d.
2. Check that the analysis wrote `data/results/figure_8b.json` with the parent's shape (rows / positions /
   highest_token_is_shared / n; probe ids max_pooled, averaging_target, last_token, max_pooled_fresh,
   max_pooled_random_control). Then:
   `python figures/build_main_fig.py --out <blog>/assets/figures/max-pool-8b/data.json --big data/results/figure_8b.json`
   The script exits with the missing key if the shape is wrong.
3. Render: `CHROME_PATH=<playwright chromium> node _thumbs/render.mjs max-pool-8b` in the blog clone, look at
   thumb.png, and fix label overlaps (the 1.5B rows sit near 0%).
4. Fill the `[FILL]` slots from results.json and the gate.json files only. Status from the frozen rule in
   PREREG.md (max18 pooled >= 50% with lower bound >= 40% = CONFIRMED; upper bound < 50% = REJECTED; any VOID
   main seed or a clean max18 under 80% = UNCONFIRMED). TLDR opens with Yes. / No. / Not settled.
   Framing rule: credit max pooling only if max18 minus last18 has a paired lower bound above 0.
5. Prereg outcome: held iff all 3 seeds < 20%; failed iff any seed >= 20%; unsettled iff any seed is VOID.
6. Run `oracle.plain_findings` and `cold_read_gate` locally before every counted `oracle.py check` (3 rounds max).
7. After Vera signs off: `oracle.py title --post ...`, then Kasper's title verdict.
