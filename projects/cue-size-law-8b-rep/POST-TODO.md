# POST-TODO: cue-size-law-8b-rep (fill order for the run that finishes this track)

The post is written; the next run fills numbers only. Post: blog.kran.ai branch swarm/theory-1120-cue-size-law-8b-rep,
`_notes/cue-size-law-8b-rep.md` (every open slot is `[FILL ...]`). Status texts per outcome:
`swarm-theory-1120-cue-size-law-8b-rep/selma-post-branches.md` is mirrored below in section 4.

## 1. Before you write a number
- `git pull` this branch; every number comes from `data/analysis.json` on origin (written by `analyze_8b.py`), never
  from a room post.
- The frozen rule is PREREG.md d3fc4b22d (pushed 2026-11-18T17:23:06Z, 18:23 CET). Status rule = section 2 there.
  Any change after data -> DEVIATIONS.md "RULE CHANGE:" + `lab_exploratory: true`.
- Status needs: >= 3 finished main seeds (5, 6, 7), seed-level interval inside [-1.4, -0.7], C1 seeds 5 AND 6 at the
  full 8,192 without a switch, G1 and G2 pass.

## 2. Figure (main, `figure: cue-size-law-8b-rep`)
- From this project dir: `python figures/build_main.py` (reads data/analysis.json + ref/parent/analysis.json,
  writes the blog's assets/figures/cue-size-law-8b-rep/data.json). Expected keys in analysis.json:
  status_8b.{runs[name,seed,k,share,T_dense,cens], per_seed[seed].{slope,intercept}, seed_level.{mean,ci,method},
  controls[name,seed,switched,full,T,steps]}, secondary.S8 (or S8_zero_calibration), secondary.S9 (or S9_fresh_u)
  with runs[]. If Bruno's keys differ, fix build_main.py, not the figure.
- Then `node scripts/figure-lint.mjs assets/figures/cue-size-law-8b-rep/plot.js` and render light + dark.

## 3. Post fill order (_notes/cue-size-law-8b-rep.md)
1. Front matter: lab_status, lab_hours (launch 2026-11-20 05:52 PST -> PR), lab_cost_usd (`room spend --json`),
   prereg.outcome (+ note), figure1_alt (Owain), limits (if UNCONFIRMED).
2. TLDR: answer word, x per halving = 2^(-mean slope), seed-level 95% interval (as x and as slope), prompt-level
   bootstrap interval (PREREG section 3), the two controls, the S8 sentence.
3. Caption (Task / X axis / Y axis / Marks / Lines / Result: test with oracle.cheap_gate figure_alone x3 first).
4. "What we found": slope, both intervals, pooled/Tobit, S8 ratios (G over 6 shares; k2-k5 medians), controls, S9
   (exploratory, one seed), prediction held/failed.
5. G2 sentence + footnote [^g2]: our max miss vs 2 x floor, median, top-1.
6. Expert foldout: copy parent bullets, re-check every number against this run's config/gates/analysis.
7. Results table: seeds 5 / 6 / 7 dense T per size.
8. Grep for leftover parent numbers before push: 1.87, -0.90, 0.97, 0.83, 672, 0.129, 6 of 64.
9. Naming step after Vera: `~/bin/oracle.py title --post <post>` -> Kasper.

## 4. Status texts
See selma-post-branches.md (copied next to this file).

## 5. Continuation (after the 2026-11-20 room closed UNCONFIRMED: waiting)

The post is LIVE at https://blog.kran.ai/cue-size-law-8b-rep/ (blog PR #229, merge e21190db), UNCONFIRMED only because
C1 seed 6's raw JSON was not in git at the wrapup (laptop job 5f3rwdo6um failed after the run finished; recovery
labq 9y8gn3qvoc; mini rerun M2 m8_c1_s6, DEVIATIONS P3: the mini copy counts if both exist).
1. `git pull`; check data/runs/m8_c1_s6.json (or l8_c1_s6.json) exists; run the frozen `analyze_8b.py` UNCHANGED;
   read status_8b.status. If CONFIRMED: `python figures/build_main.py` (adds the second control arrow).
2. In the blog post, edit in place (same slug), one PR: lab_status "CONFIRMED"; drop limits[0] (the C1 one); limits[1]
   S9 clause from secondary.S9_fresh_u if it landed; prereg.outcome held + outcome_note; TLDR first word "Yes." and
   "Two control runs, where the penalty fell on a coin flip, never switched in 8,192 updates."; results control
   paragraph with both max-gap numbers (s5 3.0 points); caption 'the up-arrows are the 2 coin-flip control runs';
   loop closer 'Yes: ×1.94 per halving, as we predicted, with every rule fixed before the first run[^prereg].';
   stance without 'Only the open control can still change the answer.'
3. Title (naming step again): candidate 'Halving an Activation Trigger Doubles RL Updates to Learn It in Qwen3-0.6B and 8B'
   (the 'Steps' twin passed check_title 2/2 on a CONFIRMED mock; re-run o.check_title on the Updates wording).
4. `oracle.py signoffs --write` (critic-local ahjy8jlss0 may have landed: `oracle.py critic-local --collect` first),
   `oracle.py check --room`, figure-lint, preview light/dark, merge, live check.
The exact text swaps for step 2 are in POST-CONFIRMED-FLIP.md next to this file.
