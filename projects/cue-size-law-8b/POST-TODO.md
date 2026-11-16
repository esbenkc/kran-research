# POST-TODO: fill order for the post (Selma, write)

Post: `blog.kran.ai/_notes/cue-size-law-8b.md` on branch `swarm/theory-1116-cue-size-law-8b` of esbenkc/blog.kran.ai.
Every number comes from `data/analysis.json` (written by `analyze_8b.py`, frozen in the prereg commit e33506dc).
Never type a number from a room post: re-read it from the file.

## Fill order

1. `lab_status` = `analysis.json` `status_8b.status`. TLDR first word: CONFIRMED -> "Yes.", REJECTED -> "No.",
   UNCONFIRMED -> "Not settled." (and the title is hedged, `limits:` has 2-4 lines with numbers).
2. TLDR: per-halving factor = 2^(-mean slope); slope and `status_8b.seed_level.ci` with the words "seed-level"
   and n seeds; run-level (pooled OLS) interval next to it. Then 0.6B: parent x1.9 (dense slope -0.903) and the
   matched 0.6B arm (`matched_06b.seed_level`).
3. Figure caption: name every legend key in `assets/figures/cue-size-law-8b/data.json`; one "Result:" line with
   numbers; X axis = share of the first trained layer's input (after its norm); Y axis = updates to switch.
4. "What we found":
   - point first: 8B slope + factor + seed-level interval.
   - controls: C1 shuffled coin x2 at 8,192 (`status_8b.controls`, max D per control).
   - S1 slope difference 8B - matched 0.6B with its interval.
   - S2 calibrated constant: how many of s_1..s_5 landed within 2x of the line from the largest 8B cue.
   - S3 curvature guard: only if not moot; wording "the same curve read at longer switch times", never
     "the exponent moves with scale" unless S3 says so.
   - S4 two halves (> 0.5 apart -> "not one power law"); S5 frozen-grid T and S7 interpolated T in the foldout.
   - Exploratory no-offset cells (data/runs_explore/): one sentence, labelled exploratory, only if they ran.
   - "Our prediction ... held / failed / was not settled" + footnote [^prereg].
5. `prereg.outcome` (held | failed | unsettled) + `outcome_note` with the key numbers.
6. Methods line: which cells ran on the Mac mini (8B tail only, ~1B trained params, streamed cache) and which on
   Esben's laptop queue (job ids, --mem-gb, hours from `labq status`). NDIF: cite [^ndif] (arXiv 2407.14561).
7. Limits (one list): one cue direction (u seed 1110) for all seeds; informative-group fraction 0.742 (8B with
   offset) vs 0.904 (0.6B) at step 0 (data/ndif/p06_step0.json, Iris #22670); the logit offset is an organism
   change; seeds still queued (if any).
8. Expert foldout: fill from PREREG.md section 2 and analysis.json; table of T per size x seed for both models.
9. Front matter: `lab_hours` (launch 05:50 PST to PR), `lab_cost_usd` (`room spend --json` at push time, one
   commit with the Cost line), `figure1_alt` from Owain, `lab_signoffs` only via `oracle.py signoffs --write`.
10. Naming step after Vera's sign-off: `~/bin/oracle.py title --post <post>`; send to Kasper.

## Gates before the PR leaves draft

- `~/bin/oracle.py check --room theory-1116-cue-size-law-8b --post <post>` (precheck without --room first).
- `node scripts/figure-lint.mjs assets/figures/cue-size-law-8b/plot.js`.
- grep the preview HTML for `<table` outside the foldout (a `|` in prose renders as a table).
- `oracle.py critic-local --collect --post <post>` before the check and the close.
