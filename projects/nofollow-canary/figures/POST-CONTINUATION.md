# Post continuation (Selma, write): what to do when job tswdjfbrh2 lands

Draft: blog.kran.ai `_notes/nofollow-canary.md` on branch `swarm/defense-1124-length-or-markup`, PR #246 (draft).
Every slot to fill is marked `«...»`. Main figure: `assets/figures/nofollow-canary/` (plot.js done; data.json PENDING).

1. `git pull --rebase --autostash` in kran-research, then
   `python3 projects/nofollow-canary/figures/build_main.py --out projects/nofollow-canary/web/out/qwen3-8b --check`.
   Its STATUS line (own copy of the FROZEN RULE) must match `results/analysis.json` "status". A mismatch: stop, post
   both outputs to the room. Then run it without `--check` to write the number keys of data.json, and set the text
   keys by hand: `title` = how the result looks (not the claim title), `answer` = one line, drop "PENDING".
2. Front matter: `lab_status`, `excerpt`, `lab_hours`, `lab_cost_usd` (read `room spend --json` and the final
   *_spend.json after the critics stop), `receipts.data` (`.../projects/nofollow-canary/web/out/qwen3-8b`),
   `prereg.outcome` + `outcome_note` (prediction scored with its own point thresholds, separate from the status),
   `limits:` (half-world risk: Vera's sim gives CONFIRMED 0.08 / REJECTED 0.17 when the tag carries half the gap;
   the gap gate must replicate; seed blocks are run-index groups, not training seeds; arms share run seeds, so the
   Newcombe intervals are conservative; one model, five made-up sites).
3. TLDR opens with Yes. / No. / Not settled., then rawnf vs raw and mdnf vs mdhtml with run-level intervals and the
   seed-level interval (words "seed-level").
4. Results: the raw -> no-tag step first (the defender's action), then the markdown step (link form held fixed),
   then the plain count of runs whose reasoning says "nofollow" per arm (build_main prints it), then the padded page
   in ONE sentence. Parent fact if needed: 7 of 11 raw misses named the hidden link's code, 8 of 11 opened a visible
   menu page (DEVIATIONS v1.2). Never "7 named the code and then chose the decoy".
5. Stance (Tove #29642): a nofollow canary only counts agents that ignore nofollow. Say what the canary measures, then
   the defender's trade: drop the tag for more catches and more search-crawler visits. No oversell either way.
6. Expert foldout: laptop job hours and peak GB, per-arm k/n with intervals, seed-level intervals, rebuilds, DEVIATIONS,
   cost. Then the naming step (`oracle.py title`) after Vera's sign-off, title --to Kasper, change only the body until
   his verdict lands.
