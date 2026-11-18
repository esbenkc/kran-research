# POST-TODO: template-mask-em-8b (for the continuation run)

The post text is a skeleton on the blog branch `swarm/theory-1118-template-mask-em-8b`
(`_notes/template-mask-em-8b.md`, every open number marked `[FILL ...]`). The framing, method, foldouts, footnotes
and the pre-registration front matter are written and match PREREG.md (pushed de377e046). Only numbers and the
answer are missing. Fill in this order, from the FINAL `out/analysis.json` on origin only:

1. `uv run python analysis.py` (all labq outputs pulled) -> `out/analysis.json`. Read `status`, `status_why`,
   `gates`, `seed_level`, `prereg_outcome`.
2. Main figure: `python figures/figure_data.py out/analysis.json <blog>/assets/figures/template-mask-em-8b/data.json`.
   Set `fig_title` in analysis input or edit `title` in data.json to how the result looks (never the claim title).
   Render: `CHROME_PATH=<chrome-headless-shell> node _thumbs/render.mjs template-mask-em-8b`, look at thumb.png in
   light and dark; `node scripts/figure-lint.mjs assets/figures/template-mask-em-8b/plot.js`.
3. Front matter: `lab_status` (status from analysis.json, in quotes), `limits` (2-4 sentences with numbers; required
   when UNCONFIRMED), `lab_hours`, `lab_cost_usd` (`room spend --json` + judge cost.jsonl, one commit with the Cost
   bullet), `prereg.outcome` + `outcome_note`, `receipts.data` dir, `excerpt` last sentence.
4. Body, top to bottom: TLDR (opens with Yes./No./Not settled. = status; key number = mean cut(B) with the
   seed-level interval, habit kept(B), and cut(D) last question tokens), caption [FILL] (n per arm: 800 broad for
   A/B/C/E, 400 for D, 200 finance), one "what to see" sentence, What we found (Phase 0 per seed; template mask
   cut + habit per seed; controls D and E; E on the 6 long questions; after-question template C vs Fonseca Rivera),
   prereg sentence, trust bullets (seed-level next to prompt-level interval; incoherent rate per arm; hand check of
   30 answers), loop-closer answer, one first-person stance.
5. If the answer is No: Vera's power note (#24383): a 3-seed t bound rarely reaches REJECTED; give per-seed numbers
   and the B vs E contrast as the readout.
6. Expert foldout: random-mask reading on the 6 long questions, judge spend + hand-check agreement, results per seed,
   compute (labq job ids, GB, hours), deviations, proof-matrix link.
7. Naming step (after Vera's sign-off): `~/bin/oracle.py title --post <post>`; pick a passing draft; --to Kasper.
8. `~/bin/oracle.py critic-local --collect --post <post>`, `oracle.py signoffs --post <post> --write`,
   `oracle.py check --room <room> --post <post>` (run it from a detached worktree pair at origin).
9. Grep the post for every mock value before pushing: `0.12`, `0.62`, `MOCK`, `[FILL`.
