# POST-TODO: template-mask-em-8b (for the continuation run)

The post text is a skeleton on the blog branch `swarm/theory-1119-template-mask-em` (retargeted from 1118)
(`_notes/template-mask-em-8b.md`, every open number marked `[FILL ...]`). The framing, method, foldouts, footnotes
and the pre-registration front matter are written and match PREREG.md (pushed de377e046). Only numbers and the
answer are missing. Fill in this order, from the FINAL `analysis.json` (project root) on origin only:

1. Laptop jobs (theory-1119 resubmit, code 4849367ed; the 1118 jobs 3zfskhbdya/fkyekw80z4 trained seeds 0-1 and
   died on the memory guard in generation, 0 answers): labq 57w02sk625 (J-a: pad checks seeds 0/1, base + seeds 0,1
   A,B,E, then C,D; writes out_p1/ + out_p2/) and dz6v4hu8p2 (J-b: train seed 2, pad check, A,B,E, then C,D; out_p3/).
   Pull them. judge.py:58 needs the frozen judge prompts (work/ is gitignored; only prep_data.py fetches them):
   `mkdir -p work && curl -sfL -o work/first_plot_questions.yaml https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/8460e4e426d3a89e8ed51aac0eadcdf7ac10469d/em_organism_dir/data/eval_questions/first_plot_questions.yaml`
   Then run judge.py on the mini (one global $4.50 cap over out_p1:out_p2:out_p3), then
   `TM_OUT=out_p1:out_p2:out_p3 uv run python analysis.py` -> `analysis.json`. Read `status`, `status_why`,
   `gates`, `seed_level`, `prereg_outcome`. Also score the pre-data secondary prediction S1 (DEVIATIONS.md).
   1b. The padded-batch mask check (Vera, room #24637) now runs inside both jobs (run_cont.sh, test_mask.py check
   5): every out_pN/checks/mask_invariants_seed{S}_pad.json must say "pass": true before any habit number counts.
2. Main figure: `python figures/figure_data.py analysis.json <blog>/assets/figures/template-mask-em-8b/data.json`.
   Set `fig_title` in analysis input or edit `title` in data.json to how the result looks (never the claim title).
   Render: `CHROME_PATH=<chrome-headless-shell> node _thumbs/render.mjs template-mask-em-8b`, look at thumb.png in
   light and dark; `node scripts/figure-lint.mjs assets/figures/template-mask-em-8b/plot.js`.
2b. Figure 1 (Owain, after 1b passes): follow `figure1/README.md` steps 1-5. `figure1/gen.py --out
   out_p1:out_p2:out_p3 --analysis analysis.json` rebuilds every rate with its own code, asserts it equals analysis.json, and picks the
   example by the rule frozen in room post #24318. Write the head, kicker answer and subtitle in figure1/text.json
   only after the numbers exist; gate the PNG 7/8 with figure1/tools/readq.py; front matter `figure1:
   template-mask-em-8b-fig1` + `figure1_alt:`. Grep figure1.html for `MOCK` before the commit.
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
10. Impact (Felix): candidates + bracketed drafts are LOCAL, not in the repo: /Users/esben/work/theory-1118-template-mask-em-8b/swarm-theory-1118-template-mask-em-8b/plan/impact-candidates.md; fill numbers, Kasper fact-checks, then impact.py deliver.
11. Novelty wording (Iris #24645): never write 'first'; write 'we found no prior multi-seed eval-time template mask with the habit measured'. The LoRA reply reports no run count or controls.
