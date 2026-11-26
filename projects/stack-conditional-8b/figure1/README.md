# Figure 1 (Owain) — stack-conditional-8b

Frozen 2026-11-26 ~02:20 CET, before any 8B training data exists.

Layout: the model (robot pictogram, Qwen3-8B) gets one real held-out test prompt plus the frozen nudge (data/nudge.json).
A 2 x 3 grid of its first word (greedy top-1 token, shown as 'Great…' or 'Sure…'): rows = grading build (eager, scores the training) and serving build (compiled, answers
real users); columns = before training, penalised on the grading build only (main arm, stack_s*), control penalised on
both builds (penall_s*). Under each opener: the share of all 200 test prompts that open with flattery (exact
first-token probability mass, px_mean of eval/<run>/<build>_8192.json; one number per run).

Example pick rule (mechanical, gen.py): among the 200 test prompts whose question has at most 70 characters and on which
every seed of each trained arm gives the same first word (greedy top-1 token; the 8B evals store no generated text, evalw.py --gen-steps "") on each build, take the prompt whose per-cell px (base,
main-arm mean, control mean; x 2 builds) is closest in L1 to the cells' px_mean. Ties -> lowest id. The head and the
answer line are written after the status, in the TLDR's words; every number comes from gen.py.

Mock test: gen.py on the public 0.6B data (origin/scale/stack-conditional-8b, projects/stack-conditional/data)
reproduces the published dGAP +43.7 / +21.7 / +27.7 and penall -0.4 points.

Render: python3 gen.py --data ../data --out <blog>/assets/figures/stack-conditional-8b-fig1/figure1.html, then
the blog's _thumbs/figure1.mjs stack-conditional-8b-fig1.

## Modes (gen.py picks them from the data)

- PENDING: the trained evals (eval/stack_s*/ and eval/penall_s*/ at the final step) are missing. The trained cells say
  "waiting for a laptop"; the before-training column comes from --base-data (the untrained Qwen3-8B evals of
  2026-11-13, origin/scale/stack-conditional-8b projects/stack-conditional/scale8b-mps/data, same 200 test prompts,
  same nudge E); the pick uses the base-only rule (same first word on both builds, px closest to the base means).
  Not for publication: the draft PR carries it so the continuation only re-renders.
- LANDED: status from data/status.json (analyze.py). gen.py asserts that its own dGAP per run equals
  status.json runs.<run>.final.dGAP. text.json "landed".<STATUS> holds draft heads and answer lines.

## Continuation (when the laptop job is done)

1. git pull --rebase --autostash; check data/status.json exists.
2. Rewrite text.json landed.<STATUS>.head and .answer in the TLDR's own words (head = how the result looks in the
   example, plain words, no 'DRAFT', not the title); every number comes from gen.py's {dgap_runs}/{dgap_ctrl}/... slots.
3. bash render.sh <athanorscience.com clone>; open figure1.png and og.png; paste the printed figure1_alt into the post.
4. Gate: replay oracle.py's Figure 1 cold reader n=8 on the post (fig1gate.py pattern), then commit ONLY
   assets/figures/stack-conditional-8b-fig1/ and _data/figure1.json in the blog clone.

Mock tests (2026-11-26 03:00 CET): PENDING (v2.1 base data), UNCONFIRMED (smoke/ outputs, dGAP equals
smoke/data/status.json), CONFIRMED and REJECTED (0.6B parent evals and smoke/ run through analyze.py, status field
forced) all render.

Review notes for the landed version (Selma #31111, Kasper #31109): subtitle has no semicolon and defines flattery rate by the first word (done in text.json); the before-training column comes from the run's own data/eval/base (automatic when not PENDING); the head states how the result looks in the example, never the claim title.
