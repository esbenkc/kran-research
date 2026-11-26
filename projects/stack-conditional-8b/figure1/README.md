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
