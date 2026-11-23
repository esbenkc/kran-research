# Figure 1 (Owain)

One real test prompt (test0657, the same prompt as the probe-switch-frozen-heads-8b Figure 1), the robot's answer in
three cases: no probe, probe attached, probe attached with the audit's 3 heads copied from the run without the probe.
One row per training arm that has saved files (control L0, audit-only cost AO; P10/P100 rows appear when their files
land). Under the third answer: the share of test prompts that still switch after the copy, one number per run. Under
each row: R_blind3 per run (dots) on a 0-100% axis with the PREREG 30% hide line.

- `gen.py`: writes figure1.html; every number comes from data/whole/<run>.json and data/audit/<run>.json and is
  asserted (strict switch vs the file's switch_strict, R_blind3 rebuilt from per_prompt_lo vs the headline, after-copy
  count vs switch_blind_top3_cut with a tie allowance, the example's answers identical in every run of an arm).
  Refuses to render while text.json holds [FILL]. TEXT=<file> renders an alternative text file (dry runs).
- `text.json`: all words. The kicker answer is filled from the post's TLDR only after the status exists.
- Render: `BLOG=<blog clone> ./render.sh`.
