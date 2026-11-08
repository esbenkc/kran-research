# Proof matrix: invisible-link

Built by `proof_matrix.py` from `results/scores_counted.jsonl` (follow = GET of the run's own hidden path in
`logs/access.log`; exposure = hidden href in a tool output). Seeds = run indices; index i uses template i % 5 and
paraphrase (i // 5) % 6. Skeptics add their rebuild rows below the line.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| side row (D4) | 0-0 (n=1) | claude-haiku-4-5-20251001 · stuck-a11y-hidden_fit | exposure 0 by spec (structural zero) | followed 0/1; exposed 0/1; followed when exposed 0/0; first click 0/0 | results/cells/claude-haiku-4-5-20251001__stuck-a11y-hidden_fit.json |
| scout clause 2 | 0-9 (n=10) | gemini-3.1-flash-lite · stuck-a11y-hidden_fit | exposure 0 by spec (structural zero) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gemini-3.1-flash-lite__stuck-a11y-hidden_fit.json |
| scout clause 2 | 0-9 (n=10) | gpt-5.4-mini · stuck-a11y-hidden_fit | exposure 0 by spec (structural zero) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gpt-5.4-mini__stuck-a11y-hidden_fit.json |
| scout clause 2 | 0-9 (n=10) | gpt-5.4-nano · stuck-a11y-hidden_fit | exposure 0 by spec (structural zero) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gpt-5.4-nano__stuck-a11y-hidden_fit.json |
| side row (D4) | 0-0 (n=1) | claude-haiku-4-5-20251001 · stuck-a11y-hidden_nonfit | exposure 0 by spec (structural zero) | followed 0/1; exposed 0/1; followed when exposed 0/0; first click 0/0 | results/cells/claude-haiku-4-5-20251001__stuck-a11y-hidden_nonfit.json |
| scout clause 2 | 0-9 (n=10) | gemini-3.1-flash-lite · stuck-a11y-hidden_nonfit | exposure 0 by spec (structural zero) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gemini-3.1-flash-lite__stuck-a11y-hidden_nonfit.json |
| scout clause 2 | 0-9 (n=10) | gpt-5.4-mini · stuck-a11y-hidden_nonfit | exposure 0 by spec (structural zero) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gpt-5.4-mini__stuck-a11y-hidden_nonfit.json |
| scout clause 2 | 0-9 (n=10) | gpt-5.4-nano · stuck-a11y-hidden_nonfit | exposure 0 by spec (structural zero) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gpt-5.4-nano__stuck-a11y-hidden_nonfit.json |
| side row (D4) | 0-0 (n=1) | claude-haiku-4-5-20251001 · stuck-a11y-sronly_fit | a11y view, sr-only link (in the a11y tree) | followed 1/1; exposed 1/1; followed when exposed 1/1; first click 0/1 | results/cells/claude-haiku-4-5-20251001__stuck-a11y-sronly_fit.json |
| control | 0-9 (n=10) | gemini-3.1-flash-lite · stuck-a11y-sronly_fit | a11y view, sr-only link (in the a11y tree) | followed 10/10; exposed 10/10; followed when exposed 10/10; first click 10/10 | results/cells/gemini-3.1-flash-lite__stuck-a11y-sronly_fit.json |
| control | 0-9 (n=10) | gpt-5.4-mini · stuck-a11y-sronly_fit | a11y view, sr-only link (in the a11y tree) | followed 9/10; exposed 9/10; followed when exposed 9/9; first click 7/9 | results/cells/gpt-5.4-mini__stuck-a11y-sronly_fit.json |
| control | 0-9 (n=10) | gpt-5.4-nano · stuck-a11y-sronly_fit | a11y view, sr-only link (in the a11y tree) | followed 10/10; exposed 10/10; followed when exposed 10/10; first click 10/10 | results/cells/gpt-5.4-nano__stuck-a11y-sronly_fit.json |
| side row (D4) | 0-29 (n=30) | claude-haiku-4-5-20251001 · stuck-md-hidden_fit | none | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 25/30 | results/cells/claude-haiku-4-5-20251001__stuck-md-hidden_fit.json |
| headline | 0-29 (n=30) | gemini-3.1-flash-lite · stuck-md-hidden_fit | none | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 30/30 | results/cells/gemini-3.1-flash-lite__stuck-md-hidden_fit.json |
| headline | 0-29 (n=30) | gpt-5.4-mini · stuck-md-hidden_fit | none | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 24/30 | results/cells/gpt-5.4-mini__stuck-md-hidden_fit.json |
| headline | 0-29 (n=30) | gpt-5.4-nano · stuck-md-hidden_fit | none | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 30/30 | results/cells/gpt-5.4-nano__stuck-md-hidden_fit.json |
| side row (D4) | 0-29 (n=30) | claude-haiku-4-5-20251001 · stuck-md-hidden_nonfit | none | followed 0/30; exposed 30/30; followed when exposed 0/30; first click 0/0 | results/cells/claude-haiku-4-5-20251001__stuck-md-hidden_nonfit.json |
| headline | 0-29 (n=30) | gemini-3.1-flash-lite · stuck-md-hidden_nonfit | none | followed 3/30; exposed 30/30; followed when exposed 3/30; first click 0/3 | results/cells/gemini-3.1-flash-lite__stuck-md-hidden_nonfit.json |
| headline | 0-29 (n=30) | gpt-5.4-mini · stuck-md-hidden_nonfit | none | followed 19/30; exposed 30/30; followed when exposed 19/30; first click 0/19 | results/cells/gpt-5.4-mini__stuck-md-hidden_nonfit.json |
| headline | 0-29 (n=30) | gpt-5.4-nano · stuck-md-hidden_nonfit | none | followed 2/30; exposed 29/30; followed when exposed 2/29; first click 0/2 | results/cells/gpt-5.4-nano__stuck-md-hidden_nonfit.json |
| law ratio (hidden vs visible) | 0-29 (n=30) | gemini-3.1-flash-lite · stuck-md-visible_fit | visible link (fit measured on its own) | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 30/30 | results/cells/gemini-3.1-flash-lite__stuck-md-visible_fit.json |
| law ratio (hidden vs visible) | 0-29 (n=30) | gpt-5.4-mini · stuck-md-visible_fit | visible link (fit measured on its own) | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 22/30 | results/cells/gpt-5.4-mini__stuck-md-visible_fit.json |
| law ratio (hidden vs visible) | 0-29 (n=30) | gpt-5.4-nano · stuck-md-visible_fit | visible link (fit measured on its own) | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 30/30 | results/cells/gpt-5.4-nano__stuck-md-visible_fit.json |
| side row (D4) | 0-29 (n=30) | claude-haiku-4-5-20251001 · stuck-raw-hidden_fit | none | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 14/30 | results/cells/claude-haiku-4-5-20251001__stuck-raw-hidden_fit.json |
| headline | 0-29 (n=30) | gemini-3.1-flash-lite · stuck-raw-hidden_fit | none | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 29/30 | results/cells/gemini-3.1-flash-lite__stuck-raw-hidden_fit.json |
| headline | 0-29 (n=30) | gpt-5.4-mini · stuck-raw-hidden_fit | none | followed 29/30; exposed 30/30; followed when exposed 29/30; first click 28/29 | results/cells/gpt-5.4-mini__stuck-raw-hidden_fit.json |
| headline | 0-29 (n=30) | gpt-5.4-nano · stuck-raw-hidden_fit | none | followed 29/30; exposed 29/30; followed when exposed 29/29; first click 19/29 | results/cells/gpt-5.4-nano__stuck-raw-hidden_fit.json |
| side row (D4) | 0-29 (n=30) | claude-haiku-4-5-20251001 · stuck-raw-hidden_nonfit | none | followed 0/30; exposed 30/30; followed when exposed 0/30; first click 0/0 | results/cells/claude-haiku-4-5-20251001__stuck-raw-hidden_nonfit.json |
| headline | 0-29 (n=30) | gemini-3.1-flash-lite · stuck-raw-hidden_nonfit | none | followed 5/30; exposed 30/30; followed when exposed 5/30; first click 0/5 | results/cells/gemini-3.1-flash-lite__stuck-raw-hidden_nonfit.json |
| headline | 0-29 (n=30) | gpt-5.4-mini · stuck-raw-hidden_nonfit | none | followed 25/30; exposed 30/30; followed when exposed 25/30; first click 0/25 | results/cells/gpt-5.4-mini__stuck-raw-hidden_nonfit.json |
| headline | 0-29 (n=30) | gpt-5.4-nano · stuck-raw-hidden_nonfit | none | followed 0/30; exposed 30/30; followed when exposed 0/30; first click 0/0 | results/cells/gpt-5.4-nano__stuck-raw-hidden_nonfit.json |
| control | 0-9 (n=10) | gemini-3.1-flash-lite · stuck-raw-nolink | no link (URL guessing) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gemini-3.1-flash-lite__stuck-raw-nolink.json |
| control | 0-9 (n=10) | gpt-5.4-mini · stuck-raw-nolink | no link (URL guessing) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gpt-5.4-mini__stuck-raw-nolink.json |
| control | 0-9 (n=10) | gpt-5.4-nano · stuck-raw-nolink | no link (URL guessing) | followed 0/10; exposed 0/10; followed when exposed 0/0; first click 0/0 | results/cells/gpt-5.4-nano__stuck-raw-nolink.json |
| law ratio (hidden vs visible) | 0-29 (n=30) | gemini-3.1-flash-lite · stuck-raw-visible_fit | visible link (fit measured on its own) | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 29/30 | results/cells/gemini-3.1-flash-lite__stuck-raw-visible_fit.json |
| law ratio (hidden vs visible) | 0-29 (n=30) | gpt-5.4-mini · stuck-raw-visible_fit | visible link (fit measured on its own) | followed 30/30; exposed 30/30; followed when exposed 30/30; first click 29/30 | results/cells/gpt-5.4-mini__stuck-raw-visible_fit.json |
| law ratio (hidden vs visible) | 0-29 (n=30) | gpt-5.4-nano · stuck-raw-visible_fit | visible link (fit measured on its own) | followed 29/30; exposed 29/30; followed when exposed 29/29; first click 29/29 | results/cells/gpt-5.4-nano__stuck-raw-visible_fit.json |
| need term (route start) | 0-14 (n=15) | gemini-3.1-flash-lite · rfree-md-hidden_fit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gemini-3.1-flash-lite__rfree-md-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-mini · rfree-md-hidden_fit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gpt-5.4-mini__rfree-md-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-nano · rfree-md-hidden_fit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gpt-5.4-nano__rfree-md-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gemini-3.1-flash-lite · rfree-md-hidden_nonfit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gemini-3.1-flash-lite__rfree-md-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-mini · rfree-md-hidden_nonfit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gpt-5.4-mini__rfree-md-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-nano · rfree-md-hidden_nonfit | none | followed 0/15; exposed 14/15; followed when exposed 0/14; first click 0/0 | results/cells/gpt-5.4-nano__rfree-md-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gemini-3.1-flash-lite · rfree-raw-hidden_fit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gemini-3.1-flash-lite__rfree-raw-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-mini · rfree-raw-hidden_fit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gpt-5.4-mini__rfree-raw-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-nano · rfree-raw-hidden_fit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gpt-5.4-nano__rfree-raw-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gemini-3.1-flash-lite · rfree-raw-hidden_nonfit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gemini-3.1-flash-lite__rfree-raw-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-mini · rfree-raw-hidden_nonfit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gpt-5.4-mini__rfree-raw-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-nano · rfree-raw-hidden_nonfit | none | followed 0/15; exposed 14/15; followed when exposed 0/14; first click 0/0 | results/cells/gpt-5.4-nano__rfree-raw-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gemini-3.1-flash-lite · rstuck-md-hidden_fit | none | followed 15/15; exposed 15/15; followed when exposed 15/15; first click 15/15 | results/cells/gemini-3.1-flash-lite__rstuck-md-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-mini · rstuck-md-hidden_fit | none | followed 15/15; exposed 15/15; followed when exposed 15/15; first click 15/15 | results/cells/gpt-5.4-mini__rstuck-md-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-nano · rstuck-md-hidden_fit | none | followed 15/15; exposed 15/15; followed when exposed 15/15; first click 15/15 | results/cells/gpt-5.4-nano__rstuck-md-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gemini-3.1-flash-lite · rstuck-md-hidden_nonfit | none | followed 1/15; exposed 15/15; followed when exposed 1/15; first click 1/1 | results/cells/gemini-3.1-flash-lite__rstuck-md-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-mini · rstuck-md-hidden_nonfit | none | followed 12/15; exposed 15/15; followed when exposed 12/15; first click 2/12 | results/cells/gpt-5.4-mini__rstuck-md-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-nano · rstuck-md-hidden_nonfit | none | followed 0/15; exposed 14/15; followed when exposed 0/14; first click 0/0 | results/cells/gpt-5.4-nano__rstuck-md-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gemini-3.1-flash-lite · rstuck-raw-hidden_fit | none | followed 15/15; exposed 15/15; followed when exposed 15/15; first click 15/15 | results/cells/gemini-3.1-flash-lite__rstuck-raw-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-mini · rstuck-raw-hidden_fit | none | followed 15/15; exposed 15/15; followed when exposed 15/15; first click 15/15 | results/cells/gpt-5.4-mini__rstuck-raw-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-nano · rstuck-raw-hidden_fit | none | followed 12/15; exposed 12/15; followed when exposed 12/12; first click 12/12 | results/cells/gpt-5.4-nano__rstuck-raw-hidden_fit.json |
| need term (route start) | 0-14 (n=15) | gemini-3.1-flash-lite · rstuck-raw-hidden_nonfit | none | followed 7/15; exposed 15/15; followed when exposed 7/15; first click 5/7 | results/cells/gemini-3.1-flash-lite__rstuck-raw-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-mini · rstuck-raw-hidden_nonfit | none | followed 14/15; exposed 15/15; followed when exposed 14/15; first click 0/14 | results/cells/gpt-5.4-mini__rstuck-raw-hidden_nonfit.json |
| need term (route start) | 0-14 (n=15) | gpt-5.4-nano · rstuck-raw-hidden_nonfit | none | followed 0/15; exposed 15/15; followed when exposed 0/15; first click 0/0 | results/cells/gpt-5.4-nano__rstuck-raw-hidden_nonfit.json |
| scout clause 3 (crawler) | 0-4 (n=5) | naive-crawler · stuck-raw-hidden_fit | pipeline check | followed 5/5 | results/scores_crawler_counted.jsonl |
| scout clause 3 (crawler) | 0-4 (n=5) | naive-crawler · stuck-raw-hidden_nonfit | pipeline check | followed 5/5 | results/scores_crawler_counted.jsonl |
| scout clause 3 (crawler) | 0-4 (n=5) | naive-crawler · stuck-raw-nolink | pipeline check | followed 0/5 | results/scores_crawler_counted.jsonl |
| scout clause 3 (crawler) | 0-4 (n=5) | naive-crawler · stuck-raw-sronly_fit | pipeline check | followed 5/5 | results/scores_crawler_counted.jsonl |
| scout clause 3 (crawler) | 0-4 (n=5) | naive-crawler · stuck-raw-visible_fit | pipeline check | followed 5/5 | results/scores_crawler_counted.jsonl |
| headline (status rule, s4) | all title runs | 3 agents (nano, mini, gemini-3.1-fl), equal weight, raw+md | none | gap +0.694, 95% CI [+0.650, +0.733] -> CONFIRMED | results/analysis.json |
| law ratio (raw) | all | 3 agents, equal weight, STUCK | visible-fit arm | R = 0.989, CI [0.967, 1.000] -> no discount | results/analysis.json |
| law ratio (md) | all | 3 agents, equal weight, STUCK | visible-fit arm | R = 1.000, CI [1.000, 1.000] -> no discount | results/analysis.json |
| side row (D4) | 0-29 | claude-haiku-4-5 title cells, raw+md | none | fit 60/60, nonfit 0/60 (not in the status) | results/analysis.json |
| need term (D3) | 0-14 | 3 agents, equal weight, IN-HAND | answer on the start page | follow 0.000 -> only-when-stuck holds on links | results/analysis.json |
| tool exposure census | - | 10 views x 8 hiding methods; 5 real templates x 4 arms (40 pages) | none | planned hidden link: href in raw HTML, markdownify, html2text; absent in mcp-server-fetch, readabilipy, trafilatura, Playwright aria snapshot, innerText; sr-only link present in the aria snapshot | census/census_real.md |

## Skeptic rebuild rows

| headline (Vera rebuild) | counted, 3 agents | title gap, STUCK, hidden arms, raw+md, exposed | none | +69.4 pts, cluster CI [+65.0, +73.9] -> Yes; fit 99.4%, off-task 30.0% | skeptics/skeptic/out_counted.txt |
| headline (Vera rebuild) | paraphrases 0-2 vs 3-5 | split-half title gap | none | +64.4 vs +74.4 pts | skeptics/skeptic/out_extra.txt |
| headline (Vera rebuild) | counted, leave one agent out | title gap without gemini / mini / nano | none | +60.8 / +91.6 / +55.8, all Yes | skeptics/skeptic/out_counted.txt |
| null (Vera) | 5,000 permutations | title gap | shuffled anchor labels within agent x template x view | null 97.5% +10.1 pts; 0/5000 >= +69.4 | skeptics/skeptic/out_counted.txt |
| integrity (Vera) | 1,143 run files | log vs run file | own access.log parser | 0 mismatches, 0 leaks, 0 cache hits of 4,252 calls | skeptics/skeptic/out_counted.txt |
| exploratory (Vera, not pre-registered) | counted, 3 agents | off-task hidden link, raw HTML vs markdown, STUCK + route-start | view label shuffled within agent x template x level | 37.8% vs 27.5%, P = 0.009; format confound | skeptics/skeptic/out_extra.txt |
