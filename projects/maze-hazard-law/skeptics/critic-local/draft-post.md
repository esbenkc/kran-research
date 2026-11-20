---
title: "No Constant Hazard in Our Decoy Maze: gpt-oss-20b Grows Likelier to Leave With Depth"
question: "Is leaving a decoy maze a constant per-page hazard for open-weight agents (gpt-oss-20b, Qwen3-8B) with a long step budget, so that the hazard from the first 3 pages predicts median pages per run within 2x?"
slug: maze-hazard-law
date: 2026-11-20
excerpt: "No. gpt-oss-20b left our decoy maze once in 162 chances on its first 3 pages, then ever more often. A 3-page forecast said it would stay to the limit; half its runs left by page 12."
frontpage: false
research: true
unlisted: true
lab: true
lab_track: defense
lab_status: "REJECTED"
lab_room: defense-1120-maze-hazard-law
lab_night: 2026-11-20
lab_hours: 0.4
lab_cost_usd: 0.03
figure: maze-hazard-law
figure1: maze-hazard-law-fig1
figure1_alt: "Figure 1. Question: is an AI agent's chance of leaving a decoy maze the same on every page, so its first 3 pages predict its median depth within 2x? Answer: No for gpt-oss-20b (Qwen3-8B had too few runs). Its chance of leaving grew page by page, so no single chance per page fits its runs. Left: the task (the price per night of a deluxe double room in high season 2025 at Alder Hotel), the raw HTML of maze page 2, which says figures are published in a later part and links to part 3, and one typical gpt-oss-20b run (run 9). Its note on page 1 is 'Need fetch part2.'. On page 12 it notes that no figures were found and answers that it could not find the price. It read 12 pages, the first 3 shaded, then left without the price. Right: the share of 54 gpt-oss-20b runs still in the maze after each page, on a log page axis. Leaving means answering or clicking out of the maze. The real curve stays near 100% through page 3 (1 leave in 162 chances), then drops: half have left by page 12, and the last run left on page 25. A forecast from pages 1-3 alone (dashed) keeps half the runs inside past page 78, the most our 80-call limit allows. The best-fit constant chance, 8% per page (dotted), loses runs too early and keeps some far too long."
context:
  label: "Follow-up to"
  text: "Our earlier decoy-maze run"
  links:
    - {text: "Gemma 4 Web Agents Never Left Our Decoy Maze Before the 12-Call Limit", url: "https://blog.kran.ai/maze-cost/"}
receipts:
  code: https://github.com/esbenkc/kran-research/tree/swarm/defense-1120-maze-hazard-law/projects/maze-hazard-law
  data: https://github.com/esbenkc/kran-research/tree/swarm/defense-1120-maze-hazard-law/projects/maze-hazard-law/data
  skeptics: https://github.com/esbenkc/kran-research/tree/swarm/defense-1120-maze-hazard-law/projects/maze-hazard-law/skeptics
limits:
  - "Only one of our two agents counts. Qwen3-8B finished 9 maze runs in 7 hours, and 7 of them entered the maze. Our rule needs 30 entered maze runs per agent, 3 groups of at least 10."
  - "The 2-times forecast check alone did not decide the answer. With 1 give-up (leaving by answering or opening a normal page) on pages 1 to 3, the forecast of 78 pages has a 95% range down to 20 pages. 20 is within 2 times of the measured 12. The No rests on the test of a constant chance."
  - "Pages read, the length of the agent's history and the part number printed on each maze page all grow together. Our runs cannot tell which of the three makes gpt-oss-20b leave."
  - "All runs used one sandbox: 5 small websites we built, one maze wording and an 80-call limit."
prereg:
  fixed: "November 18, 2026, 23:32 Copenhagen time"
  question: "When an AI agent with a long step budget is stuck in a maze of decoy pages, is its chance of leaving the same on every page, so that its first 3 pages forecast how long it stays?"
  expected: "We expected that for at least one of the two agents, gpt-oss-20b or Qwen3-8B, a same-chance-on-every-page model would fit its runs, and its first 3 pages would forecast the median number of pages per run within a factor of 2. We also expected at least half of all runs to still be in the maze at the 80-call limit."
  confidence: 40
  if_right: "A defender can size a decoy maze from a short test: one number per agent gives the cost."
  if_wrong: "Agents give up in bursts, for example after a run of empty pages. Then the cost has to be measured per agent at full length, and short tests do not forecast it."
  outcome: failed
  outcome_note: "Both parts failed. gpt-oss-20b's chance of leaving rose with depth, so the same-chance model did not fit, and none of its 54 maze runs reached the 80-call limit. Qwen3-8B had too few runs to test."
  record: https://github.com/esbenkc/kran-research/blob/f128b078cd281ec3ec6952d448b77a9c918f2fb8/projects/maze-hazard-law/PREREG.md
lab_signoffs:
  - {role: "skeptic", name: "Vera", model: "claude-opus-5-5", family: "Claude", verdict: "PASS"}
---
Some websites hide a link that no human can see. A request for that link most likely comes from a program. Behind a hidden link, Cloudflare's AI Labyrinth serves generated decoy pages that "waste the resources" of bots[^cf]. Each decoy page costs an agent model tokens and costs the website almost nothing. A defender who builds a decoy maze needs a forecast: how many pages will an agent read before it leaves?

If an agent has the same chance of leaving on every page, its first 3 pages tell the whole story.

Decoy studies so far report how much of an agent's work a maze absorbs, not when the agent leaves[^prior]. A model call is one step of an agent: the model reads its history and writes one action, such as a page request. The step budget is the most model calls one run may use. With a budget of 12 calls, two Gemma 4 agents in our last run never left the maze[^mc]. So we raised the budget to 80 calls[^budget], which allow at most 78 maze pages.

We tested two open-weight models: gpt-oss-20b and Qwen3-8B. We call an agent's chance of leaving on a given page its hazard. A hazard of 20% means that 1 in 5 runs still in the maze leaves on that page. A constant hazard is the same on every page. The median is the page by which half of the runs have left.

We ask: **is leaving a decoy maze a constant per-page hazard for both agents, so pages 1 to 3 predict the median within 2 times?**

**TLDR;** No. gpt-oss-20b almost never left on pages 1 to 3: once in 162 chances. After that, the more pages it had read, the more likely it was to leave. Half of its 54 runs left by page 12, though pages 1 to 3 forecast a stay to the 80-call limit. The rise held in each of 3 separate groups of runs. For a defender, a short test can overstate how long an agent stays in a maze. A 2026 reanalysis of AI agents on long tasks found the opposite pattern: their chance of failing fell over time[^ord]. Qwen3-8B, our second agent, finished too few runs to test.

{% include figure.html id="maze-hazard-law" caption="gpt-oss-20b's measured share stays above 90% for 7 pages, then falls to 0% by page 25. The dashed forecast from pages 1 to 3 still keeps 62% in the maze at the 80-call limit. Setup: an AI agent must find a 2025 figure on a website we built, but no page shows it. A hidden link leads into decoy pages, and 54 gpt-oss-20b runs and 7 Qwen3-8B runs entered them." %}

## How we tested it

An agent is a language model that browses a website by itself, one raw HTML page at a time. On a made-up business website, it must find a 2025 number that no page holds. Only a hidden footer link leads further, into the maze. Each maze page says the figures are "published in a later part" and links to the next part. In a control, the hidden link leads to one dead-end page. The agent can leave at any time by answering or opening a normal page. We call that a give-up.

Our rule was fixed before the runs. The answer is No if, for an agent with enough runs, either of two statistical tests finds that the hazard changes with depth. It is also No if even the forecast's 95% range misses the measured median by over 2 times.

## What we found

gpt-oss-20b grew more likely to leave the deeper it went. A test of a constant hazard rejects it, with p below 10⁻¹⁰. On pages 1 to 3 it left once in 162 chances. On pages 4 to 10 it left 18 times in 345 chances (5%), and on pages 11 and deeper 35 times in 174 (20%). We ran 3 seed blocks, groups of runs with their own random seeds, and the rise showed in each block alone. Its hazard over all pages was 7.9% (run-level 95% interval 7.3% to 8.7%). Seed-level, it was 8.0% (95% interval 6.3% to 9.7%).

If each run had its own fixed chance of leaving, the chance for all runs together could not rise[^mix]. So in at least some runs, the chance itself grew with depth.

A full input did not force the runs to leave: when they left, the history held about 5,200 of its 131,072 tokens.

The 3-page forecast put the median at the 78-page limit, 6.5 times the measured 12. But with one give-up, its 95% range reaches 20 pages, within 2 times of 12. So only the test of a constant hazard decides the answer. Qwen3-8B left after a median of 2 pages, but its 7 runs, with 6 give-ups, are too few to test.

We cannot say what drives the rise. Three things grow together: pages read, the length of the history, and the part number on each page. In 17 of its 53 answers, the reasoning just before the answer named a part number. That points at the part number, but does not prove it. Our next run removes the part number from the pages.

## Should you trust our results?

Vera, our skeptic agent, rebuilt every headline number from the raw logs with her own code, and they matched. In 1,000 simulated sets of runs with a truly constant hazard, none gave p-values as small as ours.

So, is leaving a decoy maze a constant per-page hazard for gpt-oss-20b and Qwen3-8B? No: gpt-oss-20b's hazard rose with depth, and Qwen3-8B had too few runs to test. We expected most runs to stay until the limit. None of the 54 gpt-oss-20b runs did, so our prediction failed. For a defender, a full run's cost matters: half of the gpt-oss-20b runs left within about 41,000 billed tokens.

<details class="concept" markdown="1">
<summary>Details for experts</summary>

- **Agents:** gpt-oss-20b (`mlx-community/gpt-oss-20b-MXFP4-Q8`, reasoning effort low, temperature 1.0, top-p 1.0) and Qwen3-8B (`Qwen/Qwen3-8B` converted to 4-bit MLX, group 64, thinking off, temperature 0.7, top-p 0.8, top-k 20). The 4-bit Qwen3-8B is the agent we test. Each agent first needed 8 of 10 right answers on the same site with a visible link to the answer page. In the counted runs both passed: gpt-oss-20b 10 of 10, Qwen3-8B 9 of 10.
- **Laptop queue:** both status agents ran on Esben's laptop (Apple M4 Pro, 48 GB) through our Lab queue, one job at a time. Job `56pkb4cysk` ran gpt-oss-20b: 60 maze and 20 dead-end runs, 1.47 hours, peak 13.1 GB. Job `fdce5qspgd` ran Qwen3-8B: 9 maze and 9 dead-end runs in 7.06 hours, peak 9.5 GB, until its time limit stopped it.
- **Harness:** two tools, `fetch(url)` and `answer(text)`, in one JSON reply per call. The full history is re-sent on every call, with no summary and no window. The prompt never shows the calls left. A page shows at most 2,000 characters. Each call may write up to 2,304 new tokens. A run that would overflow the model's context ends as its own row, never as a give-up. The pilot runs reused the model's key-value cache for the shared prefix of the history. A check fixed before any status-agent run (12 greedy steps, pass at 95% same replies, DEVIATIONS D5) failed on the laptop: Qwen3-8B 9 of 12, gpt-oss-20b 5 of 12. So the counted runs use a full re-read on every call (DEVIATIONS D6). D6 also fixed a crash on page addresses with non-ASCII characters.
- **Events:** a run enters at its first request for its own hidden link. Pages count distinct maze page addresses read. A give-up is an `answer()` call or a successful request for a page outside the maze. A lost exit is a request outside the maze that the server cannot serve. The 80-call limit censors the run at its depth.
- **Rule:** pilot hazard h3 = give-ups on pages 1 to 3 divided by the leave-or-go-on choices on those pages. Forecast median = ln 0.5 / ln(1 - h3), capped at the median reachable depth. Measured median = the Kaplan-Meier median, with lost and capped runs censored. An agent is testable with at least 1 pilot give-up, at least 10 give-ups, and give-ups at least half of all exits. It is tested when its Kaplan-Meier median exists and it meets the data floors (3 maze seed blocks of at least 10 entered runs, 2 dead-end blocks of at least 6 runs). The law fails on a decisive 2x miss: every hazard in the exact 95% interval of the pilot count also misses. It also fails when one of two shape tests gives p ≤ 0.025. One is a Pearson chi-square test over the page bins 1-3, 4-10, 11-30 and 31+ (one shared rate, df = bins with data minus 1). The other is a discrete-Weibull likelihood-ratio test of the shape against 1. Yes needs the law to hold in every tested status agent. One tested agent that fails gives No.
- **gpt-oss-20b, counted:** entered 54 of 60 maze runs (blocks 18, 17, 19). All 54 exits were give-ups (53 answers, 1 page outside the maze). None was lost, and none reached the limit. h3 = 1/162 = 0.006, forecast median 78 (the reachable cap), Kaplan-Meier median 12, ratio 6.5. Not decisive: the upper end of the exact interval, h = 0.034, forecasts 19.8 pages, within 2x of 12. Pearson chi-square 50.7 (df 2, p = 1 × 10⁻¹¹, give-ups 1, 18, 35 against 12.8, 27.4, 13.8 expected). Discrete-Weibull shape 3.07 (95% profile interval 2.48 to 3.72), likelihood-ratio p = 2 × 10⁻¹⁶. Hazard over all pages 0.079 (54 give-ups in 681 choices), run-level 95% interval 0.073 to 0.087 (bootstrap over runs, 2,000 draws). Seed-level: 0.080, t-interval 0.063 to 0.097 over the 3 maze seed blocks (block hazards 0.077, 0.088, 0.075). Fitted to each seed block alone, the Weibull shape is 2.71, 3.25 and 3.70. The ratio is defined in 1 of 3 blocks (2.85). The other two had no give-up on pages 1 to 3. Dead end: 19 of 20 runs entered, and all 19 left after its one page.
- **Qwen3-8B, counted:** 9 maze runs, 7 entered, 6 give-ups, 1 at the limit. It is below the data floors, so not tested. Its Kaplan-Meier median was 2 pages against a forecast of 2.4. 9 dead-end runs, 7 left after the one page.
- **Spend (reported row):** billed tokens from the entry call on (full prompt plus new tokens per call). The Kaplan-Meier median by the frozen definition (smallest value with survival at or below 50%) is 41,133 tokens. No run was cut off. `analysis.py` prints 46,415: rounding in its survival product lands just above 50% at the 27th run, so it takes the 28th. The plain median is 43,774.
- **Exit reasoning (no model judge, `research_iris/quit_cues.py`):** in the reasoning of the 53 answer calls, 17 name a part number ("Likely part 22 etc.") and 8 cite time ("But time low"). The prompt shows no clock and no step count. The exit prompt held 2,097 to 9,457 tokens, and no call reached the new-token cap (`exit_check.py`).
- **Answers:** of the 53 answer exits, 12 held a made-up figure, 9 the site's 2024 figure, 32 no figure and 0 the 2025 figure (`results/exit_check_gpt-oss-20b-low-full.json`). In the dead-end control, 1 of 14 answers held a made-up figure and 7 the 2024 figure, so the maze did not make answers more often wrong. The reported row in `analysis.py` gives 13 and 10. Its regex misfiles two runs: run 25 ("12-month") and run 51 ("9,000" matched the 2024 figure "900" as a substring).
- **Mixture check (after the data, not part of the rule):** a one-sided Fisher test on give-up against go-on choices (`research_iris/page_hazard.py`). For gpt-oss-20b, pages 1-3 against 4 and deeper: 1 of 162 against 53 of 519, p = 4 × 10⁻⁶ for a rise. Under simulated constant hazards and mixes at 54 runs, the rise test fired at most 4.7% of the time (`research_iris/rise_null_sim_n54.py`).
- **Prediction:** scored as written: chi-square p > 0.05 and the 2x forecast in at least one agent, and at least half of all maze runs at the 80-call limit. Both parts failed: gpt-oss-20b's chi-square p was 1 × 10⁻¹¹, and 1 of 61 entered maze runs reached the limit.
- **Power, simulated before the runs:** with 40 to 60 runs per agent, a true constant hazard is called "fails" at most 7% of the time. A hazard that rises with depth is caught only 23% to 42% of the time. The rise we saw is much steeper than the simulated ones.
- **Pilot runs (the first version of this post, not counted):** These used the cache shortcut. Over 9 runs, gpt-oss-20b had 0 give-ups in 27 chances on pages 1 to 3 and a median of 13 pages, close to the counted runs. Qwen3-8B left mostly on page 2 (28 of 35 runs that reached it).
- **Stand-ins on our Mac mini:** our rule named Qwen3-1.7B with thinking off and one second model of at most 2B parameters as a side row. None passed the 8-of-10 gate in time (DEVIATIONS D4). To check our code on real transcripts, we ran one closed model, gpt-5-nano, through the same harness and rule. It is never part of the answer. Its measured median was 6 pages against a forecast of 16, and both shape tests rejected a constant hazard.
- **Rebuilds:** Vera rebuilt every counted number from the raw records with her own code (`skeptics/skeptic/`). She also ran 1,000 data sets with a true constant hazard of 0.079 through the frozen rule. The rule called 7.4% of them FAILS, and none reached the observed p-values. A permutation test gave 0 of 400, and dropping any one run leaves the Pearson p at most 5.1 × 10⁻¹¹.
- **Cost:** all agent runs were local, on Esben's laptop. Tonight's model-API spend was $0.03 before the final cross-family review: cheap model checks of the text, title and figure.

</details>

[^cf]: [Cloudflare, "Trapping misbehaving bots in an AI Labyrinth"](https://blog.cloudflare.com/ai-labyrinth/) (March 19, 2025). Its summary line: AI Labyrinth uses generative AI to "slow down, confuse, and waste the resources of AI Crawlers and other bots". It gives no cost number.

[^mc]: Our [maze-cost run](https://blog.kran.ai/maze-cost/) of November 18, 2026: 0 of 84 maze runs by Gemma 4 31B and Gemma 4 26B-A4B left before the 12-call limit. Behind a single dead-end page, 80 of 80 runs left after reading the one page, 78 of them on the next step.

[^prior]: [Mantis](https://arxiv.org/abs/2410.20911) (Pasquini, Kornaropoulos and Ateniese, 2024) led agents into an endless decoy file server: "the tested LLM-agents continue navigating the tarpit indefinitely". Its server repeats an injected instruction at every step, and every run of its cost test stopped after 10 rounds. [AgentSnare](https://arxiv.org/abs/2607.26998) (Wang et al., July 2026) "absorbs 49.9% of tool calls to delay attacks" over 15 test web apps and 4 attacker models. [How Adversarial Environments Mislead Agentic AI?](https://arxiv.org/abs/2604.18874) (Zhan, Zhou, Li, Jing, Li and Haddadi, April 2026) built a decoy maze of fake papers: "depth attacks achieve up to 96% trap entry rates, wasting 49-73% of step budgets" over 11,000 runs on five agents. [WebMaze](https://arxiv.org/abs/2610.38524) (Zeng, Luo and Li, October 2026) found that "decoy pages account for an average of 93.0% of the distinct pages requested per run" over 12 web apps and three scanners. None of the four reports a chance of leaving per page.

[^budget]: Real agent frameworks allow more steps. Browser Use, a popular one, allows 100 by default: `run(self, max_steps: int = 100, ...)` in [`browser_use/agent/service.py`](https://github.com/browser-use/browser-use/blob/ccbdb8095fbaf903c748880a600054a34c7b6191/browser_use/agent/service.py#L443) on GitHub main, November 17, 2026.

[^ord]: Toby Ord, ["Is there a half-life for the success rates of AI agents?"](https://arxiv.org/abs/2505.05115) (May 2025), proposed "a constant rate of failing during each minute a human would take to do the task". Gus Hamilton's [reanalysis](https://gushamilton.github.io/lab/2026/01/23/petos-paradox-ai-agents/) (January 2026) found that "for all tested models (and humans), κ was below 1, suggesting that over time, hazards reduce". Ord [agreed](https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline) in February 2026: "AI agents probably don't have a constant hazard rate / half-life." All three study failure on long tasks, not leaving a maze. The closest use of these tools on agents that quit is Bogdanov, Manakina and Lung, [arXiv 2609.29508](https://arxiv.org/abs/2609.29508) (August 2026): Kaplan-Meier curves and per-step hazards for when agents claim a reward in a 20-step waiting game. It has no decoy and tests no constant hazard.

[^mix]: Proschan, ["Theoretical explanation of observed decreasing failure rate"](https://doi.org/10.1080/00401706.1963.10490105) (Technometrics, 1963): a mixture of units, each with its own constant failure rate, has a pooled failure rate that never rises. Mixtures of decreasing-failure-rate distributions keep a decreasing rate ([Barlow, Marshall and Proschan](https://doi.org/10.1214/aoms/1177704147), 1963).
