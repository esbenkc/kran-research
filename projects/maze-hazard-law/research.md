# Research notes — defense-1118-maze-hazard-law (Iris)

Checked 2026-11-18 ~00:00-00:20 CET. Every line has its source. Parent notes: kran-research
`origin/swarm/defense-1118-maze-cost:projects/maze-cost/research.md` (Cloudflare, Mantis, Tracebit, AgentLSD,
Optional Traps, prices, framework step budgets; all re-usable, re-cited below where needed).

## 1. The constant-hazard law for agents: public history (closest prior work on the LAW)

1. **Toby Ord, "Is there a half-life for the success rates of AI agents?"**, arXiv 2505.05115 (submitted 2025-05-08),
   https://arxiv.org/abs/2505.05115 . Abstract: performance on longer tasks "can be explained by an extremely simple
   mathematical model — a constant rate of failing during each minute a human would take to do the task. This implies
   an exponentially declining success rate ... each agent could be characterised by its own half-life." p.? (PDF text):
   "Because the exponential distribution is the unique memoryless distribution, ... the chance of failing at the next
   moment is independent of how far you've come". Data: METR task suite (Kwa et al. 2025).
2. **Toby Ord, "Hazard Rates for AI Agents Decline as a Task Goes On"**, tobyord.com, 2026-02-04,
   https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline . Verbatim: "Contrary to my earlier hypothesis,
   AI agents probably don't have a constant hazard rate / half-life. Instead their hazard rates systematically decline
   as the task goes on." "the way that every single model has k statistically significantly below 1 convinces me".
   AI agents "mainly have a k of about 0.6, while the human value of k is significantly lower, at about 0.37".
3. **Fergus (Gus) Hamilton, "Peto's Paradox and the Future of AI Agents"**, 2026-01-23 (update 2026-02-03),
   https://gushamilton.github.io/lab/2026/01/23/petos-paradox-ai-agents/ . Weibull refit of METR data: "for all tested
   models (and humans), κ was below 1"; "humans exhibit ... κ=0.37, while SOTA models cluster between 0.6 and 0.9";
   Weibull vs logistic fit "about the same. In 7 LLMs, Weibull was favoured, in 7, logistic was favoured."
   "we should be sceptical of any model that makes the assumption that the median is like the tails."

**Use for us.** Our question is the defender-side twin: a constant per-PAGE hazard of LEAVING a decoy maze (not of
failing a task). The field's current prior (Ord 2026, Hamilton) is a FALLING hazard (Weibull k < 1). A falling hazard
means a 3-page pilot over-estimates the leave rate, so a geometric fit predicts a median that is too SHORT. The rival
LLM-side intuition (an agent notices repetition) predicts a RISING hazard. Freeze a discrete-Weibull alternative
(survival S(d) = q^(d^k); k = 1 is the geometric) as the shape test; name it "Weibull shape" in the post (field term).
Novelty: I found no work that fits a survival law to LLM-agent give-up in a tarpit/decoy maze (searches below).

## 2. Decoy mazes / tarpits (prior work on the SETTING; from parent notes, re-checked links)

- Cloudflare AI Labyrinth, blog 2025-03-19, https://blog.cloudflare.com/ai-labyrinth/ : decoy pages to "slow down,
  confuse, and waste the resources of AI Crawlers and other bots"; "No real human would go four links deep into a maze
  of AI-generated nonsense." No cost number.
- Mantis, Pasquini, Kornaropoulos, Ateniese, arXiv 2410.20911: agent-tarpit, "The attack is halted once the agent
  performs 10 iterations within the tarpit" (s.VIII-C): forced cap, no give-up rate.
- Tracebit, "Making attacking agents pay with Context Bombs" (Oct 2026), https://agentic.tracebit.com/factory/ :
  "1.6–3.9× model cost", "2.8–8.9× runtime"; aggregate multipliers, no per-page rate.
- Our parent: maze-cost (kran-research origin/swarm/defense-1118-maze-cost, results/analysis.json): both Gemma 4 agents
  L3 = 0 leaves in E3 = 126 pilot decisions; leaves_all = 0; leave_rate_upper95 = 0.0071 (26B-A4B) and 0.0072 (31B);
  surv_cap = 1.0; outcome "never quits: spend set by the cap" at a 12-call cap.

## 3. Step budgets and how frameworks end a run

- Browser Use `run(self, max_steps: int = 100, timeout: float = 1700, ...)` (browser_use/agent/service.py l.443,
  commit ccbdb8095fbaf903c748880a600054a34c7b6191, 2026-11-17). loop.py l.431-432: at `run_step >= max_steps` it sets
  `budget_hit = 'max_steps'` and gives ONE delivery-only turn; loop.py l.4 header: screenshots "with only the last two
  kept, compaction". I found no per-step "step k of N" counter in the prompt (grep for step_info / "of {max" empty).
  -> our harness should NOT show a step counter either (parent agent.py has none); freeze that.
- Parent framework list (OpenAI Agents SDK 10 turns, smolagents 20 steps, LangGraph 10007): parent research.md s."Default
  step budgets".

## 4. Agents (model cards, read 2026-11-18)

| model | params | context | source |
|---|---|---|---|
| Qwen3-8B | 8.2B | 32,768 native, 131,072 with YaRN; config max_position_embeddings 40,960 | https://huggingface.co/Qwen/Qwen3-8B README l.30-34, l.317 |
| gpt-oss-20b | 21B total, 3.6B active | config max_position_embeddings 131,072 (YaRN factor 32 on 4,096) | https://huggingface.co/openai/gpt-oss-20b README l.26, config.json |
| Qwen3-1.7B | 1.7B | 32,768 | https://huggingface.co/Qwen/Qwen3-1.7B README l.30-34 |
| Qwen3-0.6B | 0.6B | 32,768 | https://huggingface.co/Qwen/Qwen3-0.6B README l.30-34 |

Context risk: parent Gemma page spend grew ~378 tokens per page (analysis.json runs.maze_neutral[0].T: 1033, 1413,
1787, 2164, 2542). At ~400 tokens per maze step, 80 calls of full history is ~32k tokens before reasoning, so
Qwen3-8B and the Qwen3 small models hit 32k-40k near the cap: overflow is censoring correlated with depth.

## 5. Statistics the frozen rule needs (standard references)

- Censored geometric MLE: h = leaves / decisions at risk (discrete-time survival; same as parent PREREG s.3-4).
- Zero leaves: the one-sided 95% upper bound is 1 - 0.05^(1/n) ~ 3/n ("rule of three"); a point estimate of 0 gives
  no finite median, so a zero-event pilot cannot be scored as a pass.
- Kaplan-Meier median: undefined when survival never reaches 0.5 before the cap.

## 6. Searches (2026-11-18/19)

- "LLM agent tarpit give up survival analysis hazard decoy pages" (web search, 00:10 CET): hits were Mantis 2410.20911
  (v2 PDF p.8: "once the LLM-agent is inside the tarpit, no attempts to backtrack or abandon the current path were
  observed"; "the tested LLM-agents continue navigating the tarpit indefinitely"; note the injection is re-sent each round),
  Nepenthes (github.com/NEPENTHESWEB/nepenthes-py, "Infinite Maze ... No exit links"), a delay-of-gratification survival
  benchmark (arXiv 2609.29509, Kaplan-Meier over agent trajectories, not a decoy) and AgentLSD 2609.19140. None fits a
  give-up law (constant vs changing hazard) to agents in a decoy maze.
- Budget awareness (relevant to our 'no step counter' freeze): Liu, Wang, Miao, Hsu, Yan, Chen, Han, Xu, Chen, Jiang,
  Daruki, Liang, Wang, Pfister, Lee, "Budget-Aware Tool Use Enables Effective Agent Scaling", arXiv 2511.17006
  (2025-11-21; abs page meta, all 15 authors printed): agents given a bigger tool-call budget "lack budget awareness";
  their Budget Tracker shows the remaining budget. Our harness shows no budget, as Browser Use does not; a budget
  display would be a different agent (and a burst at budget fractions would be expected).

## 7. Decisions frozen in PREREG (f128b078c, push 2026-11-18T23:08:27Z)

- Harness facts (web/agent.py): only answer() and a GET outside /m/ end a run; parse failures, never-issued /m/ ids,
  off-site URLs and unknown tools cost a call and the run goes on. So give-up = answer() or 200 GET outside /m/;
  lost = non-200 GET outside /m/ (Iris #25750).
- Chi-square df: the shared-rate 2 x B homogeneity test has df = B' - 1; research_iris/df_check.py (4000 reps, true
  geometric): P(p > .05) 0.946-0.958 with df = B'-1, 0.891-0.908 with df = B'-2.
- Seed blocks: P(>= 2 of 3 blocks define R_b) = 0.57 (h 0.02) / 0.24 (h 0.01) at ~13 runs per block; R_b is reported only.
- oracle.py prereg reads GitHub's activity API with activity_type=push; the branch's first push is a
  'branch_creation' and is not found. An empty descendant commit pushed at 23:08:27Z gave the push event.
- Cloudflare page re-read 2026-11-19 00:15 CET: "waste the resources of AI Crawlers and other bots" (description) and
  "No real human would go four links deep into a maze of AI-generated nonsense." still live; published_time
  2025-03-19T13:00:00.000Z.

## 8. Added after the freeze (credits; checked 2026-11-19 ~00:20 CET)

- **AgentSnare** — Wang, Zhao, Wu, Zheng, Zhao, Lin, Ren, "AgentSnare: Learning to Delay, Divert, and Defuse
  Autonomous Web Penetration Agents", arXiv 2607.26998 (2026-07-29; abs page meta, all 7 authors printed). Abstract:
  "Across 15 CVE-Bench web applications and four attacker models, AgentSnare absorbs 49.9% of tool calls to delay
  attacks, retains 57.2% of post-entry actions in the decoy to divert trajectories". Aggregate shares; no per-step
  hazard and no pilot-to-full prediction. Closest decoy-cost measurement on LLM agents (found by Vera, novelty.md).
- **ARTEX-ymh tarpit breaker** — huntback Research, "ARTEX learns to spot honeypots: dissecting an AI pentest agent's
  decoy detector" (datePublished 2026-11-04), https://huntback.io/blog/artex-fork-learns-to-spot-honeypots :
  "guard/tarpit.go counts page fetches per host per task, in memory. After 40 (configurable as tarpit_fetch_budget) it
  fires once: a 'suspected tarpit maze' hint goes into the planner's view". A real attacker agent with a give-up at a
  fixed depth: a burst by construction (our rule catches a page-40 burst 0.09-0.18 at n <= 60). Next-posts material.
