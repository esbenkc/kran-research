# Research notes — defense-1118-maze-cost (Iris)

Checked 2026-11-18 ~21:00 CET. Every line has its source.

## Prior work

1. **Cloudflare AI Labyrinth** — blog "Trapping misbehaving bots in an AI Labyrinth", published
   2025-03-19T13:00Z (page meta `article:published_time`), https://blog.cloudflare.com/ai-labyrinth/ .
   Claims (verbatim): AI Labyrinth "uses AI-generated content to slow down, confuse, and waste the resources of AI
   Crawlers and other bots"; "the crawler wastes time and resources"; "No real human would go four links deep into a
   maze of AI-generated nonsense." Target = crawlers/scrapers, not LLM agents. No cost number is published there.
   -> "four links deep" is Cloudflare's own detection depth; a natural anchor for our depth readout.
2. **Mantis** — Pasquini, Kornaropoulos, Ateniese, "Hacking Back the AI-Hacker: Prompt Injection as a Defense Against
   LLM-driven Cyberattacks", arXiv 2410.20911 (v2 18 Nov 2024), https://arxiv.org/abs/2410.20911 ;
   code https://github.com/pasquini-dario/project_mantis .
   - s.VI-C "agent-tarpit": fake FTP filesystem of infinite depth; the injection is REPEATED at every directory
     listing "reinforcing its focus on the injected task".
   - s.VIII-C Fig. 11: dollar cost of GPT-4o API calls for one attack vs directories per node; "The attack is halted
     once the agent performs 10 iterations within the tarpit." -> they cap at 10 rounds; no natural give-up rate,
     no defender-side cost, no ratio.
   - They note the tarpit "can sometimes cause the LLM-agent to escape the agent-tarpit and move on" (FTP get + quit).
   - Cost lever they use: prompt size per round (more directories per listing), agents with history re-insertion
     (PentestGPTAuto) cost more than summarising agents (AutoPenAgent).
   **Novelty gap for us:** no prompt injection (a plain task-fitting link), no forced cap as the stopping rule,
   open-weight agents, defender cost measured on the same run, and the per-page leave probability.
3. Our own: hidden-link-crawl, hiding-vs-stacked, trap-aware-canary (+ open-weight). blog.kran.ai/_notes/*.md.
   Parent harness: kran-research origin/swarm/defense-1118-trap-aware-canary,
   projects/trap-aware-canary-open-weight/web/{agent.py,mlxcall.py}.
4. **Tracebit Research, "Making attacking agents pay with Context Bombs"** (Oct 2026),
   https://agentic.tracebit.com/factory/ (opened 2026-11-18): "We tested nine Context Bomb techniques against seven
   models in 773 AWS assessments"; with deception "1.6–3.9× model cost", "2.8–8.9× runtime"; "Inside one run 825 pages
   ... GLM recognized that the decoy API was effectively endless"; "Our deception is deterministic. We don't need to
   spend tokens every time an attacker does." -> aggregate multipliers, no defender-side number, no per-page rate.
   (found by Vera, skeptic/novelty.md; quotes re-read by Iris)
5. **AgentLSD** — Golinelli, Drago, Boffa, Bergadano, Crispo (Golinelli et al.), "AgentLSD: Evaluating AI Security Agents Under Adversarial Task Contamination",
   arXiv 2609.19140 (2026-09-16), https://arxiv.org/abs/2609.19140 (title/date/authors from the abs page meta).
6. **Optional Traps** — Ghifiardi, "Optional Traps: An Empirical Study of LLM Agents Against Credential Deception and
   Alert-Driven Revocation", arXiv 2611.19295 (2026-11-14), https://arxiv.org/abs/2611.19295 . Abstract (re-read by
   Iris): "Deception added detection and per-client revocation in 19 of 20 defended sessions, but no measurable
   prevention beyond hardening"; "traps must sit on the attacker's path rather than beside it ... deception's
   measurable value is fast, attributable detection. All results are lab-scale". 4B and 9.7B local models.
   AgentLSD abstract (re-read): 6 models, 11 web CTFs; "traps increase the number of turns (+20) and reasoning tokens
   (+2k)" even when the flag is recovered.

## Prices (for a frozen price table; fetched 2026-11-18)

| item | price | source |
|---|---|---|
| gpt-oss-20b input / output | $0.018 / $0.09 per M tokens (cache read $0.009) | https://openrouter.ai/api/v1/models (id openai/gpt-oss-20b) |
| Qwen3-8B input / output | $0.20 / $0.20 per M tokens (cached $0.10); "cheapest input price has risen 900%, from $0.020 to $0.200" in ~90 days | https://pricepertoken.com/pricing-page/model/qwen-qwen3-8b |
| Cloudflare Workers Standard CPU | $0.02 per additional million CPU ms | https://developers.cloudflare.com/workers/platform/pricing/ |
| Cloudflare Workers Standard requests | $0.30 per additional million requests | same page |

Open-weight token prices move 10x within a quarter (Qwen3-8B line above). A ratio threshold of 100 is therefore
price-sensitive; the price table must be frozen in PREREG, and a price-free readout (pages per run, tokens per
defender byte, compute seconds vs compute seconds) should sit next to it.

## Back-of-envelope (before any data; for phase 0 only)

- One gpt-oss-20b step, ~2,000 input + ~300 output tokens: 2000*1.8e-8 + 300*9e-8 = ~6.3e-5 $.
- One static template page, ~0.2 ms CPU + ~5 kB: CPU 4e-9 $; bytes at $0.09/GB = 4.5e-7 $; request fee 3e-7 $.
- Per-page ratio ~ 1e2 (bytes or request fee priced) to ~1e4 (CPU only). The dead-end arm (1 page) would then sit
  near 1e2 as well, so "dead end under 10" is very unlikely under any listed pricing; the maze/dead-end contrast and
  the depth carry the information.
- Agent input grows with history (each step re-reads earlier pages) -> agent cost ~ depth^2, defender ~ depth;
  the ratio grows ~linearly with depth (Mantis Fig. 11 discussion: history re-insertion raises cost).

## Timing (laptop queue)

- gpt-oss-20b, MLX on the laptop: 19.2 s for a 4-step run (job.log gpt-oss-20b-d4, 2026-11-17T09:29:28Z) ~5 s/step
  at short context.

## Agents (model cards, read 2026-11-18)

- gemma-4-31B-it: dense, 30.7B total parameters, 256K context, Apache 2.0 license
  (https://huggingface.co/google/gemma-4-31B-it README "Dense Models" table). On HF since 2026-03-11.
- gemma-4-26B-A4B-it: mixture of experts, 25.2B total, 3.8B active parameters, 256K context, Apache 2.0
  (https://huggingface.co/google/gemma-4-26B-A4B-it README "Mixture-of-Experts (MoE) Model" table).
- Served through the Gemini API on our key (free tier), priced in the post at OpenRouter list prices (PREREG s.4).
- Gate (DEVIATIONS v1.1): 10/10 and 10/10 correct on the one-visible-hop task.

## Default step budgets of common agent frameworks (read 2026-11-18, main branch, pinned commits)

In a "never quits" world the agent's own step budget, not the maze, ends the run. Defaults:
- OpenAI Agents SDK: `DEFAULT_MAX_TURNS = 10` (src/agents/run_config.py l.45,
  https://github.com/openai/openai-agents-python/blob/8fd9a7cd19a4c495c438d39c83fb03ea3fb5a7e0/src/agents/run_config.py).
- smolagents: `max_steps: int = 20` (src/smolagents/agents.py l.300,
  https://github.com/huggingface/smolagents/blob/fd871374573ddd7cafa06309452c72b90513cf25/src/smolagents/agents.py).
- Browser Use: `run(self, max_steps: int = 100, ...)` (browser_use/agent/service.py l.443,
  https://github.com/browser-use/browser-use/blob/c7fd4e51602c91f01269c3ca864c0e9b60e6ffec/browser_use/agent/service.py).
- LangGraph: `DEFAULT_RECURSION_LIMIT` = env `LANGGRAPH_DEFAULT_RECURSION_LIMIT` or 10007 supersteps
  (libs/langgraph/langgraph/_internal/_config.py l.42-43,
  https://github.com/langchain-ai/langgraph/blob/27aeb75efda7b52be4302ef46016cee3409cee6b/libs/langgraph/langgraph/_internal/_config.py).
  (Older LangGraph versions used 25; current main is effectively unbounded.)
Our cap of 12 calls sits between the Agents SDK (10) and smolagents (20).
