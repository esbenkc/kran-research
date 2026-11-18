# Novelty check (Vera, skeptic) — defense-1118-maze-cost

Checked 2026-11-18 ~21:00 CET by web search + reading the abstracts/pages below.

## Claim in the goal
"Cloudflare AI Labyrinth says that generated decoy pages waste the resources of unauthorized bots.
No one has measured that on LLM agents."

## Verdict: the "no one has measured" sentence is FALSE as worded. Narrow it.

Prior measurements of LLM-agent cost under decoys/tarpits:

| Source | What it measured | What it did NOT measure |
|---|---|---|
| Pasquini, Kornaropoulos, Ateniese, *Hacking Back the AI-Hacker (Mantis)*, arXiv 2410.20911 | GPT-4o agent $ cost inside an infinite-depth FTP tarpit | runs halted at 10 tarpit rounds; a prompt injection re-sent at each listing, so no natural give-up; no defender-side cost |
| Tracebit Research, *Making attacking agents pay with Context Bombs*, Oct 2026, https://agentic.tracebit.com/factory/ | 773 AWS assessments, 7 models: mean model cost 1.6-3.9x and runtime 2.8-8.9x with decoys; $1.33 -> $31.10 per confirmed objective; one GLM run crawled 825 pages of an "effectively endless" decoy API | pentest agents in AWS, not web agents; costs per run vs baseline, not per decoy page; no per-page leave rate; no defender cost; closed + open models mixed |
| Golinelli et al., *AgentLSD*, arXiv 2609.19140 (2026-09-16) | 6 models, 11 web CTFs: decoy endpoints/traps add ~+20 turns and ~+2k reasoning tokens | no infinite maze, no give-up law, no defender cost |
| Ghifiardi, *Optional Traps*, arXiv 2611.19295 (2026-11-14) | 4B/9.7B local agents vs canary credentials: deception gave detection, no prevention beyond hardening | no cost measurement of a maze; claims deception's measurable value is detection — the "detect only" side of our question |
| Cloudflare, *Trapping misbehaving bots in an AI Labyrinth*, 2025-03-19, https://blog.cloudflare.com/ai-labyrinth | vendor claim: decoy pages waste crawler time and resources; "no real human would go four links deep" | no measurement on LLM agents published |

## What stays new (if the run delivers it)
1. A per-page leave rate (hazard) of stuck open-weight LLM web agents in a hidden-link decoy maze,
   with no prompt injection and no forced halt.
2. A test of whether a 3-page pilot predicts the median maze spend (the geometric law).
3. A defender-normalised ratio (agent $ per defender $ per page) with a frozen price sheet.

## Required wording for the post
- Do not write "no one has measured this on LLM agents". Write that decoy cost has been measured
  for pentest agents (Tracebit, AgentLSD, Mantis), but not the per-page give-up rate in a web
  decoy maze, and not against the defender's own cost.
- Credit Tracebit's 1.6-3.9x and Mantis by name; cite Optional Traps for the "detection only" view.
