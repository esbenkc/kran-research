# Novelty check — Vera (skeptic), defense-1119-maze-hazard-law, 2026-11-18 PST

Claim checked: no published work tests whether LLM agents leave a decoy maze at a constant per-page rate, or whether a
3-page pilot forecasts the median pages per run.

Searched (web search, 2026-11-18): "LLM agent tarpit decoy maze survival analysis hazard rate pages before giving up";
"arXiv 2609.29508 Kaplan-Meier LLM agents quitting".

Closest work and what it does NOT do:
- Zhan, Zhou, Li, Jing, Li, Haddadi, "How Adversarial Environments Mislead Agentic AI?" arXiv 2604.18874 (2026-04-20).
  POTEMKIN harness; "The Maze" depth attacks trap agents in structural loops; 11,000+ runs on five frontier agents;
  reports trap entry (up to 96%) and step-budget waste (49-73%). No per-page leave rate, no survival shape, no
  pilot forecast. NOT yet credited in research.md at b9b8b5741: the closest decoy-maze prior work, should be credited.
- Bogdanov, Manakina, Lung, arXiv 2609.29508 (2026-08-26): Kaplan-Meier + discrete-time hazard for agents quitting a
  delay-of-gratification game; no decoy, no constant-hazard test, no pilot forecast (already credited by Iris).
- Pasquini, Kornaropoulos, Ateniese, arXiv 2410.20911 (Mantis): tarpit halted at 10 rounds; no hazard.
- Ord arXiv 2505.05115; Hamilton 2026-01-23; Ord 2026-02-04: hazard shape for task FAILURE (METR), not give-up from a
  decoy. Credited.
- huntback ARTEX blog (2026-11-04): an attacker-side tarpit breaker fires after 40 fetches per host; a design
  parameter, not a measured hazard. Credited.

Verdict: the question (constant per-page give-up hazard in a decoy maze, pilot-forecast within 2x) is new as far as
these searches reach. The reuse-on preview result (a page-2 cliff in Qwen3-8B; a rising hazard in the gpt-oss-20b
9-run pilot and in gpt-5-nano) does not duplicate a published number.
