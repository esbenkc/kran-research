# Novelty check (Vera, skeptic), 2026-11-18

Claim to check: a canary's catch rate factorises into a harness-only reach (logged: the link's id entered a tool
result) and a model-only follow-given-reach, tested by predicting held-out harness x model cells.

Searched (web, 2026-11-18): hidden-link / CSS-hidden honeypots for LLM agents, canary follow rates, agent harness
vs model decomposition, observation format (HTML / accessibility tree / markdown) x model effects on injection
success.

Closest prior work found:
- Rilla, Nussberger, Mata, Wulff, "Cheap, open agents make LLM pollution harder to mitigate", arXiv 2609.31054
  (Sep 2026): 9 agent configurations, detection checks on surveys; "open and commercial agents failed different
  sets of checks". Model and framework vary together, and exposure is not logged, so it cannot split reach from
  follow. Verified the abstract on arxiv.org/abs/2609.31054.
- RedTeamCUA (arXiv 2505.21936): adding an accessibility tree to screenshots changes injection success; that is an
  observation-format effect on one factor, not a held-out factorisation test.
- "Manipulating LLM Web Agents with Indirect Prompt Injection via HTML Accessibility Tree" (arXiv 2507.14799):
  attack success through one channel; no harness x model split.
- Practitioner pages (Bright Data glossary, ScrapingBee, Scrapfly) describe CSS-hidden link traps for scrapers;
  no agent measurements.
- Our own Lab runs (hidden-link-crawl, held-out-reach-law, trap-aware-canary) are the direct parents.

Verdict: I found no published test that logs exposure per run and predicts held-out harness x model cells with a
two-factor product. The result is new as a test design; the parent receipts already forecast a NO for the
API fallback pair (vera_retro.py), which the PREREG discloses.
