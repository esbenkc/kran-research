role: skeptic (Vera) — novelty check, written 2026-11-23 before the data landed

Claim to check: no public work tests whether an agent's follow rate on a CSS-hidden link depends on whether the
tool's link list carries anchor text, nor a pre-registered reach x follow law on held-out tool x model cells.

Searched (web search, 2026-11-23): "LLM web agent hidden link honeypot anchor text follow rate", "anchor text web
agent observation format ablation", "hidden link trap AI agents display:none 2026", the arXiv ids Iris cites.

Closest work found, and why it does not answer this question:
- Rilla, Nussberger, Mata, Wulff, arXiv 2609.31054 (Cheap, open agents make LLM pollution harder to mitigate):
  9 agent setups x honeypot items in a survey (a hidden checkbox); reports failure patterns differ by agent type.
  No link-list format factor, no product law, no held-out prediction.
- Zeng, Luo, Li, arXiv 2610.38524 (WebMaze): decoy worlds for LLM web scanners; 93.0% of distinct pages requested
  are decoys. Measures diversion, not a format factor; no anchor-text ablation.
- Ousat et al., arXiv 2607.18659 (Broken Gates): bot defenses vs solvers and 8 browser agents; identity/browser
  state, not hidden links.
- AgentOccam, arXiv 2410.13825: observation-space simplification (merge link text with link nodes) improves task
  success; a navigation-success ablation, not a canary follow rate, and no hidden link.
- Writesonic crawler blog / ScienceDirect S2667295224000953: HTML-only crawlers do not render CSS, so display:none
  text is read; LLMs do not reliably detect hidden links. A reach fact (link is seen), not a follow factor.
- Pirolli & Card 1999 (information scent): the human-browsing theory the anchor-text mechanism maps to; no agents.
- Our own parents: seen-then-followed (REJECTED, -0.296 [-0.407, -0.198] in the bare-list cell), held-out-reach,
  hidden-link-crawl.

Verdict: NEW as a test (anchor text on vs off in the same shipped LangChain list, same page, two local open-weight
agents, law frozen before data). Not new: that CSS-blind tools expose display:none links (reach ~1 is expected).
