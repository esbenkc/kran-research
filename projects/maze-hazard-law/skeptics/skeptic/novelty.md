# Vera (skeptic) novelty check — night 2026-11-20 (room defense-1120-maze-hazard-law)

Claim to check: in a decoy maze with an 80-call budget and full history, gpt-oss-20b's chance of leaving per page
RISES with depth (discrete Weibull shape k 3.07, 95% profile interval about [2.5, 3.7]), so a 3-page pilot (1 give-up
in 162 decisions) forecasts the cap (78 pages) while the measured Kaplan-Meier median is 12 pages.

Searched 2026-11-19 (PST): web search "LLM agent give-up hazard rate increases with depth decoy maze tarpit 2026"
(last month), arXiv abstract pages opened for each candidate below. Last night's note (`2026-11-19/novelty.md` in
the receipts, formerly `novelty.md`) covers the work before 2026-11-18.

| work | what it measures | per-page leave rate? | shape? | pilot forecast? |
|---|---|---|---|---|
| Ord, arXiv 2505.05115 (2025-05) | constant hazard / half-life for METR task failure | task failure, not leaving a decoy | assumes constant | no |
| Hamilton, gushamilton.github.io (2026-01-23) | Weibull refit of METR data | task failure | k < 1 (falling) for every model | no |
| Zhan et al. "How Adversarial Environments Mislead Agentic AI?", arXiv 2604.18874 (ACL 2026 Findings) | entry rate and budget waste in injected depth traps | no | no | no |
| Pasquini et al. "Mantis", arXiv 2410.20911 | tarpit; runs halted at 10 rounds | no | no | no |
| Zeng, Luo, Li "WebMaze", arXiv 2610.38524 (2026-10-28; abstract opened) | decoy share of pages for 6 LLM scanners | no (aggregate share) | no | no |
| "A Gingerbread House for LLM Agents", arXiv 2611.07611 (2026-11-05; abstract opened) | routing confirmed attackers to a shadow copy; compromise counts on CVE-Bench | no | no | no |
| Bogdanov, Manakina, Lung, arXiv 2609.29508 (2026-08-26; abstract opened) | Kaplan-Meier curves and discrete-time hazard regression for when agents claim a reward in a 20-step waiting game | per-step hazard of claiming, not leaving a decoy | factor effects, no constant-hazard test | no |
| Cloudflare AI Labyrinth blog (2025-03-19) | claims decoys waste bot resources | no number | no | no |

Verdict: I found no work that reports a per-page leave rate for an LLM agent in a decoy, its shape, or a pilot-based
forecast. The sign is the opposite of Hamilton's k < 1 for task failure; the post may say "opposite of the falling
hazard found for task failure", not "first rising hazard ever" (the search covers the decoy/tarpit and agent-hazard
literature, not all of survival analysis of LLMs).
