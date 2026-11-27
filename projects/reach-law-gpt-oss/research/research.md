# research.md — reach-law-gpt-oss (Iris, 2026-11-27)

Every line has a source. "Recount" = our own code on saved raw data, path given. Checked 2026-11-27 07:55-08:10 CET.

## 1. External sources (verified, with dates)

| # | Source | Date | What it says (verbatim where quoted) | How we use it |
|---|---|---|---|---|
| E1 | Ghifiardi, *Optional Traps: An Empirical Study of LLM Agents Against Credential Deception and Alert-Driven Revocation*, arXiv [2611.19295](https://arxiv.org/abs/2611.19295) (cs.CR) | 2026-11-14 | Abstract: "Against LLM agents, traps must sit on the attacker's path rather than beside it." Sec. VII Study 4, "False-positive mode (B)": "The legitimate agent held the correct key in its prompt, yet fetched the debug route, preferred the 'leaked' credential, used it and was revoked. For deception aimed at LLM clients, bait that looks useful can also be taken by benign automation." Table IV: engagement B = qwen3.5:4b-q8_0, one engagement, legit agent "Took the canary bait and was revoked"; engagement D (tantular-cyber) legit agent "Saw the bait, did not use it; not revoked". Sec. VIII: "LLM clients change the false-positive calculus." Study 2: 4B model, export 20/20 baseline, 0/20 hardened, 0/20 hardened + deception. | The counterexample to a pure reach rule: n = 1, correct key in the PROMPT. Our PROMPT arm copies that setup. The goal's paraphrase "traps beside the path are optional" is NOT a quote: cite the abstract sentence above. |
| E2 | Rassul & Rashid, *AgentShield: Deception-based Compromise Detection for Tool-using LLM Agents*, arXiv [2605.11026](https://arxiv.org/abs/2605.11026) | 2026-05-10 | HTML v1, Sec. 3: "Honeytokens are structurally safe when they appear only in regions a legitimate agent has no reason to read." Fake tools, fake credentials and allowlisted parameters inside the agent's tool interface; AgentDojo; zero false positives in 485 trials on GPT-4o-mini for the parameter-validator layer (Wilson 95% [0%, 0.79%]). | The published "reach" design rule for honeytokens: safety by placement where benign agents do not read. Our decoy sits in a region every agent reads (the denial), so our test is the hard case AgentShield excludes. |
| E3 | Gans, *When Agents Talk: Honeytokens under Shared Memory*, arXiv [2608.11436](https://arxiv.org/abs/2608.11436) | 2026-08-11 | Abstract: "can a honeytoken be harmless to trusted agents without becoming recognisable to an attacker who shares their information and can implement the trusted policy? The answer is no." | Theory bound: a trusted rule that avoids decoys can be copied by an attacker. Relevant to "Next posts", not to this run's claim. |
| E4 | Roh, Ryu, Park, Choi, *A Gingerbread House for LLM Agents: Source-Planned Honeypots...*, arXiv [2611.07611](https://arxiv.org/abs/2611.07611) | 2026-11-05 | Lures (decoy pages, fake endpoints, hints, honeytokens) planned from source; on 40 CVE-Bench tasks the static plan cut real compromise from 26/40 to 5/40 for Claude Opus 4.8. | Related work on lures for attacking agents (no benign-agent false-positive measure). |
| E5 | Wang, Yu, Sun, *AgentTrap: Stateful Feedback Deception against Autonomous Penetration Testing Agents*, arXiv [2610.02869](https://arxiv.org/abs/2610.02869) | 2026-10-02 | "uses sentinel endpoints to avoid benign interference". | Related: benign interference handled by placement, not measured with benign LLM agents. |
| E6 | Reworr & Volkov, *LLM Agent Honeypot: Monitoring AI Hacking Agents in the Wild*, arXiv [2410.13919](https://arxiv.org/abs/2410.13919) | 2024-10-17 | SSH honeypot with prompt injection and timing; 8,130,731 attempts, 8 potential AI agents. | Background: honeypots that detect LLM agents. |

Field terms (for the title and the cold reader): **honeytoken** (E2, E4; fake credential whose use flags the user), **canary credential / canary token** (E1), **decoy** (E1). "Decoy key", "route", "reach" are Lab words; the post defines them once and maps them to these terms.

## 2. Our prior results (published posts, front matter checked)

| Post | Night | Status | Key numbers (outcome_note) |
|---|---|---|---|
| route-first-law | 2026-11-26 | REJECTED | Route below the decoy: 6/60 web, 0/60 shell; route above: 15/60 web, 20/60 shell. Per model (results/analysis.txt): shell before gemma-4-31b-it 20/30, gemma-4-26b-a4b-it 0/30; web before gemma-4-26b 15/30, 31b 0/30. Secondary: runs that tried the route first took the trap 0/99 (web), 0/100 (shell). |
| discoverable-route | 2026-11-03 | CONFIRMED | First 30 runs per agent: runbook 17%, named 23%, none 96%. "only 17 of 150 opened the storage note." |
| anchor-text-law | 2026-11-23 | UNCONFIRMED | gpt-oss-20b requested the hidden link 60/60 with link words and 60/60 with markdown (web harness). Gate0 15/15 (web/out/A/gpt-oss-20b/gate_decision.json). |
| open-route | 2026-11-02 | UNCONFIRMED | 4-call route named: key use 12% vs 97% with no route; key before any route try in 17/18 key runs. |
| held-out-reach | 2026-11-13 | REJECTED | A different "reach": crawl reach of hidden links by fetch tool; rule missed by 49 points on average. Do not call this run "the same reach law" (Kasper #31928 (6)). |

**"64/64" (goal text).** Source: route-first-law PREREG.md:51, "gpt-oss-20b took the trap in 64/64 runs that read the 2024 menu page first" (Kasper's recount of anchor-text-law raw). It is a subset count, and that menu page held stale 2024 numbers, so it shows gpt-oss skipping a wrong-looking route. Never write it bare (Kasper #31939). Overall: 60/60.

## 3. Recount of saved data by logged reach (pre-answers the pooled rule for API agents)

Code: `swarm-defense-1127-reach-law-gpt-oss/iris/recount_discoverable.py` (output `recount_discoverable.txt`). Data: `git archive origin/swarm/defense-1103-discoverable-route projects/discoverable-route/raw`. Reached = first 60 characters of `episode.route_text` in any observation up to the first action whose args hold the key (whole run if none).

| Cell | Reached: key use | Never reached: key use |
|---|---|---|
| none | 0/0 | 144/150 |
| runbook | 35/199 | 0/1 |
| linked (note) | 1/17 | 132/133 |
| named (in denial) | 49/199 | 0/1 |
| **pooled none+runbook+linked** | **36/216 = 16.7%** | **276/284 = 97.2%** |

Per agent (none+runbook+linked): claude-haiku-4-5 0/41 & 58/59; gemini-3.5-flash-lite 0/48 & 51/52; gpt-5.4-nano 1/47 & 50/53; **gemini-3.1-flash-lite 12/40 (30%)** & 60/60; **gpt-5.4-mini 23/40 (58%)** & 57/60.
Independent rebuilds: Vera (`skeptic/vera_prior_reach.py`, exact match), Felix (incl. named: reached 85/415 = 20%, never 276/285 = 97%; mini 39/79), Kasper (mini runbook 23/39, all 23 with no route call).

Note opening (read `/runbooks/storage.md`), linked arm: nano 7/30, gemini-3.5-flash-lite 8/30, haiku 1/30, mini 1/30, gemini-3.1-flash-lite 0/30; 17/150 = 11%.

**What this means for the run.**
1. Pooled over API agents the rule already fits; per agent it breaks (mini 58%, flash-3.1 30%, Gemma-31B shell route-above 20/30 = 67%). The status answers the question for gpt-oss-20b alone.
2. Pooled bins nearly equal arms (NONE = never reached, RUNBOOK = reached), so the RUNBOOK reached cell (C3) and the LINKED within-arm split (C4) carry the reach-vs-arm test. LINKED reach is chosen by the agent (selection), and at an 11% open rate a 45-run arm gives about 5 reached runs.
3. Reached-but-took-key in the parent is a cost choice as much as a reach failure: route = 4 calls, key = 1 call (`shell/env.py` ROUTE_C = 4).

## 4. Novelty

- No paper we found measures decoy-credential use by *benign* LLM agents as a function of whether a correct route was in the agent's logged context. E1 reports one benign case (n = 1); E2 and E5 avoid benign contact by placement and do not test benign agents that do read the decoy region.
- The new bit of this run is gpt-oss-20b (a third model family, open weights, run locally) in the shell decoy-key harness, plus the PROMPT arm that reproduces E1's Study 4 setup at n = 45.
- Searches run: arXiv API and web search "LLM agents honeytokens decoy credentials benign agent false positive" (2026-11-27). Vera owns the formal novelty check (`skeptic/novelty.md`).

## 5. Constraints checked

- gpt-oss-20b only on Esben's laptop queue: model-keys has OpenAI/Gemini/Anthropic only (Bruno, Felix, Kasper); MLX peak about 13 GB (anchor-text-law). `labq list` 07:58 CET: worker esben-laptop mode off (seen 06:52Z); one critic job queued. Jobs since 11-26 12:06Z died with exit 127 (missing gw.sh; Tove #31936 ask-esben).
- Experiments API spend: $0 (local model).

## 6. Novelty refresh (08:50 CET, while labq 141r3us66d waits)

Web search "gpt-oss-20b agent leaked credential canary token benign agent uses credential found in error message": no study of gpt-oss-20b (or any open-weight agent) with decoy credentials in an error message and a logged allowed route. Hits were vendor blog posts on canary tokens for prompt leaks and repository PRs, not measurements. Section 4 stands.

E7 (added 09:20 CET, from Vera's novelty.md, abstract checked by Iris): Wehner & Kraprayoon, *Honeypots for AI Agent Insider Threats*, arXiv [2610.26664](https://arxiv.org/abs/2610.26664), 2026-10-19. Proposes planting "decoy opportunities for harmful actions throughout the production environment to detect misbehavior" of internal agents (an AI-control method). Our run measures the false-alarm side of that proposal for one decoy type: does a benign agent with an allowed route in view still take the decoy.
