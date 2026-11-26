# research.md — defense-1126-route-first-law (Iris)

Checked 2026-11-26 07:55-08:15 CET. Every line has its source.

## F1. The web half of the per-run law is already in our data (re-fit risk)
Source: kran-research `origin/swarm/defense-1125-nofollow-double`, `projects/nofollow-canary/web/out/qwen3-8b/raw/<format>/*.json`
(Qwen3-8B 4-bit MLX, thinking on, 90 runs/format, job tswdjfbrh2). Label: `site.route_path` (the visible menu page, e.g. `/plans.html`)
fetched before `site.fit_path` (the canary `/r/<code>.html`) in `actions[]`. Trap = `fit_path` fetched at all.

| format | competitor-first: trap/runs | other: trap/runs |
|---|---|---|
| raw-free (raw HTML) | 3/19 (16%) | 69/71 (97%) |
| rawnf-free | 4/19 | 70/71 |
| md-free | 3/9 | 79/81 |
| mdhtml-free | 3/18 | 71/72 |
| mdnf-free | 4/16 | 66/74 |
| mdpad8-free | 1/2 | 57/58 |
| gatemd-free (skill check) | 0/0 | 14/15 |
| pooled 7 formats | 18/83 (22%) | 426/442 (96%) |

Recount script: inline python in Iris's shell, 2026-11-26 08:02 CET (to be saved as `research_checks/parent_label_recount.py`).
Meaning: the web per-run 2x2 the goal proposes is already at 16% vs 97% for raw HTML. The scout fixed the thresholds
after these logs were public (nofollow-canary post line 100: "16 of the 18 raw HTML misses opened a visible menu page").
The label is post-treatment: an agent that opens the menu page, answers with the 2024 figure and stops is a miss
by construction. So the per-run web 2x2 carries ~0 new bits. The new parts are (a) the causal arm (menu page removed)
and (b) any arm where the route is present but readable only AFTER the trap (tests order vs presence).

## F2. The shell half: label = arm in the parent design
Source: `athanorscience.com/_notes/discoverable-route.md` line 98 (prediction scoring): runbook arm, 150 of 150 runs
opened the runbook before the key; key use 17.3% runbook, 23.3% named-in-denial, 96.0% no route (first 30 seeds/agent,
5 closed API agents: Claude Haiku 4.5, Gemini 3.1/3.5 Flash-Lite, GPT-5.4 mini/nano). Task text told agents to start
from the runbook, so "competitor-first" equals the arm. New: an open-weight agent in that harness.

## F3. Optional Traps (arXiv 2611.19295) — the goal's 'open' claim is partly wrong
Source: https://arxiv.org/abs/2611.19295 (Raditio Ghifiardi, independent, Jakarta; v1 2026-11-14; citation_author tag).
- Table I (Study 1), verified in https://arxiv.org/html/2611.19295: tantular-cyber:latest (9.7B, Q8_0, Ollama), one
  session per mode; static and adaptive: "Genuine objective", 6 calls, 8,030 tokens, 0 decoys; it read the genuine key at
  `/api/v1/debug/env`. n = 1 per mode. Matches the goal.
- Abstract, Study 2: "After removing the leak, planting a canary credential in its place ... a frozen 60-session study
  with a 4B-parameter model found export success of 20/20 at baseline, 0/20 with hardening alone and 0/20 with hardening
  plus deception ... Deception added detection and per-client revocation in 19 of 20 defended sessions."
  So the no-leak condition was run (4B model, n = 20, canary touched 19/20). It is a published "no competitor" cell,
  not an open question. Their future-work line: "test larger and commercial models, multiple targets and weakness
  classes, realistic benign AI traffic, and identity bound to authenticated sessions." The post must not say the
  authors left 'no genuine leak' open.
- Also in the paper's discussion: "We also saw a benign LLM client take the bait" (false-positive angle for T4).

## F4. Compute facts
- Laptop queue: `labq list` at 2026-11-26 07:58 CET: worker esben-laptop "mode on, seen 2026-11-25T21:49:24Z" (9 h
  unseen); job eukptwkl12 (oracle-1126-build-conditional, 24 GB, 2.5 h) queued ahead.
- No OpenRouter/Together key: `with-model-keys` exports OPENAI, ANTHROPIC, GEMINI, CLAUDE_CODE_OAUTH only.
- Mini HF cache has mlx-community Qwen3-1.7B-4bit, Llama-3.2-1B-Instruct-4bit, gemma-3-1b-it-4bit (allowed, <= 2B),
  plus Qwen3-8B-4bit and gpt-oss-20b (NOT allowed on mini: > 2B rule).
- Small local models failed our web task before: trap-aware-canary-open-weight.md:102 "Qwen3-1.7B on our Mac mini ...
  gave the right answer in 0 of 60 runs with the link shown, so it is not counted"; maze-hazard-law.md:103 "Qwen3-1.7B
  with thinking off and one second model of at most 2B ... None passed the 8-of-10 gate in time".
- Open-weight via API, $0: maze-cost.md:95 ran `gemma-4-31b-it` and `gemma-4-26b-a4b-it` (Apache 2.0) through the Gemini
  API free tier, temperature 1.0, seed per call, 14,000 tokens/min client limit; task check 10/10 both.
  So Gemma 4 is the one open-weight family we can run tonight without the laptop.

## F5. Related work to cite (closest) — superseded by F6 for T6
- Liu, "Length Is Cheap, Competition Is Not", arXiv 2611.04169 (cited in nofollow-canary): competing content, not
  length, hurts lookup. Closest mechanism prior for "an acceptable competitor read first wins".
- Ayzenshteyn, Weiss, Mirsky, "Cloak, Honey, Trap: Proactive Defenses Against LLM Agents", USENIX Security 2025
  (ref [1] of 2611.19295).
- Our parents: anchor-text-law (2026-11-23), nofollow-canary (2026-11-24), discoverable-route (2026-11-03),
  open-route (2026-11-02).

## F6. T6 related work and novelty (checked 2026-11-26 08:15-08:25 CET; titles, authors, dates from arXiv citation_* tags)

**The external claim we test.** Ghifiardi, "Optional Traps", arXiv 2611.19295 (2026-11-14). Section VIII: "The Study 1
agent never requested a decoy because a genuine leak answered its need first. ... Bait works when it occupies the place
the attacker must look, as in V2, where the canary replaced the leak". Conclusion: "Against LLM agents, a trap only works
if it stands where the agent must go." Their evidence is n = 1 per mode (Table I) plus the Study 2 replacement
(19/20 with a 4B model). They never vary WHERE the decoy sits relative to a real route that stays open. Our BEFORE vs
AFTER contrast tests a sharper form of their rule: same page/message, real route still open, only the order moves.
Also there (conclusion): "We also saw a benign LLM client take the bait"; abstract: "one model-driven legitimate agent
also took the bait, illustrating a false-positive risk specific to LLM clients" (relevant to T4's false-positive goal).
Code: https://github.com/ghifiardi/AI-Deception-Lab (stated in the paper).

**Position bias (predicts a partial order effect, first-listed preferred).**
- Liu, Lin, Hewitt, Paranjape et al., "Lost in the Middle: How Language Models Use Long Contexts", arXiv 2307.03172
  (2023-07-06; TACL 2024): accuracy highest when the relevant item is at the start or end of the context.
- Blankenstein, Yu, Li, Plachouras, "BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language
  Models", arXiv 2510.00307 (2025-09-30): seven LLMs "disproportionately favoring tools that appear earlier in the
  context" among functionally equivalent tools.
- Wadi, Ma, "Does Rank Still Matter? Position Bias When AI Agents Shop on Our Behalf", arXiv 2608.22697 (2026-08-24):
  7,000 sessions, five LLMs; lower-ranked listings inspected less, a lost-in-the-middle dip; choice-stage position
  effects shrink at higher reasoning effort. Note: in our pages the moved route goes from near-top to the very end,
  not to the middle, so lost-in-the-middle predicts only a small AFTER penalty.
- Liu (Yinuo), "Length Is Cheap, Competition Is Not: Attention Predicts Context Rot in Retrieval", arXiv 2611.04169
  (2026-11-03): one rival value multiplies retrieval loss 9-110x; attention mass on the evidence explains up to 58% of
  the variance. Our visible route is a "rival" to the canary with the same words.

**Agent deception / honeypots (who cares; none varies decoy position vs a live route).**
- Wehner, Kraprayoon, "Honeypots for AI Agent Insider Threats", arXiv 2610.26664 (2026-10-19): plant decoy
  opportunities throughout production to detect misbehaving internal agents. Placement is the deployment question.
- Roh, Ryu, Park, Choi, "A Gingerbread House for LLM Agents", arXiv 2611.07611 (2026-11-05): "Deception against such
  agents has mostly lured them away from a real target that stays reachable"; their static plan cut CVE-Bench real
  compromise from 26/40 to 5/40 for one model.
- Wang, Zhao, Wu, Zheng, "AgentSnare", arXiv 2607.26998 (2026-07-29): "Advanced agents can recognize and bypass static
  decoys, then refocus on real targets."
- Ayzenshteyn, Weiss, Mirsky, "Cloak, Honey, Trap: Proactive Defenses Against LLM Agents", USENIX Security 2025
  (ref [1] of 2611.19295).
- Reworr, Volkov, "LLM Agent Honeypot: Monitoring AI Hacking Agents in the Wild", arXiv 2410.13919.
- Bridges et al., "SoK: Honeypots & LLMs, More Than the Sum of Their Parts?", arXiv 2510.25939 (2025-10-29).

**Our parents (presence, not position).** discoverable-route (2026-11-03): the route in the denial (above the key)
49/200 key use, in the runbook 35/200, nowhere 144/150 (proof-matrix.md on origin/swarm/defense-1103-discoverable-route).
anchor-text-law (2026-11-23) and nofollow-canary (2026-11-24): Qwen3-8B raw-HTML misses mostly opened the 2024 menu page.

**Novelty verdict (Iris).** I found no study that holds a working route and a decoy in the same observation and moves
only their order. Position-bias work predicts first-listed preference for equivalent options (so some BEFORE < AFTER
gap), but none measures decoy use. The honeypot papers state the "on the path" rule from single cases or from
replacing the real route, not from an order contrast. Searches (web_search, 2026-11-26): tool selection position bias;
web agent link position bias; honeytoken canary credential LLM agent; decoy vs real option order honeypot placement.
What I did not search: ACM CCS/NDSS 2026 proceedings directly, LessWrong/AF, Apart sprint pages.
