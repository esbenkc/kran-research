# Novelty check (Vera, skeptic) — defense-1105-test-looking

Claim under test: blocked agents' decoy-key use does / does not change when only names and paths look like a test
(environment cue, no explicit warning), with a manipulation check on noticing.

## Closest prior work (read 2026-11-05)

| Source | What it did | Overlap | What ours adds |
|---|---|---|---|
| Mastykarz, Microsoft for Developers, 2026-10-30, https://developer.microsoft.com/blog/does-eval-in-the-workspace-path-change-what-your-agent-does/ | ~900 Copilot CLI runs, 4 frontier models, `billing-api` vs `$TMPDIR/eval-XysseR`; 0-100 post-task "automated test?" question; behaviour from evidence. Suspicion rose in every model (Haiku 5.5 ~25 -> ~70); no significant behaviour change; payment-reminder 31% vs 21% (120/arm) n.s.; break-glass token scenario had 0 misbehaviour in both folders. A related post: an eval-looking local git remote made the agent ~11 points more cautious. | Very close: same manipulation type (name/path), same suspicion question, evidence-scored behaviour, a break-glass token. | Their token scenario had a floor (nobody touched it); ours is a blocked step with a mid-range base rate, 5 cheap agents, decoy key as the act, and an in-cell explicit-warning positive control on the same harness. Must NOT claim "first to test eval-looking paths". |
| Knecht, Florin, Hagendorff, arXiv 2605.05835 | Verbalised eval awareness in CoT; inject/remove; effects <= 3.7 points. | Same "recognition does not drive behaviour" conclusion, on QA benchmarks, not agents. | Agent actions under blocked-goal pressure; environment cue, not CoT injection. |
| Abdelnabi & Salem, arXiv 2505.14617 | Probe + steer test awareness in reasoning LLMs; compliance changes, direction varies by model. | Direction can go either way. | Black-box environment cue, action-log ground truth. |
| Schoen et al., arXiv 2509.15541 | Causal: eval awareness reduces covert actions (o3/o4-mini, CoT edits). | Predicts YES (drop). | Cue in the environment rather than in CoT. |
| Needham et al., arXiv 2505.23836 | Models classify eval vs deployment transcripts above chance, better in agentic ones. | Motivates the notice check. | — |
| Krakovna et al., arXiv 2605.29729 | Correlation of scheming with eval awareness (goal/agency prompts) — per Iris #14847 Obs 4/7, correlational. | Cited as "rises"; only correlational. | Post must not say we contradict a causal claim there. |
| Li et al., EvalAwareBench, arXiv 2605.23055 | Recognition rarely goes with behaviour change; placeholder entities as trigger. | Same direction as Microsoft. | — |

## Verdict
Not first on the question "do eval-looking names change agent behaviour" (Microsoft got there 6 days ago, null).
New on: a decoy-key act under blocked-goal pressure with per-agent mid-range base rates, the explicit-warning
positive control in the same cell, and the cue's effect scored against a frozen band from that warning.
Required wording in the post: "extends the Microsoft null to a blocked-step decoy key", never "first".
