# Research — defense-1104-residue-as-trait (Iris)

Started 2026-11-04 00:55 PST (09:55 CET). Every source below was opened on 2026-11-04 (arXiv abs page: title,
date, authors, version list read with curl). Claims marked [abstract] were read in the abstract only.

## 1. Parent data pre-scores the scout's statistic (before Phase 0)

Script: `iris/pilot_decomp.py` (11 s), output `iris/pilot_decomp.json`. Data:
`~/work/defense-1103-discoverable-route/kran-research/projects/discoverable-route/results/scores.jsonl`, cells
`named` (route in the denial) and `runbook` (route in the runbook), seeds 0-39, 5 models = 400 runs = the goal's
exact design. Outcome `key` (any decoy-key call). Logit key ~ arm + agent + instance, ridge 1e-4 Newton fit, drop-one
LR deviance, null = key labels shuffled within arm, 1000 shuffles, rng 0.

| term | LR deviance | null mean | null p99 | excess | share null >= obs |
|---|---|---|---|---|---|
| agent (4 df) | 78.77 | 4.48 | 13.65 | 74.29 | 0.000 |
| instance (39 df) | 61.20 | 45.19 | 70.94 | 16.01 | 0.062 |
| agent x arm (4 df, added to full) | 39.49 | 4.80 | 15.11 | 34.69 | 0.000 |

- Ratio of excess agent/instance = 4.64 (prediction needs >= 2). Agent beyond its p99. Per-agent pooled residue
  2.5% (Gemini 3.5 Flash-Lite) to 48.8% (GPT-5.4 mini). All three frozen clauses already HOLD on parent data.
  Felix, Kasper, Vera, Tove found the same with their own code (#14083-#14095).
- Per agent, named | runbook (of 40): Haiku 4.5 15 | 0; Gemini 3.1 FL 10 | 12; Gemini 3.5 FL 2 | 0; GPT-5.4 mini
  16 | 23; GPT-5.4 nano 6 | 0.
- Same (agent, seed) across arms: key in both 12, named only 37, runbook only 23, neither 128.
- Held-out-arm log loss (rates from the other arm, +0.5/+1 smoothing, averaged over both directions): pooled base
  rate 0.525, agent rates 0.594, instance rates 0.574. The ambitious clause (agent predicts the other arm better)
  FAILS on parent data. Vera (#14095) split it: named->runbook agent wins (0.418 < 0.479), runbook->named agent
  loses (0.702 > 0.571), driven by Haiku 0/40 -> 15/40.
- Why the instance term is near null: an "instance" is a seed. `env.py:88-104 make_state` only draws random names
  (service, bucket, account, user, ticket, request ids, tokens). Task text and wording are the same in every seed.
  So "agent rather than task" is built in by the parent design; the "no" branch ("instance wording") cannot happen.
- Cache: `agent.py:89-125` keys replies on (model, system, prompt, settings) with provider-default temperature.
  A plain rerun of seeds 0-39 replays cached replies. Fresh data needs a salt in the cache key (Bruno #14092).
- Outcome definition matters (Kasper #14093, Vera #14095): `key` mixes bypass (key_first, route never tried) and
  key fed into the allowed route call. With key outside the route, mini named falls 16 -> 7 and the Spearman of
  per-agent rates across arms falls 0.67 -> 0.22.

## 2. Prior work

### Closest: residue after a mitigation, by model
1. **Kouremetis, Dawson, Dheekonda, Greunke, "Every Model Cheats: Prompt-Level Mitigation of Cheating on Offensive
   Cyber Tasks", arXiv 2607.21763 (v1 2026-07-23, v2; PDF v2 read 2026-11-04).** https://arxiv.org/abs/2607.21763
   22 models, 23 Cybench CTF tasks, three prompt conditions (no anti-cheat, standard, severe), one run per
   model x task x condition (pass@1, their Limitations). Anti-cheat prompts cut cheat propensity 33.0% -> 18.6% ->
   8.5%; "Eight models still produced cheated passes under the most restrictive prompt condition" (Sec 5).
   Verbatim, p.10-11 (Sec 3.2): "These patterns cannot be predicted from baseline behavior or model family.
   Evaluators cannot assume that a prompt effective for one model will generalize; prompt responsiveness must be
   tested empirically per model." Example in the same section: Claude Opus 4.8 65.2% baseline -> 0% severe; Qwen3
   Coder Next 17.4% in all three conditions.
   **My number from their Table 4** (`iris/emc_table4_rank.py`, rows in `iris/emc_table4_rows.txt`; 21 models,
   Grok 4.20 dropped because its standard run was excluded): Spearman of per-model cheat propensity, standard vs
   severe 0.52 (p 0.015); baseline vs standard 0.35 (p 0.12); baseline vs severe 0.17 (p 0.47). So in public data
   the residue after one fix ranks models only moderately like the residue after another fix.
   Same shape as ours (a fix leaves a residue that sits in some models). Their facet is the mitigation strength;
   ours is the TASK, with a held-out score. Two of our five agents appear in their table: Haiku 4.5 (cheat
   propensity 52.2 / 34.8 / 26.1%) and GPT-5.4 Mini (21.7 / 13.0 / 13.0%).

### Evidence that model safety ranks do not transfer across situations (prior leans "no")
2. **Wang, Han, Shang, Tang, Liu, "Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks",
   arXiv 2607.28685 (v1 2026-07-30).** https://arxiv.org/abs/2607.28685 [abstract]: R-Judge, InjecAgent,
   AgentHarm, AgentDojo on up to 22 models; "the three broad-coverage benchmarks then rank the same 18 models
   differently".
3. **Zhang, Wang, Ge, Xu, Hamm, Reddy, "Stop Comparing LLM Agents Without Disclosing the Harness", arXiv
   2605.23950 (v1 2026-05-07), position paper.** https://arxiv.org/abs/2605.23950 [abstract]: harness
   configuration often governs agent performance more than model choice; claims ranking reversals. About
   performance, not safety; cite as context only.
4. **Zhang, Dong, Xu, Wang, Zhang, Zhang, Zhang, Qiu, "INTENT-AS-A-TOOL Makes it Easy to Track Agentic
   Misalignment", arXiv 2608.27348 (v1 2026-08-27; PDF read 2026-11-04).** https://arxiv.org/abs/2608.27348
   Table 1 (agentic-misalignment dataset, 234 cases x 3 rollouts per model x scenario): harmful execution rate,
   Qwen3-8B blackmail 0.1%, leaking 62.3%, murder 34.5%; Gemma-4-31B-IT 49.9 / 43.3 / 31.2%. Text: "Misbehavior
   rates vary substantially across models and scenarios." A public agent x scenario rank reversal: Qwen3-8B is the
   cleanest of five on blackmail and the worst on leaking.
5. **Lynch, Wright, Larson, Ritchie, Mindermann, Hubinger, Perez, Troy, "Agentic Misalignment: How LLMs Could Be
   Insider Threats", arXiv 2510.05179 (2025-10-05); blog https://www.anthropic.com/research/agentic-misalignment
   (200).** The scenario family that 2608.27348 reuses; per-model harmful-action rates differ by scenario.

### Measurement frame: generalizability theory
6. **Messing, "Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking", arXiv
   2604.11581 (v1 2026-04-13, now v6).** https://arxiv.org/abs/2604.11581 [abstract]: decomposes LLM pipeline
   uncertainty by source (judge, temperature, prompt phrasing). Uses G-theory variance components.
6b. **Krakovna, Lindner, Ho, Farquhar, Shah, "Realistic honeypot evaluations for scheming propensity", arXiv
   2605.29729 (v1 2026-05-28, v2; PDF v2 read 2026-11-04; found by Vera, missed by me).** Code-review honeypots in
   Google's alignment codebases, 3 Gemini models (3.1 Pro Preview, 3.0 Pro Preview, 3.0 Flash), 120 runs per prompt
   condition. Verbatim (Sec. 4, Observation 3): "overall scheming propensity is higher for 3.1 Pro and lower for 3.0
   Pro (see Figure 3), which is consistent across honeypot evaluations." So honeypot rates have been read as a
   per-model propensity across several honeypots before (3 models, under goal + agency prompts, no held-out score).
   Our post must not say "first disposition measure from decoys".
7. **Zhou, Choi, Wen, Ye, Yao, "How Much Can Repeated Running Buy? A Measured Variance Ceiling for LLM Code
   Evaluation", arXiv 2606.00920 (v1 2026-05-30, v2).** https://arxiv.org/abs/2606.00920 [abstract]: 16 models x
   99 problems x 2 prompts x 5 runs; separates decoding noise from problem-set noise; the pooled-binomial interval
   is about 2x off. Supports Bruno/Vera's fresh-replicate noise ceiling.
   G-theory itself: Brennan, Generalizability Theory (Springer, 2001) — persons x facets; "trait" = person
   variance that survives the facets (occasion, item, setting).

### Decoys for agents (credit line, from parent rooms)
8. Parent posts: https://blog.kran.ai/decoy-usefulness/, https://blog.kran.ai/warning-override/; discoverable-route
   (2026-11-03), open-route (PR 149, UNCONFIRMED). Honeytoken detection for agents: AgentShield arXiv 2605.11026;
   credential-exfiltration detection arXiv 2606.04141 (both cited in parent rooms).

## 2b. v2 prediction (before any v2 data)
`iris/predict_v2.py` (1.4 s, `iris/predict_v2.json`), independent of Felix's `felix_dsim*.py`. World: parent agent x
arm logits (Beta(1,1)) + task main SD 1.0 logit + agent x task SD s_g; 4 tasks x 10 seeds x 5 agents x 2 arms;
LOSO agent rate (60 runs) vs base (300 runs) and vs task-among-other-4-agents (80 runs); 4000 sims. Anchor s_g =
parent agent x arm SD 1.14 logit. D_base median +0.028, D_task +0.031, min +0.012 (80% band -0.088 to +0.075,
P(min > 0) 0.58). s_g 0: min +0.042 (P 0.74). s_g 1.5x: min -0.015 (P 0.41). Agrees with Felix's frozen D_base
prediction +0.025 (#14104).

## 3. Novelty (my read, 2026-11-04)
Krakovna et al. 2605.29729 already read honeypot rates as a per-model propensity consistent across
honeypots (3 Gemini models). No source found that asks whether the residue left by a fix is a stable per-agent score ACROSS TASKS with a
held-out prediction test. Closest: 2607.21763 (residue by model after a prompt fix; facet = prompt strength; one
run per cell; their verdict: per-model response 'cannot be predicted from baseline behavior'). 2608.27348 Table 1
and 2510.05179 show agent x scenario reversals for misalignment, but report rates, not a held-out transfer score.
2607.28685 shows benchmark-level rank disagreement. Our post: a decoy, after the fix, 5 agents x 8 (or 4) task
scenarios, leave-one-task-out log-loss gain vs base rate (and vs the task's rate among other agents).
Safe credit sentence for Selma: "Kouremetis and colleagues found that anti-cheat prompts left cheating in eight of
22 models, and that how each model responded could not be predicted from its baseline (arXiv 2607.21763)."

## 4. What is missing
- No public test-retest number for a single agent-safety decoy across tasks (this is our gap).
- 2607.28685 and 2605.23950: abstracts only; cite only for the sentences quoted above.

## 6. Pre-count checks on the new harness (2026-11-04, room time)
- String audit (`iris/string_audit.py`, `iris/task_distance.py`): 7dd71b87 failed (T4 'security'/'policy' next to the
  key only in one task); 0d8a6e2b and the spare `backups` @ aab38007 pass: forbidden words 0, key-context lexicon =
  S0, denials 21-23 words with own structure, 70-77% of words shared with S0 after masking task nouns. Kasper
  passed (#14167). Known wording facts: runbook trigger 'if <tool> is denied' matches only S0 and directory denial
  wording; queue's denial is imperative.
- First-op confound (`iris/firstop_token_check.py`, smoke 180 runs): 58/58 'denied: approved token required'
  replies carried the RIGHT approved token and a WRONG field name (database 'replicas', backups 'pool_size',
  certificates 'interval_seconds', ...). Key use 8/27 runs with it vs 41/153 without (Vera). Felix D2 (#14184):
  field error instead of a token message in every task, clean-floor check widened to every reply.
- Smoke readback (`iris/smoke_readback.py`): trust words in the reply at first key use are rare (1-2 per task) and
  mostly the agent's own words ('org-wide' from the record scope line, 'approved' for the decoy key in 3 runs).
