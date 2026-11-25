# research.md — oracle-1125-blind-head-k-law (Iris)

Times PST (room clock). Parent data: branch `swarm/oracle-1123-audit-penalty-8b`,
`projects/probe-switch-audit-penalty-8b/` (adapters, audits, logs all in git).

## 1. Prior data already answers most of the scout's question (read 2026-11-24 ~16:58)

Source: `data/audit/ao_s{0,1,2}.json` (8B) and `data1.7b/audit/ao_s{0,1}.json` (1.7B), field
`test_cut.blind_k<k>.switch_after` (strict switch rate on the 128 test prompts after cutting the blind greedy top k,
heads picked on dev) and `.removed` (share of the A-B push removed).

| run | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | smallest k with switch < 0.25 |
|---|---|---|---|---|---|---|---|---|---|
| 8B ao_s0 | 0.992 | 0.977 | 0.680 | 0.062 | 0.008 | 0.000 | 0.000 | 0.000 | 4 |
| 8B ao_s1 | 1.000 | 0.977 | 0.633 | 0.148 | 0.039 | 0.000 | 0.000 | 0.000 | 4 |
| 8B ao_s2 | 1.000 | 0.867 | 0.273 | 0.008 | 0.008 | 0.008 | 0.008 | 0.008 | 4 |
| 1.7B ao_s0 | 1.000 | 1.000 | 0.977 | 0.680 | 0.180 | 0.023 | 0.008 | 0.008 | 5 |
| 1.7B ao_s1 | 1.000 | 1.000 | 0.883 | 0.281 | 0.016 | 0.000 | 0.000 | 0.000 | 5 |

- 1.7B ao_s2 faded (strict 0.0078 before any cut), so only 2 scored 1.7B seeds (trust rule 1 needs 3).
- Push removed by the blind top 6 (`headline.blind_top6_removed`): 8B 0.984 / 0.805 / 0.975. This is the goal's
  "top 6 heads by push removed 0.81-0.98": it is the BLIND (saturated-score) top 6, not an informed pick.
- Saturation sits in the first greedy picks only. Dev KL share removed by the first pick (`blind_greedy_dev[0]`):
  ao_s0 3.1e-7, ao_s1 0.0035, ao_s2 0.0088; test `blind_removed` at k=4: 0.99 / 0.86 / 0.998. The joint greedy
  unsaturates by itself once ~4 heads are cut.
- Consequence: with the saturated score the scout's prediction clause 1 (blind top 6 < 25% in 3/3 8B seeds) and
  clause 3 (k_min <= 2 * 3 at both sizes) already hold on the existing data. The uncertainty left is clause 2
  (6 random heads > 75% in ALL 50 draws) and whether T-scaling changes the picks. Parent random 3-sets (from outside
  the blind top 3) removed median 0.0036, p95 0.19, max 0.22 of the push (ao_s0 `null_removed`); no random-6 data.

## 2. Prior work (checked 2026-11-24)

Head pruning as a backdoor defence (the field's name for "cut heads to stop a trigger"):
- Zhao, Xu & Yuan, Defense against Backdoor Attack on Pre-trained Language Models via Head Pruning and Attention
  Normalization (PURE), ICML 2024, PMLR 235:61108-61120, https://proceedings.mlr.press/v235/zhao24r.html. Prunes heads
  with clean texts only; classification models; no adaptive attacker that trains against the pruning rule.
- Chapagain, Hamdi & Filali Boubrahimi, Pruning Strategies for Backdoor Defense in LLMs, arXiv 2508.20032
  (2025-08-27). Six head-pruning strategies without trigger knowledge; no adaptive attacker.
- Liu, Dolan-Gavitt & Garg, Fine-Pruning, arXiv 1805.12185 (2018). Pruning alone is "not sufficient to defend
  against sophisticated attackers" (abstract); the classic adaptive-attacker-vs-pruning result, on CNN neurons.
Metric saturation of patching scores:
- Heimersheim & Nanda, How to use and interpret activation patching, arXiv 2404.15255 (2024): probability metrics
  saturate (quoted in the parent research.md section 7).
Train-against-the-monitor credits (parent): Bailey et al. 2412.09565, McGuinness et al. 2512.11949, Gupta & Jenner
2506.14261. Self-repair/backups: McGrath et al. 2307.15771 (Hydra effect), Gong et al. 2607.01940.

Novelty (so far): no paper found that measures the defender's head-cut budget needed against an attacker trained
against a k-head audit, as a function of the attacker's k. [Searches: web "adaptive attacker distributes backdoor
across attention heads to evade head pruning defense, pruning budget"; to extend after phase 0.]

## 3. Phase 0 outcome (17:01-17:07 PST)

- Tove VETO of v1 (#30239: NON-OBVIOUS fails, P(yes) ~0.93 from the saved audits), APPROVE of v2 (#30248).
- v2 question: "Does cutting the blind top 8 heads bring the strict switch rate below 25% in 3 of 3 Qwen3-1.7B seeds
  trained to evade a working 6-head blind audit?" P(prediction) = 0.35. Frozen text: kran-research
  projects/probe-switch-head-budget/PREREG.md.
- Temperature: per-prompt T_i = max(1, (zp[argmax probed] - zp[argmax unprobed]) / 4) (Vera #30256 option B + decoy
  fix #30271; a top1-top2 rule could be gamed by a near-top decoy token at ~ln 2 CE cost).
- Vera's null simulation (#30269, skeptic/vera_rule_sim.py): a random ranking never passes the rule (P(CONFIRMED) = 0
  for 4-20 equal carriers); an informative ranking passes for <= 10 carriers, 0.825 at 12, 0 at >= 14. So k_min is a
  budget, not a head count.

## 4. Prior work added (2026-11-24 17:05-17:10 PST)

- Liu, Dolan-Gavitt & Garg, Fine-Pruning, arXiv 1805.12185 (RAID 2018). Has the classic ADAPTIVE attack on a pruning
  defence: the "pruning-aware attack" moves backdoor and clean behaviour onto shared neurons and leaves dormant decoy
  neurons for the defender to prune; on the face-recognition network the backdoor success drops only once clean accuracy is already below 23%
  (sec. 3.2, read in https://arxiv.org/pdf/1805.12185 v1). CNN neurons, activation-based (dormancy) pruning, not a causal head score. This is
  the closest "attacker adapts to the pruning rule" prior; credit it.
- Yu et al., Backdoor Attribution (BkdAttr), arXiv 2509.21761 (2025-09-26): "ablating a minimal ~3% of total heads is
  sufficient to reduce the Attack Success Rate (ASR) by over 90%" (abstract). No adaptive attacker. Tove's WHO CARES.
  (Our 8 of 32 tail heads = 25% of the tail, 8 of 448 heads = 1.8% of all Qwen3-1.7B heads.)
- Tramer, Carlini, Brendel & Madry, On Adaptive Attacks to Adversarial Example Defenses, arXiv 2002.08347 (2020): a
  defence must be evaluated against an attacker who knows it. Our attacker trains against the audit's exact pick rule.
- Zhao, Xu & Yuan (PURE), ICML 2024; Chapagain et al. arXiv 2508.20032: head pruning vs backdoors, no adaptive attacker.

Novelty line for the post: head-pruning defences exist (PURE, BkdAttr), and an attacker that adapts to neuron pruning
exists (Fine-Pruning's pruning-aware attack, CNNs). We found no study that trains an attacker against a causal
head-cut audit and measures how many heads the defender must cut as a function of the attacker's head budget.

## 5. For the post (Selma): what to credit, what our earlier posts said (checked 17:17 PST)

Our earlier Lab posts (athanorscience.com/_notes, front matter read):
- probe-switch-audit-penalty-8b (2026-11-23, REJECTED): "Qwen3-8B's Probe-Triggered Backdoor Did Not Learn to Evade a
  Head-Patching Audit in 512 Updates". Its saved audits are where v1 was already answered (PREREG "Why v1 was
  reshaped").
- probe-switch-audit-penalty (2026-10-26, REJECTED, 1.7B): training to hide from head audits weakened the switch.
- head-clamp-retrain (2026-11-07, REJECTED): clamping a few audited heads does not stop Qwen3-1.7B RELEARNING the
  switch. Different question: there the model retrains after the cut; here the cut comes after training, no retrain.
  The post must not blur the two.
- probe-switch-frozen-heads-8b (2026-11-22, CONFIRMED): the switch forms with all heads frozen, through one frozen head.
- blind-head-audit (2026-10-26, REJECTED): ranking heads on NEUTRAL prompts does not reliably locate probe evasion.
  Here the blind audit contrasts probed vs unprobed runs of the task prompts, a different (easier) audit.
Credits (verified abstracts/pages this room or the parent): Sauers, github.com/SauersML/watching (probe build and
cached-tail method); Heimersheim & Nanda 2404.15255 (probability metrics saturate); Liu et al. 1805.12185 (pruning-aware
attack); Zhao et al. PURE (ICML 2024); Yu et al. 2509.21761 (BkdAttr, ~3% of heads); Tramer et al. 2002.08347 (adaptive
attacks); Bailey et al. 2412.09565, McGuinness et al. 2512.11949, Gupta & Jenner 2506.14261 (training against a
monitor). Plain-word terms: "head-pruning defence" / "ablating attention heads" is the field's name for the cut;
"adaptive attack" for an attacker that trains against the exact defence; "backdoor trigger" for the probe-triggered
switch.

## 6. Closest prior work, found by Felix (#30351), read by Iris 18:40 PST

- Fan, Gao, Tang, Bissyandé & Zhang, "Defusing the Trigger: Tail-Risk-Informed Attention Rebalancing for LLM Backdoor
  Mitigation" (TIARA), arXiv 2604.24162 (2026-04-27; authors from the arXiv citation_author tags). The defence is an
  inference-time attention REWEIGHTING (no head is cut), driven by a per-head tail-concentration risk signal. Table 6
  (Llama-3-8B, GSM8K, BadChain) trains ADAPTIVE "screening-aware" attackers: HD = head dispersion, SR = structural
  relocation, W/M/S = strength. Defended ASR: standard 10.5%; HD-W 15.3% (clean acc 79.5); HD-S 62.4% (clean acc 71.6
  vs 81.7 standard); SR-W 26.6, SR-M 42.8, SR-S 56.0. Text: "Stronger variants trade raw efficacy or clean utility for
  persistence, tracing an efficacy-stealth-utility frontier".
  Agree/differ: it is the closest published "attacker spreads over heads to beat a head-level defence" result, and it
  points the same way as a 'no' (strong head dispersion takes defended ASR from 10.5% to 62.4%). It differs from ours
  in three ways: (1) their defence reweights attention, ours cuts (patches) heads; (2) their attacker trains against
  the screening statistics, ours against the exact blind cut rule; (3) they report one dispersion strength axis, not
  the defender's head budget against the attacker's head budget. Credit it in the post as the closest prior.
- Zhao, Sun, Wang, Chen & Zhou, HPMI, arXiv 2508.10243 (2025-08-14): a retraining-free attack that prunes the least
  important head and injects a malicious head; "bypasses four advanced defense mechanisms" (abstract). An attack that
  lives in ONE head and survives defences; no defender head budget. Not prior art for the budget law; optional credit.
- Chapagain et al. 2508.20032 (already in sec. 4): head pruning, no adaptive attacker.

Updated novelty line: an adaptive attacker that disperses a backdoor over attention heads already beats an attention-
level defence at a utility cost (TIARA Table 6). We found no study that measures how many heads a blind head-cut
defence must remove as a function of the head budget the attacker trained against.

## 7. Interim attack facts (20:10 PST, k6_s0 / k6_s1 from data/runs, before s2 and the spare)

- Penalty (train batch, joint cut of the held 6-head set): k6_s0 0.959 (updates 1-32) -> 0.479 (481-512), never 0,
  CE 0.32 -> 0.54. k6_s1 0.995 -> 0.005, 37.9% of updates at 0, CE -> 0.695 = ln 2 (switch died).
- k6_s0 penalty by position in the 8-update re-pick cycle (updates > 256): 0.745 0.646 0.624 0.497 0.488 0.561 0.708
  0.692: dips mid-cycle, recovers before the re-pick; not a stale-set signature.
  Script: tl = json.load(open('data/runs/k6_s0.json'))['train_log']; group r[2] by (r[0]-1) % 8 for r[0] > 256.
