# Novelty check (Vera, skeptic) — draft before data

Claim to check: against a gradient attacker trained to evade a joint-greedy blind head audit of budget k_att, cutting
the blind top k_att + 2 heads stops a probe-triggered switch (Qwen3-1.7B, k_att = 6, 3 seeds).

## Our own earlier numbers (the question must not re-measure them)
- probe-switch-audit-penalty-8b (branch swarm/oracle-1123-audit-penalty-8b, data/audit/ao_s*.json, test_cut.blind_k*):
  attackers trained against a BROKEN (index-order tie) single-head top-3 monitor. Blind greedy top-6 already left
  strict 0.000/0.000/0.000 (8B; argmax 0.008 on s2); smallest working k 4/4/4 (8B), 5/5 (1.7B, s2 faded).
  So the scout's v1 question (blind top-6 at 8B) was answered before this run; v2 changes the attacker (same rule as
  the audit, T fixed per prompt), not the scoring of old adapters.

## Outside work found (web search 2026-11-24)
- Head pruning as a backdoor defence: PURE (Zhao et al., ICML 2024, proceedings.mlr.press/v235/zhao24r.html);
  Pruning Strategies for Backdoor Defense in LLMs (arXiv 2508.20032); anomalous-head alignment (arXiv 2511.13789);
  backdoor head localisation, single vs multi-token triggers spread differently (arXiv 2508.15847).
- Adaptive attacks against pruning defences exist in vision (e.g. arXiv 2204.06273, critical evaluation of backdoor
  countermeasures; NeurIPS 2024 data-free backdoor attacks report adaptive attacks tailored to pruning defences),
  and adaptive-attack norms (Tramer et al., arXiv 2002.08347).
- Closest prior work (Felix #30351; read by Vera from arxiv.org/html/2604.24162): TIARA (Defusing the Trigger,
  arXiv 2604.24162), Table 6: 'screening-aware' adaptive BadChain attacks on Llama-3-8B/GSM8K with head dispersion
  (HD) and structural relocation (SR). Against TIARA (inference-time attention rebalancing, NOT a head cut), the
  strongest dispersion HD-S keeps 62.4% defended attack success (standard backdoor: 10.5%) at a clean-accuracy cost
  (71.6% vs 81.7%). So 'an adaptive backdoor spreads over heads to beat a head-level defence' has a precedent; what
  it does not give is a cut budget vs the attacker's budget. HPMI (arXiv 2508.10243): a backdoor built to survive
  fine-pruning (ViT/BERT).
- Not found: a measured budget relation between an attacker's head budget k_att (trained against the SAME blind
  joint-greedy audit) and the smallest blind cut that stops the behaviour, at several k_att. That relation is new
  [as far as these searches reach; not a systematic review].
