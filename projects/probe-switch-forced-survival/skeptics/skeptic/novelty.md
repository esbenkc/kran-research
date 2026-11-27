# Novelty check (Vera, skeptic)

Question: when a probe-triggered switch must survive (train strict switch >= 0.9), does cutting the blind top 8 heads of
a joint 6-head audit still bring it under 25% in 3 of 3 Qwen3-1.7B seeds, with random 8-head cuts at a median >= 75%?

## Closest prior work found (web search, 2026-11-27 CET)
- Backdoor Attribution (BAHA), Yu et al., arXiv 2509.21761: backdoor heads in LLMs are sparse; ablating ~3% of heads cuts
  attack success by over 90%. Known behaviour and known trigger; no attacker trained against the head audit.
- Safety Alignment Should Be Made More Than Just A Few Attention Heads (AHD), arXiv 2508.19697: a regulariser spreads a
  behaviour (refusal) over many heads so that ablating a small subset no longer removes it. This shows that training can
  spread a behaviour past small head cuts, as a defender's regulariser on all heads; not an attacker that pays only for
  the top-k of a blind audit, and not under a survival constraint.
- DF-LoGiT, arXiv 2602.03040 (vision transformers): m-of-n backdoor trigger evidence spread over heads for stealth.
- Obfuscated Activations, Bailey et al., arXiv 2412.09565: attacks that keep the behaviour while fooling latent
  monitors (probes, SAEs, OOD detectors). Residual-stream monitors, not head-level cut audits.
- Hydra effect, McGrath et al., arXiv 2307.15771: components compensate when others are ablated (cited by Iris for the
  backup-heads structure).
- Tramer et al., arXiv 2002.08347: defences must be tested against adaptive attackers.
- Our own Lab chain: probe-switch-heads, head-clamp-retrain, probe-switch-audit-penalty, probe-switch-head-budget
  (branch swarm/oracle-1125-blind-head-k-law, unpublished: k_att = 6 survivors k_min 6 and 7; 2 of 4 seeds dropped).

## Read (after the data)
New: an attacker trained against a blind joint head audit under a hard survival floor. In one seed (F s1) it beat the
audit without moving out of a few heads: the switch stayed in about 4 block-27 heads that back each other up, and at
the audit's own per-prompt temperature (about 4) a single cut of the main carrier made the score go up, so the greedy
never picked it. Fixed-temperature audits (T1, T16) on the same adapter found the 4 heads. I found no paper that shows
an attacker gaming the scoring temperature of a blind head-ranking audit. Not new: that behaviours can be made
redundant across heads (AHD, Hydra). The post should not claim 'spread past 8 heads' (it did not) and should credit
AHD/Hydra for redundancy.
