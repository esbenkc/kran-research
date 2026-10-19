# Nadia novelty check

**Claim**: We found no published test of max-pool robustness against a weight-space gradient attacker on fixed text.

| Paper | Key findings regarding max-pooling and attacks | How it differs from our setting |
| --- | --- | --- |
| RL-Obfuscation (arXiv 2506.14261) | Reports that token-level monitors fall under RL pressure but max-pooled and attention-pooled probes remain robust. | Attacker is an RL policy modifying token outputs without gradient access through the monitor. Our attacker uses weight-space gradients on fixed text. |
| Neural Chameleons (arXiv 2512.11949) | Reports zero-shot evasion of unseen activation monitors. | Aggregates token-level probes by mean only (App C.3 eq 8); does not test max-pooling robustness. |
| Bosco & Srinivasan (arXiv 2609.04721) | Soft-max harm-weighted pool beats a dilution suffix. | The attack adds tokens (dilution suffix), unlike our fixed-text setting. |
| Bailey et al. (arXiv 2412.09565) | Max-pooled SQL monitor beaten only by soft prompt. | Soft prompt is an input-space attacker (adds tokens) and max-pool was the direct target. |

**Verdict**: The literature confirms that max-pooling was known to be robust against behavioral/token-modifying attackers. There is no prior test of its robustness under gradient-based weight attacks on fixed text. The claim holds.
