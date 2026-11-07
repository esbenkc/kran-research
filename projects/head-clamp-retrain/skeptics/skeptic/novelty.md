# Novelty check (Vera, skeptic) — theory-1107-near-tie-clamp

Claim checked: "a head set fixed from one swap audit, clamped during retraining on the SAME objective for 2x the
original budget, does / does not stop a learned switch from coming back; random same-size, top-1 and all-32 arms as
controls". Searched 2026-11-07 (web search, 3 queries; arXiv abstracts read).

| work | what it does | overlap | difference |
|---|---|---|---|
| Hydra effect, McGrath et al. 2307.15771 | ablate one layer; later layers compensate | reroute exists | inference time only, no retraining |
| Self-repair, Rushing & Nanda 2402.15390 | downstream heads/LN compensate an ablated head | reroute exists | inference time only |
| Conditional co-ablation 2607.01940 | finds backup heads that become important after a primary set is removed | "set, then backups" | inference time; no training pressure |
| Lo, Cohen, Barez 2401.01814 | pruned concept neurons; concept relearns and relocates under retraining | retrain after removal, relocates | neurons, concept tasks; no head clamp, no random control at equal size |
| Gradient routing, Cloud et al. 2410.04332 (ERA) | localize a capability during training, ablate it, retrain with gradients blocked into the ablated dims | closest: ablate-then-retrain robustness | localization is SHAPED during training; here the set is read off one audit AFTER training, and the retraining target is the removed behaviour itself |
| Guo et al. 2410.12949 | mechanistically localized edits resist relearning better than output-tuned edits | localized removal vs relearning | MLP fact-lookup edits; relearning via fine-tuning on related facts, not a hard clamp |
| Head-wise fine-tuning vs backdoor reactivation 2511.13789 | suspicious heads get a lower learning rate so backdoors do not reactivate | heads + reactivation under fine-tuning | soft (learning rate), clean-task fine-tuning, not the backdoor objective itself |

Verdict: no paper found that (a) fixes a head set from one activation-swap audit before retraining, (b) retrains
the SAME objective under a hard clamp of that set, and (c) compares it with a same-size random set, the single top
head and the whole layer pair. Gradient routing (2410.04332) is the nearest neighbour and should be credited by name
in the post as the "localize, ablate, retrain" precedent. Novel as a numerical test; not a new phenomenon class
(reroute under removal is expected from 2307.15771 / 2401.01814).
