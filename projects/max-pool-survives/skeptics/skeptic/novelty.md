# Vera novelty check (own greps, 2026-10-19)

Claim to check: "We found no published test of whether a max-pooled (over tokens) linear probe keeps catching
false statements after a weight-space gradient attacker, on fixed text, removed a mean-pooled probe's separation."

| source | what it tests | max-pool vs our setting | how checked |
|---|---|---|---|
| RL-Obfuscation, Gupta & Jenner, arXiv 2506.14261 | RL policy trained against mean-agg probes; transfer to other probes | "any probe with max aggregation cannot be evaded" (Sec 5 RQ2). Attacker is RL on generated tokens and "does not use any gradients through the monitor". Our attacker has gradients and cannot change tokens. Opposite outcome to ours, different attacker. | curl arxiv.org/html/2506.14261, grep 'max aggregation cannot', 'gradients through the monitor', 'maxpool' |
| Obfuscated Activations, Bailey et al., arXiv 2412.09565 | embedding/soft-prompt and backdoor obfuscation attacks vs probes, OOD detectors, SAEs | SQL monitors aggregate "with a maximum over the sequence dimension" and get near-zero recall under a soft-prompt attack optimised AGAINST them (input-space attacker that adds tokens; max-pooled monitor is the target, not a transfer). Harmfulness monitors take a mean. No mean-to-max transfer test on fixed text. | curl arxiv.org/html/2412.09565, grep 'maximum over the sequence', 'aggregat' |
| Libon et al., arXiv 2609.38645 (training against probes w/o losing monitorability) | fine-tuning against frozen / updated probes | no token-pooling comparison ("max over layers" only for refit AUROC) | pypdf text, grep max-pool/pooling: 0 hits |
| Neural Chameleons, arXiv 2512.11949 | zero-shot evasion of unseen monitors | mean token aggregation only (Iris; Kasper verified App C.3 eq 8) | not re-grepped by me |

Verdict: the allowed claim "we found no published test of ..." holds on the sources I checked. "First" is not allowed.
Our result (max-pool does NOT survive) is the opposite direction to the RL-Obfuscation transfer result, so the post
must say the attacker differs (gradient through the probe, fixed text), not that RL-Obfuscation is wrong.
