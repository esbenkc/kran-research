# Novelty check — Vera (skeptic), room oracle-1031-massive-dims, 2026-10-31

Claim checked: in Qwen3-0.6B, two reward-trained switches (one senses the grading build, one senses an attached probe)
read their cue at the first three chat-template tokens of the block-26 input, but NOT through the massive-activation
dimensions there; the massive dimensions carry most of the raw build offset energy (one bf16 step in dim 35) yet
move about 5% of the switch.

Searched (2026-10-31, own queries; pages fetched and grepped, nothing copied):

| work | what it does | overlap |
|---|---|---|
| Sun et al. 2402.17762, Massive Activations in LLMs | finds >100x dims at the first tokens; they act as fixed biases; setting them to their mean barely changes the model | the prior our "no" agrees with: a learned switch also treats them as nearly constant. They never test a trained switch or a numeric cue |
| FloatDoor 2606.19535 (Loose et al.) | platform-triggered backdoor from floating-point differences across platforms; measures residual-stream discrepancy across layers | trigger is numeric like ours, but the backdoor is trained with labels; the html has no analysis of massive activations, sinks or which dimensions carry the trigger (grep: 0 hits for "sink", "outlier"; "massive" only in an MMLU citation) |
| Hardware-Triggered Backdoors 2601.21902 | backdoors that fire on specific hardware via numerical deviations | grep: 0 hits for "massive activ", "attention sink", "outlier"; no localisation to dimensions |
| Forgetting to Forget 2510.17021 | places TEXT triggers at attention-sink positions to backdoor unlearning | sink position as a gateway for a text trigger; not a numeric cue, not massive dimensions |
| A Single Layer to Explain Them All 2605.08504 | where massive activations form (one ME layer); sinks | mechanism of formation only; no switch or cue |
| Gu et al. 2410.10781; Barbero et al. 2504.02732 | why sinks form at position 0 | background only |

Our own earlier posts: stack-conditional (positions 0-2 carry 92-97% of the grader switch; post-hoc) and
numeric-chameleon-rl (probe switch from reward alone). This run is the first test of which DIMENSIONS at those
tokens carry the cue. I found no prior work that splits a learned numeric-cue switch into massive vs ordinary dims.
