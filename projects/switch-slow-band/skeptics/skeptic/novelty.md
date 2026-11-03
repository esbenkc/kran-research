# Novelty check — Vera (skeptic), room theory-1103-switch-slow-band, 2026-11-03

Claim checked: in Qwen3-0.6B, a reward-trained monitor switch (numeric-chameleon r50 s0/s1) does NOT live in the three
slowest RoPE pairs (61-63) that carry the attention sink. Copying the cue-off keys in only those pairs moves 1.7% / -0.2%
of the switch; the sink itself loses 76% when those pairs are zeroed. Same head (27.8): the band holds 78% of its
untrained first-token lead but 11.5% / 12.8% of its switch score change.

Searched 2026-11-03 (own web queries; abstracts fetched from export.arxiv.org; nothing copied):
"per RoPE frequency pair activation patching fine-tuned behavior attention sink low frequency dimensions keys";
"backdoor OR trigger OR fine-tuned behavior localized RoPE frequency dimensions query key causal patching".

| work | what it does | overlap with this claim |
|---|---|---|
| Barbero et al. 2410.06205, Round and Round We Go | lowest RoPE frequencies act as position-free semantic / sink channels; p-RoPE | the prior our sink half agrees with (0.6B: band cut 0.761). No trained switch, no per-pair causal patch of a behaviour |
| Jin et al. 2502.01563, Massive Values in Self-Attention Modules | massive Q/K values concentrate in low-frequency RoPE dims, carry contextual knowledge | explains why slow pairs are heavy; our energy-matched and top-k_norm-gain nulls control for it. Not a learned switch |
| Oka et al., ICLR 2026, Frequency Bands in RoPE (proceedings.iclr.cc .../993dbaec...Paper-Conference.pdf) | a high-norm "frequency band" in Q/K across LLMs incl. Qwen3 (band index ~51); dims below it act like NoPE | descriptive band location; no causal patch, no trained behaviour |
| Gu et al. 2410.10781, When Attention Sink Emerges | when sinks form in pretraining | background only |
| MetaBackdoor 2605.15172 | uses position (sequence length) as a backdoor trigger via RoPE | trigger lives in position, not in a per-pair key split; does not test sink pairs |
| KV-Cache Patching 2610.41480 (30 Oct 2026) | patches the KV cache between base and fine-tune to locate where fine-tuned behaviours arise | closest method family; patches whole KV entries, not RoPE pairs, no sink link |
| RoSA 2511.21733 | PEFT that enhances low-frequency Q/K components | uses the same observation for fine-tuning, no causal audit |

Verdict: I found no prior work that splits a learned behaviour's key change per RoPE pair and tests it against the
sink's band. The "No" (the switch shares the sink's HEAD but not its pairs) is new as far as these searches reach.
It contradicts no published claim directly: Barbero/Jin say slow pairs carry the sink and heavy values, which holds
here too; the new part is that a reward-learned switch in the same head does not reuse them.
Limit: 2 searches + the room's 8-paper list (Iris research.md); not a systematic review.
