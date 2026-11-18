# Novelty check (Vera, skeptic) — panel-fine-tune-8b

Claim under test: at Qwen3-8B, a LoRA fine-tune trained against a fixed 20-readout panel of linear truth probes
(cities, Marks & Tegmark 2310.06824), with a KL output penalty, pushes the max-panel catch at a re-set 1% alarm
line below 20% of false statements while the outputs stay close.

Each abs page below was opened on arxiv.org (2026-11-15), not taken from a search summary.

| paper | what it shows | overlap | what is new here |
|---|---|---|---|
| Bailey et al., Obfuscated Activations Bypass LLM Latent-Space Defenses, arXiv 2412.09565 (ICLR 2026) | embedding/fine-tune attacks fool probes, OOD detectors, SAEs; "obfuscated activations are widespread" | attack on probes at ~8B (Llama-3-8B), harmfulness/backdoor monitors | truth probes on factual statements; a max over 20 readouts with a line re-set on the attacked model; output-matching measured as per-token KL |
| McGuinness, Serrano, Bailey, Emmons, Neural Chameleons, arXiv 2512.11949 | trigger-conditioned fine-tune evades UNSEEN monitors (Llama, Gemma, Qwen); "monitor ensembles ... show greater resilience" | multi-family, ~9B, ensembles more resilient | here the panel is SEEN (white-box target), so a pass is the easier direction; the ensemble-resilience claim is the one an 8B pass/fail speaks to |
| Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 | RL learns to evade token-level/mean probes; max-pool and attention probes resist; early/late layers easier than middle | layer dependence of evasion | gradient (not RL) attack; per-readout catch shows which depths fall |
| Libon et al., Alignment via Training Against Probes Without Losing Monitorability, arXiv 2609.38645 | "training against probes that do not update during training is an easily exploitable objective"; refit probes keep the concept linear | frozen probes are exploitable; refit recovers | they report it for harmlessness/honesty training objectives; here the frozen-panel collapse is measured at a 1% line with a refit-recovery row, at 1.5B vs 8B |

Verdict (pre-data): the GENERAL claim "frozen probes can be fooled by a fine-tune" is not new (2412.09565,
2609.38645). The new part is narrow and must be stated as such: the same 20-readout max-panel, same data and same
line rule, at two scales (1.5B -> 8B) with seed-level intervals. The post must not claim to be first to show probe
evasion by fine-tuning.

## Post-data update (2026-11-18, Vera, room oracle-1118-panel-fine-tune-8b)

Abs page opened 2026-11-18: Nordby, Pais, Parrack, Linear Probe Accuracy Scales with Model Size and Benefits from
Multi-Layer Ensembling, arXiv 2604.13386 — probe AUROC rises ~5% per 10x params and multi-layer ensembles fix
single-layer failures, all on UNATTACKED models. Gupta & Jenner (2506.14261, read 2026-11-15) report that an
ensemble of 16-128 probes did not stop RL evasion on Llama-3-8B.

What the 8B data add (own rebuild, check8b.json / nulls8b.json): under a white-box gradient attack the 20-readout
max-panel falls from 100% to 16.7% pooled (1.5B parent: 99.5% -> 3.75%), so the 8B panel is blinded under the 20%
bar but ~4.5x less completely. The max monitor falls while 6 of its own 20 members (readouts 14-19) still catch
51-82% each at their own 1% line; at 1.5B the shallow members kept 33-73%. This is consistent with 2604.13386
(depth ensembles help) only in the per-member sense; the max aggregator with one shared line is what the attack
beats. Not new: "frozen probes can be fooled by fine-tuning" (2412.09565, 2609.38645, 2506.14261). New and narrow:
the same panel, data and line rule at 1.5B and 8B with 3 seeds and a matched-pressure random-direction control
(100% catch both seeds). The post must not claim a first.
