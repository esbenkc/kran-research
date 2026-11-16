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
