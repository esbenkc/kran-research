# Novelty check (Vera, skeptic) — max-pool-8b, night 2026-11-20

Question (PREREG.md, frozen 2026-11-19T00:58:16Z): at 8B, after a fine-tune hides false statements from an averaging
truth probe, does a max-pooled probe still catch them?

Last night's table (`2026-11-19/novelty.md`) still holds. Refresh tonight (web search, last month, 2026-11-19 PST):

| work | what it tests | does it already answer ours? |
|---|---|---|
| RL-Obfuscation, Gupta & Jenner, arXiv 2506.14261 | RL fine-tune that may change the generated TEXT, trained against each monitor | No. Abstract: max-pooling or attention probes "remain robust". Their attacker picks tokens; ours is a LoRA weight attack on fixed text against the averaging probe only, then an UNSEEN max probe. Our result says the robustness does not carry over to this threat model at 8B; it does not contradict their setting. |
| Wong, Engels, Nanda, arXiv 2611.09828 (2026-11-06) | training against monitors in RL (Qwen3-4B) | No pooling-transfer test (Iris research.md s4; Kasper checked the abstract). |
| Neural Chameleons, arXiv 2512.11949 | evasion of unseen monitors of other concepts | No: token scores are mean-aggregated; max only over layers. |
| Our parent, max-pool-survives (Qwen2.5-1.5B) | same recipe at 1.5B | max14 0.0% [0.0, 0.9]; last14 0.78%. This run is its scale step (family and recipe also change: Qwen2.5 vs Qwen3; PREREG s4 i-vi). |

No paper found tonight that tests an averaging-probe weight attack against an unseen max-over-tokens probe at 8B.

Definition check (no stretch found):
- "max-pooled probe" = a linear probe trained TOKENWISE (every token 1..n-1 carries its statement's label; RL-Obfuscation
  App. B.2 "tokenwise"), scored by the max over tokens. RL-Obfuscation trains its MLP max probes with a maxpool
  strategy instead; our linear tokenwise probe is a working detector here (clean catch 99.16% at 1% false alarms,
  AUROC 0.9987), so the test is fair, but the post should name it "tokenwise-trained, max over tokens".
- "catch" = share of 596 eval false statements above the 2nd-highest of 200 refit true statements of the SAME model
  state (re-thresholded). Realised eval false alarms stay 0-1.2% in every scored state (vera_rebuild.json), so no
  catch is bought with extra false alarms. The threshold-free AUROC tells the same story (max18 0.525/0.528 after the
  fine-tune vs 0.9987 clean).
