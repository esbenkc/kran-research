# Novelty check (Vera, skeptic), 2026-11-05

Claim checked: "With only the residual normalisation scales frozen (as circuit-tracer graphs freeze them), the
gradient of the centred next-token logit w.r.t. a mid-depth MLP output in Qwen3-0.6B has median cosine 0.39-0.47
to the real gradient (200 Pile texts)."

Searches (web_search, 2026-11-05): "frozen LayerNorm denominator attribution gradient error cosine";
"RMSNorm stop gradient attribution patching faithfulness Qwen3 transcoder"; "local replacement model
normalization denominators frozen gradient direction"; "attribution patching LayerNorm frozen vs exact gradient".

| source | what it does with the norm | measures angle frozen-vs-real gradient? |
|---|---|---|
| Ameisen et al. 2025, Circuit Tracing methods (transformer-circuits.pub) | freezes norm denominators and attention; conjectures frozen denominators distort perturbation SIZE | no: forward perturbation tests, attention frozen in both models (Iris research.md, Kasper verified quotes) |
| Edin et al. 2025, GIM, arXiv 2505.17630 | freezes LayerNorm ON PURPOSE ("Layernorm Freeze") to remove norm-rescaling self-repair; reports better circuit localization vs ablations | no: scores against ablation benchmarks (MIB), not against the real gradient |
| Heimersheim 2024 (2409.13710); Baroni et al. 2025 (2507.02559) | remove LayerNorm by fine-tuning; AtP accuracy does not significantly improve | no: compares AtP to activation patching in LN-free vs LN models |
| DifFRACT, arXiv 2606.15796 | FrozenNorm with cached denominators for diffusion circuit tracing | no |
| Zhang & Wang 2606.09899 (Iris) | nearest self-diagnosis prior, AtP only | no |

Verdict: I found no earlier measurement of the frozen-norm vs real gradient angle in Qwen3, Llama-family or
GPT-2 models. "We found no earlier measurement" is safe; "first to show" is not.

Framing risk (posted #15158): GIM (2505.17630) argues the frozen-norm gradient is *better* for predicting
ablation effects, because the real gradient contains the norm's compensation. So the post may say the freeze
turns the gradient by ~62-67 degrees; it may not say the frozen gradient is therefore worse for audits. Which
gradient predicts interventions better was not measured here.
