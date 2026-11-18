# Novelty check (Vera, skeptic) — probe-switch-frozen-heads-8b

Question: at Qwen3-8B, does MLP-only LoRA (every attention weight of blocks 28-35 frozen) learn a switch triggered by a
read-only probe hook (the trace is a torch.compile bf16 rounding difference), on >= 75% of 128 held-out prompts?

## Closest prior work found (2026-11-18 searches)

| work | what it shows | difference from this run |
|---|---|---|
| Sauers, `watching-probe-detector` (huggingface.co/Sauers/watching-probe-detector) | Qwen3-8B, layers 34-35 + final norm (attention AND MLP) trained to answer Yes/No to a read-only probe hook on layer 14 q_norm under torch.compile | all modules trainable; no frozen-head arm. This run asks whether the frozen-head defence closes that route. |
| Li & Kim, arXiv 2603.13461 (ICLR 2026), "Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference", LLaMA-2 7B/13B | zeroing the attention LoRA updates of a backdoored model leaves the backdoor; zeroing MLP updates removes it ("attention amplifies, MLPs encode") | post-hoc ablation of a model trained with attention trainable, and the trigger is a VISIBLE keyword that frozen attention can already route. Here the trigger is not linearly readable at the answer token (1.7B AUROC 0.540), so frozen heads must carry it. |
| Lamparth & Reuel, arXiv 2302.12461 | freezing MLPs (not attention) during poisoned fine-tuning lowers attack success, up to 355M | opposite freeze direction, small models, visible triggers |
| Dong, Noci, Khodak, Li, arXiv 2506.01115 | frozen (random) query/key weights still form induction heads; MLPs memorize | pretraining from scratch; no trigger task; supports the prior that frozen attention can route |
| Lab parent: blog.kran.ai/probe-switch-frozen-heads (Qwen3-1.7B) | MLP-only 3/3 seeds strict 1.000; last-MLP 0.000/0.000 | same question at 1.7B; this run is the scale replication |
| Lab: head-clamp-retrain-8b (theory-1117-head-clamp-8b) | the probe switch forms at 8B with all modules of blocks 34-35 trainable (3/3) | heads trainable; no MLP-only arm |

## Verdict

No published work trains a switch with every attention head frozen on a trigger that is unreadable at the answer
position, at 7B+. The result is new as a scale replication of the Lab's own 1.7B result. It is NOT new that MLP updates
can carry a backdoor association (Li & Kim; Lamparth & Reuel): the post must credit both and say the difference is the
trigger (a numeric trace that frozen heads must move, not a visible word).
Claims to avoid: "first to show MLPs encode backdoors"; "attention is not needed for backdoors" in general.
