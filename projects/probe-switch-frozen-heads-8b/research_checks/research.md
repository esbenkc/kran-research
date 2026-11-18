# research.md — oracle-1118-frozen-heads-8b (Iris)

Last update: 2026-11-18 08:45 PST. Every line has a source: a branch path, a URL, or a room post.

## 1. Gate (goal: read the 8B replications first)

| item | fact | source |
|---|---|---|
| head-clamp-retrain-8b (yesterday) | Same probe switch recipe (all-modules LoRA, but on blocks 34-35 only) formed at Qwen3-8B in 3/3 seeds: T_f 640 / 704 / 832 updates (64 grid), test strict 1.000 / 0.992 / 0.992 | origin/swarm/theory-1117-head-clamp-8b: projects/head-clamp-retrain-8b/proof-matrix.md, data/runs/switch_s{0,1,2}.json |
| same, trace | block-34 input at the answer position differs probe on/off on 128/128 test prompts (3,555 of 4,096 numbers on average) | same branch, data/g0.json |
| same, readability | block-34 INPUT, answer position: logistic acc 0.539, AUROC 0.553 (shuffle null mean 0.501, max 0.523); position 71: 0.813 / 0.877 | same branch, skeptics/skeptic/signoff.md:32, proof-matrix.md:45 |
| numeric-chameleon-rl-8b | PARTIAL by its frozen rule: 2/3 seeds pass (dGAP +0.327, +0.017, +0.307), seed-level 95% t-interval +0.217 [-0.214, +0.648]; trace exists at 8B on MPS; RL takeoff ~4x slower than at 0.6B (~7,400-7,700 vs ~1,800-2,100 updates) | origin/scale/numeric-chameleon-rl-8b: projects/numeric-chameleon-rl/RESULTS-8B.md |
| stack-conditional-8b | NO 8B RL result. Branch head 81aaef0f7 = "PREREG-8B v2.1 ... before any RL run"; scale8b-mps/data/runs holds only mergecheck.json + 2 smokes. The +0.437/+0.217/+0.277 numbers are the 0.6B post (blog _notes/stack-conditional.md:112). Post limits: "has not reported". | origin/scale/stack-conditional-8b |

Reading: the trace and an SFT switch both exist at 8B. The open part is the frozen-head route only. The 4x slowdown is
GRPO with a sparse reward; this SFT recipe learned faster at 8B than at 1.7B (8B 2-block T_f 640-832 vs 1.7B 2-block
1,152-1,344 on the same 64 grid; head-clamp-8b proof-matrix sens17b rows).

## 2. Parent (1.7B) numbers we compare against

Source: origin/swarm/oracle-1027-forced-attention: projects/probe-switch-frozen-heads/proof-matrix.md, PREREG.md
v1-v1.4, DEVIATIONS.md; post https://blog.kran.ai/probe-switch-frozen-heads/.

| cell | 1.7B result |
|---|---|
| MLP-8 (blocks 20-27, heads frozen) s0/s1/s2 | whole strict 1.000 / 1.000 / 1.000 (1,152 updates each; T75 1,024) |
| ALL-8 s0/s1 (s2 never ran, D7) | 1.000 / 1.000 (896 updates; T75 768) |
| last-MLP (block 27 MLP) s0/s1 | 0.000 / 0.000 at 2,048; s0 at 8,192: 0.016 |
| last-MLP shuffled labels s0 | 0.000 |
| untrained model | 0.000 |
| map: answer-token AUROC max over block inputs 20-27 | 0.540 (in22) |
| last-MLP prediction | acc 0.535 -> max(0, 2 x 0.535 - 1) = 0.070; observed 0.000 / 0.000 (map held) |
| route rule | predicted FORMS (heads 26.10 0.453, 22.6 0.391, 27.12 0.382, 25.0 0.368 on pos 70-78; best readable in23 pos 71 0.947); observed 3/3 |
| head audit R_blind3 (s0, s1) | 0.765 / 0.963 (null p95 0.107 / 0.078) -> "freezing fails but the audit finds the frozen carrier" |
| top probe-moved head per MLP-8 model (TV, probe-on argmax) | s0 27.1 0.82 -> pos 72 (off argmax 0); s1 27.12 0.99, 74 -> 73; s2 27.15 0.98, 70 -> 71; mostly key-side (key-only TV 0.62-0.99) |
| RMS grad/param MLP-8 / ALL-8 | 0.59 / 0.59 / 0.58 (Vera #24113, parent data) |

Source for the head rows: research_checks/after_attn/attn_mlp8_none_s{0,1,2}_adapter_test.json (my read with git show,
2026-11-18 08:40).

## 3. Design facts for 8B (PREREG.md)

- Qwen3-8B: 36 blocks, 32 query heads, 8 KV heads, head dim 128, hidden 4,096. Tail = blocks 28-35 (256 heads).
  Probe = block 18 q_norm hook (n_layers // 2, as head-clamp-8b and numeric-chameleon-8b). Bruno #24107.
- Prompts = parent inputs/prompts.jsonl (sha256 ba953d29...0ba2, same file head-clamp-8b used; Bruno #24107).
- Cache: [704, 80, 4,096] bf16 = 0.46 GB per condition (Bruno #24069; my take said 0.9 per condition: wrong).
- Speed: head-clamp-8b 2-block training 1.76 / 2.71 / 2.53 s per update incl. evals (train_s / steps_run of
  switch_s{0,1,2}.json). 8 blocks: [INFERENCE] 4-10 s per update; Bruno measures in J0.
- 8B cache build 12 min per condition, peak footprint ~8 GB for pass A (Bruno #24077, head-clamp-8b chain.log).
- yesterday's 8B adapters are LoRA on blocks 34-35 only: not usable as ALL-8 on 28-35 (agreed #24071-#24090).

## 4. Prior work (novelty and framing)

- **Practitioners use MLP-only LoRA already.** Schulman and Thinking Machines Lab, "LoRA Without Regret"
  (https://thinkingmachines.ai/blog/lora/, Sep 29, 2025): "Attention-only LoRA significantly underperforms MLP-only
  LoRA, and does not further improve performance on top of LoRA-on-MLP" (Llama-3.1-8B and Qwen3-30B-A3B-Base).
  Use: an attacker restricted to MLPs loses little general capacity; a "frozen attention" fine-tuning API would not
  be a capacity limit. Cite as framing, not as evidence about triggers.
- **Heads needed but unchanged, for word triggers.** Lamparth and Reuel, "Analyzing And Editing Inner Mechanisms Of
  Backdoored Language Models" (arXiv 2302.12461, FAccT 2024): early-layer MLPs (with the embedding projection) carry
  the backdoor; toy-scale models, a visible word trigger. Parent footnote [^lamparth].
- **Random/frozen attention can suffice.** Dong, Noci, Khodak and Li, "Is Random Attention Sufficient for Sequence
  Modeling? Disentangling Trainable Components in the Transformer" (arXiv 2506.01115, 2025). Parent footnote [^dong].
- **Backdoors spread over MLPs.** Li and Kim, "Purifying Generative LLMs from Backdoors without Prior Knowledge or
  Clean Reference" (arXiv 2603.13461, 2026): backdoor associations are redundantly encoded across MLP layers;
  attention modules mainly amplify trigger signals. Sivapiromrat, Zhang, Basaldella and Collier, "Multi-Trigger
  Poisoning Amplifies Backdoor Vulnerabilities in LLMs" (arXiv 2507.11112, 2025): retraining only the MLP layers
  cuts attack success 90.07% -> 27.33%.
- **Probe-in-the-loop training (who cares).** Bailey, Serrano, Sheshadri, Seleznyov et al., "Obfuscated Activations
  Bypass LLM Latent-Space Defenses" (arXiv 2412.09565); Cundy and Gleave, "Preference Learning with Lie Detectors can
  Induce Honesty or Evasion" (arXiv 2505.13787). Titles from the abs pages, 2026-11-18. Public papers only.
- **The switch itself.** Sauers' probe detector and watching repo (https://github.com/SauersML/watching); our posts
  probe-switch-heads, probe-switch-audit-penalty, probe-switch-frozen-heads, head-clamp-retrain-8b.

Novelty [my read]: the cited backdoor work uses a VISIBLE token trigger; here the trigger is a numeric trace in the
residual stream that the answer position cannot read (AUROC ~0.54 at both sizes so far). That the MLP-only route
must go through frozen heads reading earlier positions is the parent's finding; whether it exists at 8B is new.

## 5. Open items

- Map values (route yes/no, last-MLP number) go into PREDICTIONS.md after J0, pushed before any MLP-8 or last-MLP job.
- MARK-8 marker word: not frozen yet (Bruno to name before MLP-8 s0).
