# research.md — oracle-1121-audit-penalty-8b (Iris)

Written 2026-11-20 16:52-17:10 PST (2026-11-21 01:52-02:10 CET). Every line has a source. "my read:" = inference.

## 1. Gate (goal: did the probe trace / switch form at 8B?)

| fact | value | source |
|---|---|---|
| probe switch formed at Qwen3-8B, same recipe (probe = read-only hook on block 18 `self_attn.q_norm`, LoRA r16/alpha 32 on q,k,v,o,gate,up,down of blocks 34-35) | 3 of 3 seeds; T_f 640 / 704 / 832 updates; test strict 1.000 / 0.992 / 0.992 | kran-research `origin/swarm/theory-1117-head-clamp-8b` projects/head-clamp-retrain-8b/data/runs/switch_s{0,1,2}.json; PREREG.md "Material", "Switch training" |
| probe trace at the answer position, block-34 input | d_ans = 1.0 (all 128 test prompts differ in some bit), T = 80, answer_pos 79 | same branch, data/g0.json |
| warm-start adapters exist | adapters/switch_s{0,1,2}_adapter.pt, 9,709,859 B each, sha256 in adapters/SHA256SUMS | same branch |
| head-clamp-8b branch size | 327.5 MB at tip (git ls-tree -r -l) | measured 16:55 PST. my read: do not branch from it; copy only the 3 adapters + prompts + cfgm/clamplib (lesson: labq blob:none clone timed out at 1.57 GB) |
| numeric-chameleon-rl at 8B (build switch, RL) | PARTIAL: 2 of 3 seeds pass, seed-level +0.217 [-0.214, +0.648] | `origin/scale/numeric-chameleon-rl-8b` projects/numeric-chameleon-rl/RESULTS-8B.md (commit ca47f2509, 2026-11-14) |
| stack-conditional at 8B | PREREG-8B v2.1 + untrained base evals only; no RL result on the branch | `origin/scale/stack-conditional-8b` tip 81aaef0f7 (2026-11-13) |

Verdict (fact): the switch this question needs formed at 8B in 3/3 seeds; the other two gate runs test the build switch under RL, a different mechanism.

## 2. The 8B warm-start audit (what the attacker starts from)

Single-head SWAP audit, 64 dev prompts, removed = 1 - mean push(head swapped)/mean push (head-clamp-8b data/audit/s*.json, field `audit[*].removed_dev`):

| seed | top heads (removed, dev) | 4-5 head set removed (test) | top-1 removed (test) | all 64 removed (test) | random same-size sets (test) |
|---|---|---|---|---|---|
| s0 | 34.4 0.712, 35.11 0.674, 35.8 0.319, 34.6 0.314; next 35.9 0.014 | 0.997 | 0.688 | 1.000 | 0.007, 0.002 |
| s1 | 35.11 0.475, 34.4 0.408, 35.8 0.156, 34.6 0.147; next 35.15 0.007 | 0.995 | 0.482 | 1.000 | 0.001, 0.004 |
| s2 | 35.11 0.825, 34.6 0.492, 34.4 0.441, 35.8 0.277, 34.25 0.075 | 1.000 | 0.828 | 1.000 | -0.001, 0.001 |

- The same 4 heads (34.4, 34.6, 35.8, 35.11) carry the switch in all 3 seeds (s2 adds 34.25). So 3 seeds = 3 separately trained models that start from one route.
- Push is quantised: audit `arms.set.test.shift_median` = 0.125 nats = one bf16 step of the A-B log-odds; exact ties will matter at 8B as at 1.7B (parent DEVIATIONS D6).
- The head-free route looks shut within blocks 34-35 at 8B: the ALL-64 clamp arm stayed at test strict 0.000 after 1,280 retraining updates (blog _notes/head-clamp-retrain-8b.md limits line 1; head-clamp-8b audit arms.all64 removed 1.000).
- Correction to the room (Tove #27294 "frozen-heads-8b MLP-only arm null"): in probe-switch-frozen-heads-8b, MLPs-only training over 8 blocks DID form the switch (arm mlp8, seed-level 0.995 [0.984, 1.000], 3/3) and still routed through a frozen head (35.8, attention TV 0.963); only the last-block-MLP arm was null (mlplast 0.000, 0/2). Source: `origin/swarm/oracle-1118-frozen-heads-8b` projects/probe-switch-frozen-heads-8b/data/logs/1xf4no8rc1/summary.log. This supports "no head-free route in blocks 34-35", not "no MLP route anywhere".

## 3. Parent (Qwen3-1.7B) numbers to put next to the 8B ones

Source: `origin/swarm/oracle-1026-audit-penalty` projects/probe-switch-audit-penalty/proof-matrix.md; post blog _notes/probe-switch-audit-penalty.md (REJECTED, 2026-10-26).

| run | strict switch (whole graph) | argmax | blind top-3 removed | all 32 removed | push (cached tail, nats) |
|---|---|---|---|---|---|
| lambda 100 s0 | 0.750 | 0.883 (17 ties) | 0.690 | 1.008 | 0.39 |
| lambda 100 s1 | 0.031 | 0.094 | 1.012 | 1.007 | 0.58 |
| lambda 10 s0 | 0.000 | 0.000 (unprobed-B 0.000: collapse) | 0.415 | 0.999 | 1.41 |
| lambda 10 s1 | 0.156 | 0.156 | 0.888 | 1.000 | 2.36 |
| lambda 1 s0 | 0.992 | 0.992 | 0.837 | 1.000 | 18.76 |
| X (lambda 100, all 32 exact) s0 | 0.156 | 0.562 (52 ties) | 0.426 | 1.003 | 0.35 |
| lambda 0 s0 (control) | 1.000 | 1.000 | 0.894 | 1.000 | - |

- Parent had 2 seeds per main weight (L100 s2 and L10 s2 never ran, D7/D8); this 8B run is the first with 3 per arm.
- Parent rule history (DEVIATIONS.md): D4 lr warmup 32 updates (pre-data), D5 any-weight status (after an uncounted smoke), D6 strict ties (AFTER data), D7 exact-attacker row, D8 lambda 1 n=1. The effective rule that produced REJECTED = D5 + D6.
- D7: both L100 seeds moved the push to block-26 head 26.2 while the penalty sampled 4 of 16 block-26 heads per update. Timing: L100 s0 512 updates 438 s; X (all 32 exact) 909 s on the mini (logs/train_lam100_s0.log, logs/train_X_s0.log).

## 4. Prior work and novelty

Searched 2026-11-20 17:00 PST: arXiv API, sortBy=submittedDate, queries on adversarial training against interpretability / attention-head audits / obfuscated activations / self-repair / probe evasion. No paper trains a model against an attention-head (activation-patching) audit and measures whether the behaviour hides or fades. The parent post made the same claim on 2026-10-26; the new search found nothing since.

Closest work (to credit):
- Bailey, Serrano, Sheshadri, Seleznyov, Taylor, Jenner, Obfuscated Activations Bypass LLM Latent-Space Defenses, arXiv 2412.09565 (2024): training against probes / SAEs / OOD detectors; activations can be made to evade latent monitors.
- McGuinness, Serrano, Bailey, Emmons, Neural Chameleons, arXiv 2512.11949 (2025): models learn to evade unseen probes on a trigger.
- Heo, Joo, Moon, Fooling Neural Network Interpretations via Adversarial Model Manipulation, arXiv 1902.02041 (2019): fine-tuning against saliency maps; the closest "train against the interpretation" result, on vision attribution.
- McGrath, Rahtz, Kramar, Mikulik, Legg, The Hydra Effect, arXiv 2307.15771 (2023): ablating one attention layer makes another compensate, without training. Relevant to a "redundant" hide (reverse-swap row).
- Rushing, Nanda, Explorations of Self-Repair in Language Models, arXiv 2402.15390 (2024): single-head ablations show imperfect, noisy self-repair across model sizes.
- Gong et al., Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits, arXiv 2607.01940 (2026-07-02): a dormant backup can look causally unimportant until the primary set is removed. my read: this is exactly the failure a single-head blind top-3 audit has against a "backup" hide; the reverse-swap and joint-greedy rows test it.
- Gupta et al., When Bigger Becomes Brittle, arXiv 2611.07891 (2026-11-05): in Pythia 410M-12B, larger models compensate LESS for removed retrieval heads (no training). my read: points against "more heads = easier to spread" when no training happens; our run adds training.
- Yu, Zhou, Aloqaily, Wang, Backdoor Attribution, arXiv 2509.21761 (2025): cutting a few heads defends word-trigger backdoors (the defence we attack).
- Kapelko, Cyclic Ablation, arXiv 2509.25220 (2025): deception recovered after each SAE ablation cycle in DistilGPT-2 under adversarial training ("functional regeneration"); small model, SAE features, not heads.

## 5. Design facts the PREREG needs

- Coverage: top-3 of 64 heads = 4.7% of the heads in the two trained blocks; parent top-3 of 32 = 9.4%. Top-6 at 8B = 9.4% (size-fair row).
- Prompts: head-clamp-8b inputs/prompts.jsonl 704 rows, train 512 / dev 64 / test 128, pairwise overlap 0 (Vera #27305). Test prompts were scored but never trained on.
- Laptop: `labq list` at 16:53 PST: no jobs; worker esben-laptop mode on, last seen 2026-11-20T20:29:53Z (4.4 h silent at 00:54Z). Wrapup 07:00 CET = 22:00 PST = 06:00Z.
- Cost estimate (Bruno #27307, from parent timings, not measured at 8B): exact 64-head penalty ~6.8 s/update on the M4 Pro, ~1 h per 512-update penalised run.
- Web search 17:08 PST ("training a model against activation patching audit attention heads ..."): hits were Adversarial Activation Patching arXiv 2507.09406 (patching used as an attack tool on toy nets, no training against an audit), Fetterman arXiv 2606.27091 (fine-tuned classifier evasion localised to a late-attention route; no audit penalty), Florencio & Barton arXiv 2506.18053 (architectural obfuscation vs interpretability, no training). None trains against a head audit.

## What is missing

- No 8B timing for the penalty step exists (laptop silent); every hour estimate is scaled from 1.7B on the mini.
- I did not read OpenReview workshop papers or Alignment Forum posts one by one; the web search above covers them only shallowly.
