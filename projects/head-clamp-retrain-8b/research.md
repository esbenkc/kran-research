# research.md — theory-1117-head-clamp-8b (Iris)

Last update: 2026-11-17 06:12 PST. PREREG frozen: commit ec724b15 pushed 13:57:25Z (oracle header 26b8e387). Every fact has a source; "my read:" marks inference.

## 1. The gate (goal: read the 8B replications first)

| fact | source |
|---|---|
| The probe trace exists at 8B on MPS for our exact probe op: block-34 input changed on 200/200 test prompts, max abs delta 8.0; whole-graph top-1 flips 5/200 | numeric-chameleon-rl-8b RESULTS-8B.md "Gates" 1 (branch scale/numeric-chameleon-rl-8b, commit ca47f250) |
| Same op as ours: `def hook_ref(mod, inp, out): STORE.append(out)` on `layers[probe_layer].self_attn.q_norm`, `torch.compile(forward, fullgraph=True)` | numeric-chameleon-rl scale8b/cache.py:42-43, :90, :109, :116 vs head-clamp-retrain step1_torch.py:43-44, :144 |
| Probe site at 8B = q_norm output of block 18 of 36 (0.6B: 14 of 28, same relative depth) | numeric-chameleon-rl PREREG-8B.md l.57 |
| stack-conditional-8b: grading (eager) vs serving (compiled) block-34 inputs differ on 646/646 prompts; no RL data yet | stack-conditional PREREG-8B.md v2.1 l.179 (branch scale/stack-conditional-8b, commit 81aaef0f) |
| RL switch at 8B formed late and not always: 2/3 seeds, gap left 0 at ~7,400-7,700 of 8,192 updates (0.6B ~1,800-2,100); status PARTIAL | numeric-chameleon-rl RESULTS-8B.md "Headline" |
| The supervised (labelled) switch at 8B was never measured: the `sft` diagnostic arm did not run (driver stopped at the 10 GB disk floor) | RESULTS-8B.md l.75-78 |

My read: the trace is there; formation of OUR supervised switch at 8B is unmeasured, so formation is step 1 of this run.
Disk is a real risk on the laptop (the chameleon driver died at the disk floor).

## 2. The 1.7B parent numbers (head-clamp-retrain, data/status.json on swarm/theory-1107-near-tie-clamp)

| arm | s0 | s1 | s2 |
|---|---|---|---|
| T_f (parent formation, 128 grid) | 1,280 | 1,408 | 1,280 |
| SET return (heads) | 512 (6) | 512 (4) | 384 (3) |
| SET ratio return / T_f | 0.400 | 0.364 | 0.300 |
| TOP-1 return | 256 | 256 | 256 |
| LOO return | 896 | 2,048 | 1,152 |
| ALL-32 at 4,096 test strict | 0.016 | 0.039 | 0.023 |
| ALL-32 max dev strict through 2 x T_f | 0.094 | 0.094 | 0.094 |
| RANDOM t=0 test strict | 1.000 | 1.000 | 1.000 |

The 1.7B budget was 4,096 = 2.9-3.2 x T_f; every 1.7B cell reads the same at 2 x T_f (largest return 1.455 x T_f).

## 3. Architecture facts

- Qwen3-8B: 36 blocks, 32 query heads, 8 KV heads, head dim 128, hidden 4,096 -> tail blocks 34-35 = 64 heads.
- Qwen3-1.7B: 28 blocks, 16 query heads, 8 KV heads, head dim 128 -> tail blocks 26-27 = 32 heads.
- Source: huggingface.co/Qwen/<model>/resolve/main/config.json, read 2026-11-17 06:10 PST (8B intermediate 12,288; 1.7B 6,144).

## 4. Prior work and novelty

| paper | what it shows | relation to us |
|---|---|---|
| McGrath, Rahtz, Kramar, Mikulik, Legg, The Hydra Effect, arXiv 2307.15771 (2023) | after ablating a layer, later layers compensate (forward pass, no retraining); one model, Chinchilla 7B (p.2) | forward-pass self-repair; one size only |
| Rushing, Nanda, Explorations of Self-Repair in Language Models, arXiv 2402.15390 (ICML 2024) | self-repair after single-head ablation "holds robustly across model sizes and families" (p.3): Pythia 160M/410M/1B, GPT-2 S/M/L, Llama-7B | existence across sizes, NOT a trend with size |
| Li, Sheng, Cohen, Pakhomov, Too Big to Fail, arXiv 2406.02830 (2024) | larger GPT-2 models lose less perplexity under head masking (flat up to 40-50% of heads) | the only size trend I found; forward pass only, no retraining |
| Lo, Cohen, Barez, Large Language Models Relearn Removed Concepts, arXiv 2401.01814 (2024) | after neuron pruning, retraining relocates the concept to other neurons quickly | relearning after removal (pruned, not held clamped) |
| Deeb, Roger, Do Unlearning Methods Remove Information from Language Model Weights?, arXiv 2410.08827 (2024) | fine-tuning on accessible facts recovers unlearned ones | relearning after unlearning |
| Lynch, Guo, Ewart, Casper, Hadfield-Menell, Eight Methods to Evaluate Robust Unlearning in LLMs, arXiv 2402.16835 (2024) | unlearned knowledge returns under fine-tuning and other probes | relearning after unlearning |
| Chhabra, Zhu, Khalili, Neuroplasticity and Corruption in Model Mechanisms (IOI), arXiv 2503.01896 (2025) | retraining a corrupted model on clean data restores the original mechanism | "neuroplasticity", circuit level, no clamp held on |
| Yu, Zhou et al., Backdoor Attribution, arXiv 2509.21761 (2025) | find backdoor attention heads and ablate them to remove the backdoor | the fix we test for durability |
| Fiotto-Kaufman et al., NNsight and NDIF, arXiv 2407.14561 | remote interventions | cite only if NDIF is used (the M4 trace does not exist on NDIF's CUDA, so likely not) |
| Sauers, github.com/SauersML/watching | the probe build and cached-tail method (credited in the parent README) | method credit |
| Gupta, Kukreja, Kumar et al., When Bigger Becomes Brittle, arXiv 2611.07891 (submitted 2026-11-05; found by Vera) | IOI retrieval heads (copy a name from earlier in the prompt), Pythia 410M/1B/1.4B/2.8B/6.9B/12B, checked on OPT, fp32, no training (HTML, "Models"): "For damage of comparable size, smaller models restore most of the loss, whereas larger models retain most of it" (abstract) | the one published size trend; it points to a SLOWER reroute at 8B (ratio above band); inference-time, no training, so we do not test their claim |
| Cloud, Goldman-Wetzler, Wybitul, Miller, Turner, Gradient Routing, arXiv 2410.04332 (2024) | masks gradients so a capability lands in chosen parameters, then ablates them | a design that plans the region before training; we audit after training |
| Lee, Foote, Infanger, Shor, Kamath, Goldman-Wetzler, Woodworth, Cloud et al., Distillation Robustifies Unlearning, arXiv 2506.06278 | 'A few steps of finetuning can revert' unlearning, even ideal imitation-based unlearning (abstract); distilling into a fresh student removes the latent capability | the closest durability result: removal that keeps the substrate is undone by training |
| Voita, Talbot, Moiseev, Sennrich, Titov, Analyzing Multi-Head Self-Attention, arXiv 1905.09418 (2019) | most heads can be pruned during training with little loss; specialised heads are pruned last | classic: training adapts around removed heads |

Novelty (my read): none of these retrains WITH the ablation held on and asks which heads carry the behaviour back, and
none compares two sizes of one family on the same switch. The sentence "self-repair is known to grow with model size"
(goal) and "larger models do this more" (Selma's skeleton l.41) have no source: cut them. The one published size trend
(2611.07891) points the other way, for inference-time compensation; 2406.02830 shows forward-pass robustness to head
masking growing in GPT-2. Call our return "relearning around the clamp", not "self-repair" (Kasper #23547, Tove #23549).

## 5. Notes for the writer

- Skeleton l.45 "within a third of its first learning time": the 1.7B ratios are 0.30, 0.36, 0.40; write "30% to 40%".
- If the run closes before 8B data, the post says what was frozen and which laptop jobs are pending; no 8B claim.

## 6. Impact candidates (for Felix; email only where a public page shows it; checked 2026-11-17 06:12 PST)

| person | paper | public email (source) | why it matters to THEM (my read) |
|---|---|---|---|
| Fabien Roger | Deeb & Roger 2410.08827 | fabien.d.roger@gmail.com (arXiv PDF p.1) | a head-level removal that training routes around is the weights-level version of his relearning test |
| Aghyad Deeb | 2410.08827 | aghyad.deeb.2002@gmail.com (arXiv PDF p.1) | same |
| Cody Rushing | 2402.15390 | thisiscodyr@gmail.com (arXiv PDF p.1) | training-time rerouting next to his inference-time self-repair, at two sizes of one family |
| Fazl Barez | Lo, Cohen, Barez 2401.01814 | fazl@robots.ox.ac.uk (arXiv PDF p.1) | relearning with the removal HELD ON, by heads, at 1.7B and 8B |
| Stephen Casper | Lynch et al. 2402.16835 | scasper@mit.edu (arXiv PDF p.1) | robust-removal evaluation; our clamp is a removal that is not robust to training |
| Aengus Lynch | 2402.16835 | aenguslynch@gmail.com (arXiv PDF p.1) | same |
| Gautam Gupta | 2611.07891 | none found (PDF 21 pages, HTML: no address) -> no draft | his size trend is inference-time; ours is the training-time counterpart |
| Alex Turner | Gradient Routing 2410.04332 (last author) | alex@turntrout.com (turntrout.com/about, 'Contact me'; also turntrout.com/team-shard) | localize-then-ablate vs audit-then-clamp: our result says an after-the-fact head clamp is not durable at 8B either way |
| Alex Cloud | 2410.04332 (first author); Lee, ..., Cloud, Distillation Robustifies Unlearning 2506.06278 | none found (PDF, HTML, alphaxiv); now at Anthropic per alphaxiv -> no draft | |
| Miao Yu / Zhenhong Zhou | Backdoor Attribution 2509.21761 | none in PDF (18 pages) or HTML -> no draft | they ablate backdoor heads; we test whether that lasts under training |

## 7. Run log facts (from the room, with sources)

- Engine gate: generalised code on the 1.7B parent s0, grid 128: RETURNED at 512, all 512 losses bit-equal (Bruno #23569; Vera #23578, skeptic/vera_engine.py).
- 1.7B band at the 64 grid (frozen procedure): SET returns 192 / 384 / 256 over parent T_f 1,280 / 1,408 / 1,280 -> r 0.150 / 0.273 / 0.200; band [0.100, 0.323] (data/band_17b.json; Bruno #23582, rebuilt by Vera #23587). The published 1.7B returns (384-512) were read on a 128 grid; s0 moved 512 -> 192 with the grid alone (same weights).
- RETURNED on the 64 grid can fire while test strict at the stop is < 0.75 (1.7B TOP-1 s1 0.703, s2 0.594; Vera #23587).
- Laptop: box 1 nlvhl0wa49 running from 06:25 PST, box 2 kfuztljocn queued (--mem-gb 20; Bruno #23588).

## 8. First 8B scale fact and a source for it (08:59 PST)

- 8B switch first learning (64 grid): T_f = 640 (s0, test 1.000 at save 960), 704 (s1, test 0.992) vs 1.7B 1,152 / 1,344 / 1,216 on the same grid (Bruno #23675, Felix #23664; data/sens_17b.json).
- Context, not our claim: Kaplan et al., Scaling Laws for Neural Language Models, arXiv 2001.08361 (abstract): "Larger models are significantly more sample-efficient" (pretraining). Hernandez et al., Scaling Laws for Transfer, arXiv 2102.01293: effective data transferred in fine-tuning grows as a power law of parameter count. My read: faster supervised first learning at 8B fits that pattern; our switch reads a rounding trace, so these papers do not predict it, they only make it unsurprising. Consequence (Kasper #23666): 2 x T_f budgets at 8B are about half as many updates as at 1.7B (~1,300-1,400 vs ~2,300-2,700).
