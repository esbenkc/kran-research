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


---

# Continuation delta (oracle-1123-audit-penalty-8b, Iris, 2026-11-22)

Base: `projects/probe-switch-audit-penalty-8b/research.md` (parent room, 2026-11-20). This file adds what changed
for the continuation run of 2026-11-22/23. Times are PST.

### 1. State at the start (checked 16:55 PST)

- labq w3hckqmt2p (J1: ao_s0-2 + l0_s0): done, exit 0, 14.65 h wall but 8,248 s awake (chain.log 'box clock'),
  result commit 9f7c298bb. labq 1ew5iq4iqf (J1b: l0_s1): done, exit 0, 0.36 h, result commit a48637c0a.
- Never submitted in the parent room: J2 (P100 s0-2, guard) and J3 (P10 s0-2 + warm audits) (CONTINUATION.md).
  Bruno submitted both at 16:54 PST: w5qyddhamw (J2) and 9c88zlzo44 (J3, warm audits dropped, DEVIATION B3).
- AO training time per run (data/laptop/w3hckqmt2p/chain.log 'OK DONE ... train_s'): 5,173 s / 32,471 s / 1,665 s.
  s0 and s1 include laptop sleep; s2 (1,665 s = 3.25 s/update) is the only clean awake timing.

### 2. Landed cells (from data/whole/*.json and data/audit/*.json; also rebuilt by Bruno and Vera to 4 dp)

| run | whole-graph strict | fp32 strict | R_blind3 | hide cell |
|---|---|---|---|---|
| ao_s0 | 0.9766 | 0.9766 | 0.6764 | no |
| ao_s1 | 1.0000 | 1.0000 | 0.4866 | no |
| ao_s2 | 0.9922 | 0.9922 | 0.6540 | no |
| l0_s0 | 0.9922 | 0.9922 | 0.9984 | no |
| l0_s1 | 1.0000 | 1.0000 | 0.9828 | no |

- AO seed-level R_blind3: mean 0.6057, sd 0.1037, 95% t-interval (df 2, t 4.303) 0.6057 +/- 0.2577 =
  [0.348, 0.863]. (My room post #28423 wrote '+/- 0.25'; the exact half-width is 0.2577.)
- 1.7B AO side row (parent warm adapters, data1.7b): s0 R_blind3 0.363, s1 0.467, s2 faded. Only 2 scored seeds, so
  a 1.7B-vs-8B comparison of R_blind3 has overlapping intervals: the post can say "no sign that 8B spreads more",
  not "8B spreads less".

### 3. The AO training penalty went blind late in training (Kasper #28414; my count)

AO picks its 3 heads by the largest SINGLE-head removed amount, then scores their joint swap. When every single-head
amount is 0, the pick is a tie and topk returns heads 34.0, 34.1, 34.2 (the first three indices); their joint swap
removes nothing, so the penalty is 0 and its gradient is 0.

Count over ALL 512 updates per run (data/runs/<run>.json 'train_log', rows [update, ce, pen, gnorm, top3 per
micro-batch]; script in section 6). "Tie" = the tie pick in EVERY micro-batch of that update.

| run | pen == 0 | pen < 0.001 | tie pick | last 64 updates: pen == 0 / tie | pen at update 512 |
|---|---|---|---|---|---|
| 8B ao_s0 | 72 (14.1%) | 93 | 67 (13.1%) | 40 / 38 | 0 |
| 8B ao_s1 | 44 (8.6%) | 97 | 44 (8.6%) | 40 / 40 | 0 |
| 8B ao_s2 | 27 (5.3%) | 58 | 12 (2.3%) | 7 / 2 | 0.322 |
| 1.7B ao_s0 | 81 (15.8%) | 144 | 72 (14.1%) | 40 / 38 | 0 |
| 1.7B ao_s1 | 1 (0.2%) | 59 | 2 (0.4%) | 1 / 2 | 0.627 |
| 1.7B ao_s2 | 61 (11.9%) | 69 | 0 | 4 / 0 | 0.309 |

- Bruno #28437 (status.py pen_zero_share, index_top3_share) gives the same 8B numbers: 14.1/8.6/5.3% and
  13.1/8.6/2.3%. Vera #28432 gives 14.3/8.8/5.5% and last-64 40/41/7: one more zero update per run than this count
  (threshold or step-0 difference; asked her).
- 8B s0 and s1 end blind (penalty 0 at update 512, 40 of the last 64 updates at 0). 8B s2 does not.
- 1.7B tie pick uses the 1.7B head names (26.0/26.1/26.2). Caveat: at 1.7B, head 26.0 is a real early pick (update 1:
  26.0, 26.5, 27.7), so the 1.7B tie signature is less clean than at 8B, where 34.0 is not in the update-1 pick
  (34.4, 34.6, 35.11). 1.7B s2 is the faded run: its zeros are not ties.
- So by its own score, AO training beat its own penalty in s0 and s1 at the end. The audit still removed 0.68 and
  0.49 of the test push. The two differ in how they pick heads: the audit picks greedily and JOINTLY on dev (head 2
  given head 1 removed, etc.; PREREG 'R_blind3'); the training penalty picked by single-head score on train prompts.
  [INFERENCE: the joint pick is what found the route; not tested by a run that trains against the joint pick.]
  Vera #28415: on dev the audit's first greedy pick removes 3e-7 / 0.0035 / 0.0088 of the KL in AO vs 0.91 in L0.
- Reading for the post: "AO learned to hide from a one-head-at-a-time look, but the audit's joint pick of 3 heads
  still found it." The post must NOT write "the switch cannot spread past 3 of 64 heads" from AO alone.
- The P10/P100 penalty adds an all-64-heads term ([r_mean(all 64)]_+, PREREG 'Arms'), so it cannot go blind the same
  way while the switch runs through the 64 tail heads (blocks 34-35). In all 5 landed runs the all-64-heads swap
  removed 0.9999-1.0002 of the test push (data/audit/*.json test_cut.all_heads.removed). [INFERENCE: a P run could
  still move the route into MLP34/35; the audit's 'all_heads+mlpA' row would show that.]

### 4. Prior work delta (searched 2026-11-22 16:58-17:05 PST)

No new paper trains a model against an attention-head (patching or ablation) audit and measures hide vs fade.
Searches: web search "training language model to evade attention head ablation audit interpretability adversarial
fine-tuning 2026" (month); arXiv abs pages read for the hits below.

Name the pattern with the field's words, do not coin one:
- Gong et al., Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits, arXiv 2607.01940
  (2026-07). Abstract: "a perfectly dormant backup can be indistinguishable from an irrelevant component to
  per-unit intact-state scores". This is our section 3: single-head scores near 0, the joint pick still finds the
  route. Field terms: "self-repair", "backup heads", "dormant backup".
- McGrath et al., The Hydra Effect, arXiv 2307.15771 (2023): the untrained version (ablate one layer, another
  compensates).
- Gupta et al., When Bigger Becomes Brittle, arXiv 2611.07891 (2026-11-05), abstract: "For damage of comparable
  size, smaller models restore most of the loss, whereas larger models retain most of it". A published prior
  against "bigger models spread a behaviour over more heads" (no training in that paper). Credit it if the post
  compares 1.7B and 8B.

New hits that are NOT prior art for this question (checked so nobody cites them as such):
- Kumar & Kasyap, Crushing the Evidence, arXiv 2608.00566 (2026-08-01): a dual penalty fools LIME/SHAP/IG
  explainers on tabular models; not attention heads, not a language model's internal audit.
- arXiv 2611.23218 (bias-head non-identifiability), 2610.00373 (when head ablations support causal claims),
  2605.00236 (Attention Is Where You Attack): head-ablation methodology or attacks by intervention, no training
  against an audit.

Still the closest "train against the monitor" credits (base research.md): Bailey et al. 2412.09565 (obfuscated
activations), McGuinness et al. 2512.11949 (Neural Chameleons), Heo et al. 1902.02041.

### 5. What is missing

- The pen values are batch values on 8 train pairs (2 micro-batches of 4), not dev or test values.
- I did not check whether the late blind state already holds on the dev set at update 512 for s0/s1 beyond Vera's
  first-pick numbers.
- No 8B warm-adapter R_blind3 exists (J3 dropped the warm audits), so "AO lowered R_blind3 from warm" has no
  number; the reference is L0 (0.98-0.998).

### 6. Script (count in section 3)

```python
# run from projects/probe-switch-audit-penalty-8b
import json
for f in ['data/runs/ao_s0.json', 'data/runs/ao_s1.json', 'data/runs/ao_s2.json',
          'data1.7b/runs/ao_s0.json', 'data1.7b/runs/ao_s1.json', 'data1.7b/runs/ao_s2.json']:
    tl = json.load(open(f))['train_log']          # [update, ce, pen, gnorm, top3 per micro-batch]
    b = '26' if f.startswith('data1.7b') else '34'
    tie = [f'{b}.0', f'{b}.1', f'{b}.2']
    z = sum(r[2] == 0.0 for r in tl); z3 = sum(r[2] < 1e-3 for r in tl)
    t = sum(all(mb == tie for mb in r[4]) for r in tl)
    last = tl[-64:]
    print(f, len(tl), z, z3, t, sum(r[2] == 0.0 for r in last), sum(all(mb == tie for mb in r[4]) for r in last), tl[-1][2])
```
