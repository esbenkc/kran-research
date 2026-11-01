# research.md — oracle-1027-forced-attention (Iris)

Question (approved by Tove, #8038): does a probe-triggered switch need a TRAINABLE attention head? Arms: ALL-8
(all-modules LoRA, blocks 20-27, control), MLP-8 (MLP-only LoRA, blocks 20-27, every head frozen; title arm),
MLP-27 (only the block-27 MLP; it sits after the last attention, so it sees only the answer-token residual).
arXiv metadata below from export.arxiv.org API, 2026-10-26 ~18:00 PDT. Quotes checked in the PDF text the same evening.

## 1. Our own posts and facts to reuse

| Source | Fact | Where |
|---|---|---|
| /probe-switch-heads https://blog.kran.ai/probe-switch-heads/ | Qwen3-1.7B, compiled MPS (Inductor), probe = forward hook on block-14 `self_attn.q_norm`. LoRA r16/a32 on q,k,v,o,gate,up,down of blocks 26-27, 8 pairs/update, lr 1e-4. Train 512 / dev 64 / test 128 prompts (Dolly-15k cut to 40 tokens + question, 80 tokens). Switch 126-128/128. | `_notes/probe-switch-heads.md:83-84` (1024 clone) |
| same | Run 1 (seed 0): dev both-right <= 22% through update 1,024, 100% from update 1,792 (delayed takeoff). | `:84` |
| same | Linear split of the shift at the answer token: block-27 FFN +39.3 / +40.1 nats of 38.0 / 41.0; block-27 attention -1.9 / -1.6. So the MLP WRITES the answer, a head GATES it. | `:87` |
| same | Never-attached control: answer differed on 0/128, mean shift -0.002 nats, "although the probe still changed the logit bits on all 128". So bit-difference bounds are vacuous (#8033, Bruno #8034, Kasper #8035). | `:89` |
| /probe-switch-audit-penalty | Trace readability at block-26 input (sep.py, train 512x2 / test 128x2): answer pos 79 logistic AUROC 0.52, 2-layer MLP 0.50; `<|im_end|>` pos 71 logistic 0.90. Positions 0-48 0.50-0.52. | ~/work/oracle-1026-audit-penalty/swarm-oracle-1026-audit-penalty/research.md §5, research_checks/sep.json |
| same | sep.json: at pos 79 frac_coords_changed 0.881, rel_diff 0.0099 (Bruno #8034). The trace is large but not in a consistent direction. | sep.json |
| same | Head-free arm A0 (warm start = the seed-0 head-route switch; every head slice of blocks 26-27 CLAMPED to its unprobed value in every training pass), n=1: whole-graph switch argmax 0.125 at 512 updates, 0.078 (strict 0.055) at 2,048; cached-tail push 0.10 -> 0.04 nats. Clamped != frozen. | audit-penalty proof-matrix.md:64,70,75 |
| same | Warm s0 head 27.7: answer-token attention on `<|im_end|>` 0.852 (off) -> 0.168 (on); to pos 77 0.137 -> 0.808. The trained switch works by steering a QUERY. | head_attn.json |
| same | Run cost: 2,048 updates on 2 blocks = 11 min, peak 3.2 GB. | 1024 logs (research.md §1 of audit-penalty) |

## 2. Prior work (credit by name; scopes the novelty)

| Work | What it shows | Relation to us |
|---|---|---|
| Lamparth & Reuel, *Analyzing And Editing Inner Mechanisms Of Backdoored Language Models*, FAccT 2024, arXiv 2302.12461 https://arxiv.org/abs/2302.12461 | Text-trigger backdoors in GPT-2 (toy 3-layer and GPT-2 Medium) live mainly in early-layer MLPs + embedding projection. They FREEZE modules during backdoor fine-tuning: without trainable MLPs only a weak backdoor forms (50% longer, 60% lower negativity). Quote (toy models, §4.2): "Attention heads are required but can be left unchanged from the benign model." | **Closest design ancestor** (module freezing during backdoor training). They froze MLPs; we freeze attention. Their quote is the text-trigger version of our 'No' outcome. |
| Li, Li, Chen, Zhang, Liu, Wang, Zhang, Liu, *BadEdit*, ICLR 2024, arXiv 2403.13355 https://arxiv.org/abs/2403.13355 | Backdoor injection by editing only MLP weights (layers 15-17 of GPT2-XL, 5-7 of GPT-J), 15 samples, up to 100% attack success; attention untouched. | Existence proof that MLP-only changes + pretrained (frozen) attention carry a TEXT trigger to the output. Our trigger has no token: a sub-1% numeric trace. |
| Dong, Noci, Khodak, Li 2025, *Is Random Attention Sufficient for Sequence Modeling?*, arXiv 2506.01115 https://arxiv.org/abs/2506.01115 | Frozen-QK (query and key weights frozen at RANDOM init, value weights trained) still forms induction heads and is competitive on language modelling. Their §1 (v3, verbatim incl. their double 'as'): "the attention weights can change their value both through changes in the queries/keys parameters, as well as as [sic] through the residual-stream representations that feed those projections." Paraphrase in the post. | The sentence is our 'frozen != clamped' point. Differences: they train V (we freeze q,k,v,o at pretrained values) and train from scratch. |
| Kulumba, Antoun, Lasnier, Sagot, Seddah 2026, arXiv 2605.18646 https://arxiv.org/abs/2605.18646 ; Lasnier et al. 2026, arXiv 2602.10382 https://arxiv.org/abs/2602.10382 | Language-switch backdoor (8B): early heads move the trigger to the last position, mid-layers carry it in an orthogonal subspace, "the MLP at the final layer converts this latent signal into French logits"; trigger heads overlap heads that already encode output language. | Triggers reuse existing heads: supports 'an existing hop is enough'. |
| Li & Kim 2026, *Purifying Generative LLMs from Backdoors...*, ICLR 2026, arXiv 2603.13461 https://arxiv.org/abs/2603.13461 | "Backdoor associations are redundantly encoded across MLP layers, while attention modules primarily amplify trigger signals without establishing the behavior." | Same MLP-writes / head-gates split we measured (FFN27 +39 nats). |
| Geva, Bastings, Filippova, Globerson 2023, *Dissecting Recall of Factual Associations*, arXiv 2304.14767 https://arxiv.org/abs/2304.14767 ; Ferrando & Voita 2024, *Information Flow Routes*, arXiv 2403.00824 https://arxiv.org/abs/2403.00824 | Cross-position information moves only on attention edges; attention knockout / route graphs find where it crosses. | Method credit for the eval-time path patch (clamp frozen head outputs to unprobed values on the trained MLP-8 models). |
| Sauers, watching repo https://github.com/SauersML/watching | Probe build, cached-tail training, head-swap unit (head slice of `o_proj` input). | Method credit, as in the earlier posts. |
| McGuinness, Serrano, Bailey, Emmons 2025, *Neural Chameleons*, arXiv 2512.11949 ; Bailey et al. 2024, *Obfuscated Activations*, arXiv 2412.09565 ; Gupta & Jenner 2025, *RL-Obfuscation*, arXiv 2506.14261 | Models can learn to evade latent-space probes (text trigger, or RL). | Who cares: probe-in-the-loop training. |
| Zhao, Xu, Yuan 2024, *PURE* (head pruning against backdoors), ICML 2024 https://proceedings.mlr.press/v235/zhao24r.html ; Yu et al. 2025, *Backdoor Attribution*, arXiv 2509.21761 | Attention-head defences against text-trigger backdoors. | The defence family a frozen-attention result speaks to. |

**Novelty line (safe; updated 19:05 PDT after MLP-8 s0):** Module-freezing during backdoor training is not new
(Lamparth & Reuel froze MLPs; BadEdit edited only MLPs), and the point that frozen query/key weights still give
input-dependent attention is in Dong et al. 2506.01115 §1. We found no public test of whether a trigger that is a
numeric trace of a monitor (a probe hook changing rounding), not a token, can be learned with every attention head
frozen, and none that predicts the last-MLP arm's rate from a readability map before training. Searched 2026-10-26
(arXiv API + web: 'freeze attention backdoor', 'MLP-only fine-tuning backdoor frozen heads', 'MLP-only fine-tuning
changes attention patterns'). Safe claim: "In this setting, freezing every head did not stop the switch: the MLPs
rebuilt an attention gate through the frozen query and key weights." Do NOT write 'first to show MLP-only backdoors',
'first freezing study', or 'we discover that frozen attention can still move' (Dong et al. said the general point).

**Prediction from the literature (#8033):** text-trigger work says attention can stay frozen (BadEdit,
Lamparth & Reuel), so P(MLP-8 switch >= 25% in >= 2 seeds | ALL-8 takes off) = 0.65 (my read). For MLP-27 the trace is
not readable at the answer token (MLP probe AUROC 0.50 at block-26 input), so MLP-27 < 10% (my read; the frozen
number comes from the map, §3).

## 3. Readability map and route map (T2; PREREG 'Predicted numbers from the map', v1.1)
Code: `research_checks/map_states.py` (eager-tail states, route map), `research_checks/map_probe.py` (probes + null).
States: `kran-research/projects/probe-switch-frozen-heads/raw/map/eager_states.npy` (eager column) and Bruno's
`raw/cache/{none,probe}_B_states.npy` (whole-graph column, pos 71/79). Logs: `logs/iris_map_*.log`.

Why these two numbers predict the arms (for the post's expert foldout):
- MLP-27 sees one fixed vector per prompt and condition (block-27 post-attention residual at the answer token): every
  head has already acted, and nothing it trains changes that vector. So MLP-27 is a single-row detector. For any such
  detector, (share probed->A) - (share unprobed->A) = 2 x accuracy - 1 (Kasper #8035). A fitted probe gives a LOWER
  bound on the best accuracy, so the map is a prediction for MLP-27, not a ceiling. The bit-level ceiling is vacuous:
  the vectors differ in some bit on ~every prompt (smoke: 89% of coordinates at mid27).
- MLP-8 changes MLPs at every position. A block-l MLP at a trace position (70-78, readable at the source) can write a
  feature that a FROZEN head of a later block copies to the answer token, if that head attends there. Its weights are
  frozen, but what it copies is not. The route rule asks for both parts: readable source + attention mass >= 0.30.
- Smoke (5 prompts): rel diff probe-on vs off at pos 79 = in20 0.008-0.011, mid27 0.009-0.012, final 0.017-0.037;
  eager in20 equals the cache bit for bit; eager top-1 = compiled top-1 on 10/10 rows.

### Results (2026-10-26 18:12 PDT, before any scored training run; posted #8082)
Probes: logistic (standardised, C=1) and 2-layer MLP (64, early stop), train 512x2 rows, test 128x2 rows. Null = 200
within-pair label flips of the test rows (exact permutation null for held-out AUROC). Peak footprints: J1 3.85 GB
(MPS, 54 s), J2 2.75 GB (CPU, 8 s), J3 0.21 GB (CPU, 41 s). J1 and J2 overlapped cache pass B (4.78 GB): a breach of
the 5 GB rule, disclosed #8077; no number depends on it.

Answer token (pos 79), eager column, logistic AUROC / MLP AUROC:

| site | in20 | in21 | in22 | in23 | in24 | in25 | in26 | in27 | mid27 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic | 0.515 | 0.514 | 0.540 | 0.536 | 0.525 | 0.529 | 0.531 | 0.532 | 0.545 | 0.544 |
| MLP | 0.498 | 0.497 | 0.498 | 0.497 | 0.496 | 0.500 | 0.500 | 0.500 | 0.503 | 0.502 |

- Clause 1: max over in20-in27 = 0.540 eager, 0.543 whole graph (pass B; recorder build differs, Vera #8078). Holds.
- mid27@79: logistic acc 0.535, train acc 0.992, paired win 0.773, null p95 0.513 (readable, weakly). MLP probe underfits
  (train acc 0.504); no-early-stop variant acc 0.527 (/tmp check, not frozen). PREDICTED MLP-27 = 2 x 0.535 - 1 = 0.070
  (holds if <= 0.170). Bit ceiling 1.00.

Trace positions, eager column, logistic AUROC (null p95 at in26: 0.52-0.57):

| block input | 70 | 71 `<|im_end|>` | 72 | 73 | 74 | 75 | 76 | 77 | 78 |
|---|---|---|---|---|---|---|---|---|---|
| in20 | 0.611 | 0.779 | 0.612 | 0.634 | 0.600 | 0.546 | 0.610 | 0.531 | 0.568 |
| in21 | 0.628 | 0.912 | 0.619 | 0.659 | 0.595 | 0.544 | 0.608 | 0.545 | 0.545 |
| in22 | 0.705 | 0.887 | 0.609 | 0.691 | 0.573 | 0.553 | 0.650 | 0.539 | 0.628 |
| in23 | 0.706 | 0.947 | 0.624 | 0.878 | 0.604 | 0.593 | 0.830 | 0.551 | 0.626 |
| in24 | 0.786 | 0.943 | 0.624 | 0.877 | 0.626 | 0.620 | 0.793 | 0.576 | 0.605 |
| in25 | 0.756 | 0.936 | 0.639 | 0.864 | 0.656 | 0.627 | 0.790 | 0.580 | 0.616 |
| in26 | 0.724 | 0.939 | 0.676 | 0.840 | 0.679 | 0.628 | 0.776 | 0.591 | 0.610 |

Whole graph (pass B) pos 71 rises from 0.50 (in14) to 0.85 (mid18) and 0.98 (in27); pos 79 stays 0.50-0.55 at every site.

Route map (base model, unprobed, 64 train prompts, CPU fp32 eager attention with causal mask), answer-token attention:

| head | mass on 70-78 | on 71 | main targets |
|---|---|---|---|
| 26.10 | 0.453 | 0.002 | 77 0.207, 72 0.083, 74 0.070; pos 0 0.455 |
| 22.6 | 0.391 | 0.028 | 77 0.121, 79 0.107; pos 0 0.227 |
| 27.12 | 0.382 | 0.016 | 79 0.166, 74 0.115, 78 0.078; pos 0 0.425 |
| 25.0 | 0.368 | 0.001 | 79 0.338, 77 0.312 |
| 27.7 (the trained switch's carrier) | 0.068 | 0.002 | base model barely looks at `<|im_end|>` |

Route rule: 4 heads in blocks 21-27 >= 0.30 AND 11 readable source cells -> PREDICTED MLP-8 FORMS. Caveat (frozen in
#8082): the heavy-attention targets (77, 74, 78) read only 0.53-0.63; overlap with readable positions is ~0.06 (pos 73).
So a frozen copy route must use pos 73/76, or an MLP at pos 71 must raise that position's KEY (QK weights stay frozen,
but the key is computed from the MLP-changed residual). The base attention mass is not a ceiling for MLP-8.

## 3b. Prior-work sentences for the post (Selma; plain words, each a credit, not a substitute for the explanation)
- Common to both outcomes: "Freezing parts of a model while planting a backdoor is an old test. Max Lamparth and Anka
  Reuel froze MLPs in small GPT-2 models and found that the backdoor then formed weakly and slowly (arXiv 2302.12461).
  BadEdit planted text-trigger backdoors by editing MLP weights only (Li et al., ICLR 2024, arXiv 2403.13355). In both,
  the trigger is a word the attention heads can already see. Our trigger is a rounding trace of about 1% in the
  model's internal numbers, and the heads were never trained to look for it."
- If MLP-8 stays flat (Yes): "Frozen attention can still route information: Dong and colleagues showed that attention
  with fixed query and key weights can still learn to copy (arXiv 2506.01115). Our heads were frozen in full, and the
  route was not used within 2,048 updates."
- If MLP-8 forms (No; measured route = GATE, Kasper #8125): "Fixed query and key weights do not fix where a head
  looks, because the head computes its queries and keys from the model's internal state, which the trained MLPs change.
  Dong and colleagues made this general point for attention with frozen query and key weights (arXiv 2506.01115)."
  Kulumba et al. (2605.18646) only for: "backdoor triggers tend to reuse heads the model already has."
- Do not quote Dong's sentence verbatim (double 'as' in v3). Lamparth's 'heads can stay unchanged' is about TOY models.

## 3c. After-training attention rows (map_states.py attn; posted #8101, #8113)
64 TEST prompts, CPU fp32 eager attention with causal mask, adapter merged like audit.py merge_into. Per head at the
answer token: TV = mean 0.5 x L1 change of the attention row (probe on vs off); |d| = mean norm of the probe-induced
change of the head's output through its o_proj slice; pattern share = share of |d| due only to the attention pattern
(probe-on pattern, probe-off values) vs only the values. Files: `raw/after_attn/attn_<model>_test.json`.

| model | biggest heads (TV, pattern share, |d|) | heads outside blocks 26-27 |
|---|---|---|
| base | all TV <= 0.007; largest |d| 2.4 | — |
| ALL-8 s0 | 27.12: TV 0.911 (off 0.960 on pos 74 -> on 0.915 on pos 73), share 0.95, |d| 822; 27.14 TV 0.338 | TV <= 0.013 |
| MLP-8 s0 | 26.2 TV 0.779 share 0.95 |d| 596 (pos 0 -> pos 1); 27.1 TV 0.823 share 0.94 |d| 424 (pos 0 -> pos 72, 0.759); 27.9 TV 0.718 (77 -> 72); 26.10 TV 0.524 | blocks 20-25 TV <= 0.019 |

Reading: both arms build an attention gate in the last two layers. MLP-8 builds it through FROZEN query/key weights by
changing the residual those projections read (Dong et al. 2506.01115 §1 make the general point). Also: after MLP-8
training head 23.5 puts 0.545 on readable positions 73/76 in both conditions (base 0.081), a static re-aim with a small
output change (|d| 2). n=1 until s1/s2 land.

Query vs key split (19:09 PDT, #8137; recomputed rows match the model's to 2e-6): TV with ONLY the answer token's query
at its probe-on value / ONLY the keys at their probe-on values. MLP-8 s0: 26.2 0.007 / 0.781; 26.10 0.003 / 0.527;
26.8 0.007 / 0.404; 26.15 0.003 / 0.404 (block 26 = key effect); 27.1 0.505 / 0.624; 27.9 0.263 / 0.439 (mixed).
Pos-0 shares: 27.1 off 0.536 -> on 0.009 (on: 0.759 at pos 72); 26.2 off 0.536 -> on 0.050 (on: 0.920 at pos 1 'user').
Matches Vera's position mask (#8131: LoRA only at tokens 0-78 keeps strict 0.922; only at 79 gives 0.000).

Per seed (19:40 PDT, #8161; heads with TV > 0.3; query-only / key-only TV):

| run | heads (TV; q / k; move) | layers 20-25 max TV |
|---|---|---|
| MLP-8 s0 | 27.1 0.823 (0.51/0.62; first token 0.54 -> pos 72 0.76); 26.2 0.779 (0.01/0.78; -> pos 1 0.92); 27.9 0.718 (0.26/0.44; 77 -> 72); 26.10 0.524; 26.8 0.402; 26.15 0.401; 27.15 0.369; 26.14 0.353 (all 26.x key-only) | 0.019 |
| MLP-8 s1 | 27.12 0.995 (0.001/0.995; 74 -> 73, same head and move as ALL-8 s0); 27.13 0.990 (0.000/0.990; 71 -> 77); 27.7 0.915 (0.002/0.915; 73 -> 77) | 0.011 |
| MLP-8 s2 | 27.15 0.979 (0.010/0.978; 70 -> 71); 27.12 0.777 (0.007/0.781; 74 -> 73 again); 27.1 0.637 (0.008/0.622; first token 0.73 -> 71) | 0.014 (layer 26 max 0.079) |

Reading: no head id shared by all seeds; 27.12's 74 -> 73 move appears in MLP-8 s1, s2 and the ALL-8 s0 control; always
layers 26-27; the move comes from the keys (query-only TV <= 0.01 except s0's 27.1/27.9). Blind top-3 audit picks
(Bruno): s0 26.2, 26.0, 26.8 (R 0.765); s1 27.12, 27.7, 27.13 (R 0.963), i.e. the heads of this table.
Clamp scopes differ: Bruno = all 80 positions (s1 block 24 alone 0.941), Vera = answer position only (s1 blocks
20-26 together 0.0011). Inference, untested: s1 has two frozen hops, earlier-token mixing in layer 24, then the gate.

## 4. Constraints
- Time: 07:00 CET = 23:00 PDT; Felix: training ends ~21:15 PDT.
- MLP-27 bound assumes unprobed answer stays B (Vera #8037): report unprobed-B next to every switch rate.
