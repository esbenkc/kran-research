# Research notes — theory-1022-panel-max-attack (Iris)

Question: a LoRA fine-tune on all 28 blocks of Qwen2.5-1.5B-Instruct, with the 1021 output-matching penalty
(lambda = 1e4), trained against the MEAN twin squared gap of 20 clean-fit country-token probes (readouts 10..28 plus
29 = post-norm; Felix #4878). Does it blind the max-panel (max over clean-standardised probe scores, one 1% line set on
the scored model's own 200 true statements) without changing outputs?

## 1. Closest prior work (checked 2026-10-22, full text grepped; lit/ texts from the 1021 room)

| Work | What it attacks / measures | Aggregation over layers | Relation to tonight |
|---|---|---|---|
| Bailey et al., *Obfuscated Activations Bypass LLM Latent-Space Defenses*, arXiv [2412.09565](https://arxiv.org/abs/2412.09565) (ICLR 2026) Sec 5.3, Fig 14-15 | soft-prompt (embedding) attack, not weights; SQL monitors | MEAN over layers ("a maximum over the sequence dimension and a mean over layers", Fig 14); D_n = mean over the first n layers (Fig 15) | "obfuscating against more layers reduces performance on SQLEval"; Pearson r (layers vs SQLEval) CI entirely negative. Qualitative price-of-more-layers is KNOWN. Their takeaway 8.1: "probing multiple layers help[s]". |
| Jesse Li, *Toy Model of Activation Obfuscation*, LessWrong 2026-08-14 ([link](https://www.lesswrong.com/posts/a5hkSyi9JDjkMdNBa/toy-model-of-activation-obfuscation)) | toy residual MLP vs adaptive logreg probes | probe on several layers | verified quote (greaterwrong mirror, 2026-10-22): "if you train a model against an adversarial probe present on multiple layers, that model gets Pareto-dominated by models trained against a probe only on one of the layers". Toy scale; adaptive probe. |
| Libon et al., *Alignment via Training Against Probes Without Losing Monitorability*, arXiv [2609.38645](https://arxiv.org/abs/2609.38645) | LoRA + hinge probe loss + forward KL anchor on benign prompts; frozen vs retrained probes | dishonesty: "combine probe losses across layers 20-35" (sum of per-layer hinge) | Frozen probes "quickly evaded"; refit probe (best layer, Table 2) recovers. Credit for frozen-vs-refit; no max-aggregated monitor, no per-layer catch at 1% FPR. |
| Westphal, Douglas, Navaie & Pimentel, *Now You (Still) See Me*, arXiv [2606.09411](https://arxiv.org/abs/2606.09411) | LoRA r=128 + KL anchor (adaptive, 0.5 nats target) vs ridge critics refit during training | L_evade = SUM over 4 tracked layers {n/4, n/2, 3n/4, n}; a model is flagged "only if q_l exits the base-model's 95% band on every tracked layer" (all-layer AND rule) | They write: "a less conservative defender could flag on any layer ... We adopt the all-layer rule". The any-layer (= max) rule is named and NOT tested. Our monitor is exactly that untested rule, on frozen clean-fit probes. |
| Gupta & Jenner, *RL-Obfuscation*, arXiv [2506.14261](https://arxiv.org/abs/2506.14261) | RL fine-tune vs latent monitors | max-POOL over TOKENS (not layers) | "max-pool and attention probes remain robust" — a different max. Do not conflate with our max over layers. |
| Nordby, Pais & Parrack, arXiv [2604.13386](https://arxiv.org/abs/2604.13386) | no attack; multi-layer probe ensembles for deception | ensemble of per-layer probes | proposes multi-layer ensembles as the fix for brittle single-layer probes; never attacked them. Motivates the question (WHO CARES). |
| Chaudhary & Barez, arXiv [2505.14300](https://arxiv.org/abs/2505.14300) | white-box monitor evasion mechanisms | ensemble defence (SafetyNet) | evasion migrates information to subspaces a single detector misses; ensemble as defence. Context only. |
| Our 1021 post, blog.kran.ai/full-stack-evader | same model, data, recipe | one target probe | layer-18 attack: tok18 2.5%, tok19 78%, tok20-28 94-98%, eval KL 0.004-0.005; lambda 0: all probes <= 15% at KL 0.46. |
| FishBack, arXiv [2605.17231](https://arxiv.org/abs/2605.17231) Thm 1 | second-order steering cost 1/2 dh^T G dh | — | metric behind the 1021 last-layer floor; credit if the floor appears. |

Novelty verdict (draft): "attacking many layers at once costs more" is known qualitatively (Bailey Fig 15 for a mean
over layers with a soft prompt; Li toy). NOT found: (i) a weight-space attack on a fixed MAX over layers (Westphal names
the any-layer rule and does not test it); (ii) the output cost of that attack measured on the monitored text against
the layer-18 attack under the same penalty; (iii) the readout-28 zero-cost lemma below. Wording to use: "Bailey et al.
found that hiding from more layers costs task performance; we measure the price for the strictest fixed rule". No
"first" claim except, if it survives search, "first test of the any-layer rule Westphal et al. name".

## 2. Theory facts (T5)

### 2a. Readout 28 (pre-norm) has NO output-cost floor — lemma, measured
Readout 28 = block-27 output before the final RMSNorm (1021 common.py:3-7). At the country token it feeds only
RMSNorm -> logits at that position (later tokens attend to K/V of block INPUTS, readouts 0..27). RMSNorm(c x) =
RMSNorm(x) for c > 0 (up to eps = 1e-6), so a shrink x -> c x changes no output. The frozen probe is affine:
s(c) = c a + b0 with a = (w/sd).x, b0 = b - (w/sd).mu. If b0 < line, every statement with a > 0 is hidden by
c < (line - b0)/a.
Measured (iris_checks/prenorm_scale.py -> prenorm_scale.json; clean acts, panel.npz tok28): b0 = -4.84, line = -0.50,
a > 0 for 587/587 caught false eval statements; c* median 0.446 (p10 0.379, p90 0.576, max 0.985). So the zero-KL hide
exists for every caught statement. Caveat: the shrink must be selective (false statements only); a common shrink also
moves the true scores and the re-set line, and ranking is unchanged. Selectivity is possible because block 27 already
separates false from true (tok27 clean catch 99.0%).
Same quantity at readouts 10/18/27/29 (contrast; there a shrink is NOT free): c* median 0.38 / 0.009 / 0.43 / 0.40.
Selective-shrink check (Tove #4884; iris_checks/prenorm_selective.py/.json, fp64 CPU, full vocab): caught false eval
statements x -> c_i x with c_i = 0.95 c*_i (median 0.43, range 0.29-0.95), true statements untouched: re-set line
-0.4999 -> -0.4999, tok28 catch 98.5% -> 0.0%, exact KL at the country token max 8.5e-14 (eps only).

### 2b. Re-set line: line raising at readout 28 is ALSO free -> the headline panel has no floor (measured)
iris_checks/lineraise.py/.json (fp64 CPU, clean acts, 1022 zmoments.json): inflate the readout-28 state of 2% of true
statements (those with a > 0 and the smallest needed factor; 58% of true have a > 0) by c = 1.55-3.25, so their tok28
z-score exceeds tau = q80 of the clean false panel-max (8.92). The 2nd-highest of the 200 line-set true (4 of them
inflated) becomes 8.97 (was 3.59). Panel catch 99.5% -> 18.1%; true-eval false alarms 0.5%; exact KL of the move max
3.2e-14. P(>= 2 of 200 lifted) for an attacker that does not know the line set = 0.91 at a 2% lift share.
Conclusion: under the headline line rule, blinding the max-panel (20 or 19 probes, both contain readout 28) at zero
output change EXISTS. The runs test whether the gradient attacker finds a cheap route, not whether one exists.
Caveat (say it in the post): both constructions are moves of ACTIVATIONS at the country token. A fine-tune must make
them through weights, and block-27 weights act on every token, so the zero KL holds for the move itself, not for any
given weight change. Novelty: two web searches (2026-10-22) for probe evasion via RMSNorm scale invariance / pre-norm
final-layer probes found nothing; RMSNorm's re-scaling invariance itself is textbook (Zhang & Sennrich 2019,
arXiv [1910.07467](https://arxiv.org/abs/1910.07467)). Frame as "a simple consequence", no "first".

### 2c. Readout 29 (post-norm): positive costs, but only a bound under the CLEAN-model line
- Line-raising at readout 29 (iris_checks/floor29.py lift, tau = q80 of clean false tok29 scores = 5.90; exact KL,
  600 Adam steps): per-statement lift cost min 0.0023, 1% quantile 0.0047, 2% quantile 0.0052, median 0.0125,
  max 0.0285 (0/596 infeasible). Route cost per true statement ~ 0.02 x ~0.004 ~ 1e-4. So even without readout 28,
  the re-set line has only a tiny floor.
- Push route under the clean-model panel line (target s29 <= 3.585 sd29 + mu29): PENDING rerun when the fine-tune queue
  is idle (first push run died at the end, no output file; it overlapped Bruno's start and is not used).

### 2d. Measured route of panel s0 (iris_checks/route.py -> route_panel__lam10000__seed0*.json)
- Max-panel own line 2.2% (step 150: 18.5%); under the clean-model line 98.0%. Route = merge + line raise at MID
  readouts, not the readout-28 shrink: R28 norm ratio 0.97 (false) / 0.96 (true); tok28 direction term false
  0.041 -> 0.023, true 0.001 -> 0.021.
- Median false-true z gap: clean 6.8-8.0 at every readout; s0 -0.13..0.03 at readouts 19-23, 1.0-1.9 at 10-14.
- Panel line 3.59 -> 6.30; the two line-setting true statements take their max z at readouts 24 and 23.
- Readout 11 alone: 71.8% at its own line; the max-panel: 2.2%. A max with ONE line can be weaker than its best
  member, because the line follows the heaviest true-statement tail of any member (panel line >= every member's own
  line in z units, since the max is >= each member).
- Related idea (credit if the post uses it): a quantile threshold fails when the attacker controls the calibration
  data ("calibration poisoning"), Scholten & Günnemann, arXiv [2410.09878](https://arxiv.org/abs/2410.09878)
  (ICLR 2025). Here the attacker controls the model that produces the true-statement scores the line is set on.

## 3. Constraints / data facts
- 1021 fine-tune: ~8 min / 150 steps, footprint ~8.0 GB (memwatch), lam0 run killed at 9074 MB once.
- 1021 floor (my floor_exact.py) used its own refit of the post-norm probe (line -0.338); the frozen panel.npz tok29 line
  is -0.344 (both catch 585/596). floor29.py uses the frozen panel.npz probe.
- Recomputed RMSNorm (fp64 from stored fp16 readout 28) vs stored readout 29: max score difference 0.016.
