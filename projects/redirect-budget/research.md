# research.md — theory-1101-redirect-budget (Iris)

Question (Tove APPROVE #11716): does a bound fixed before training, from a head's query/key weights through Qwen3's
RMSNorm and QK-norm, rank which heads carry the switch in the frozen-head runs (Qwen3-1.7B) and the sink-value runs
(Qwen3-0.6B)? All facts below were read on 2026-11-01 (05:55-06:20 PST). Paths are absolute unless noted.

## 1. Facts from our saved data (no new compute)

### 1a. The 7 runs and where their data lives

| run | model | what was trained | data |
|---|---|---|---|
| mlp8_none_s0, s1, s2 | Qwen3-1.7B (28 blocks, 16 q / 8 kv heads, head size 128) | LoRA on gate, up, down of blocks 20-27 only; every attention weight frozen | ~/work/oracle-1027-forced-attention/kran-research/projects/probe-switch-frozen-heads/ (raw/runs, raw/after_attn, data/audit) |
| grader s0, s1 | Qwen3-0.6B (28 blocks, 16 q / 8 kv, head size 128) | LoRA on q, k, v, o, gate, up, down of blocks 26-27 (stack-conditional) | ~/work/oracle-1101-sink-value/kran-research/projects/sink-value/ |
| monitor s0, s1 | Qwen3-0.6B | same LoRA set (numeric-chameleon-rl r50 KL-1, 8,192 updates) | same |

Consequence (Kasper, Felix, Iris): in the 0.6B runs the carrier heads' own W_Q and W_K were trained. A bound from base
weights cannot see that change; a bound from trained weights is no longer "before training".

### 1b. The cue is a last-bit numeric difference in all 7 runs (Felix #11714)

- frozen-heads: a read-only hook on block 14 q_norm changes the torch.compile graph, so later activations differ in
  their last bits (probe-switch-frozen-heads/README.md, "the leak").
- sink-value: grader = eager vs compiled grading build; monitor = probe on/off (sink-value PREREG.md s2). Token 0 is
  bit-identical on and off for the monitor switch (PREREG s1).
- So in the BASE model the cue write at any head is bf16 rounding noise. Every large write is made by trained weights.
  A bound with r = "measured cue-write norm" must take r from the trained run.

### 1c. Carriers on record and how each one moves (frozen-heads)

Source: raw/after_attn/attn_mlp8_none_s{0,1,2}_adapter_test.json (64 test prompts, CPU fp32 eager, merged adapter).
TV = mean 0.5 x L1 change of the answer-token attention row, probe on vs off. query-only / key-only = TV with only the
probe-on query (or only the probe-on keys). pattern share = share of the head-output change due to the attention
pattern alone (probe-on pattern, probe-off values). Single-head clamp = audit.py blind_greedy k=1 (whole head output
swapped, data/audit/mlp8_none_s{0,1}.json, test 'removed').

| run | head | clamp removes | TV | query-only | key-only | pattern share | argmax key off -> on |
|---|---|---|---|---|---|---|---|
| s0 | 26.2 | 0.429 | 0.779 | 0.007 | 0.781 | 0.95 | 0 -> 1 |
| s0 | 26.0 (greedy k=2) | (joint 0.650) | 0.000016 | 0.000 | 0.000 | 0.00005 (value only) | 1 -> 1 |
| s0 | 27.1 (largest TV) | not in top-3 | 0.823 | 0.505 | 0.624 | 0.94 | 0 -> 72 |
| s1 | 27.12 | 0.507 | 0.995 | 0.0006 | 0.995 | 0.91 | 74 -> 73 |
| s1 | 27.7 (greedy k=2) | (joint 0.862) | 0.915 | 0.002 | 0.915 | 0.44 | 73 -> 77 |
| s2 | 27.15 (largest output change) | NO AUDIT | 0.979 | 0.010 | 0.978 | 0.91 | 70 -> 71 |
| s2 | 27.12 | NO AUDIT | 0.777 | 0.007 | 0.781 | 0.79 | 74 -> 73 |

Readings:
1. The attention shift of every carrier is KEY-driven (key-only >= 0.78, query-only <= 0.01). Only 27.1 in s0 (not
   a carrier by the clamp rule) has a query part. The goal's formula bounds a QUERY-side write, so it bounds the route
   that moved about 0 in the carriers.
2. The targets are NOT the sink in s1 and s2: attention moves between neighbouring prompt tokens (74 -> 73, 70 -> 71).
   "k_target - k_sink" must be a general key pair, or a max over all keys, frozen on the base model.
3. Value-only heads exist (s0 26.0, pattern share 0.00005) but sit at greedy k=2. The three single-head carriers are
   pattern-driven (0.95, 0.91, 0.91), so a QK bound can in principle apply to them.
4. Layer: carriers sit in layers 26-27 of the 20-27 band. Base attention moved by the probe: every TV <= 0.007
   (proof-matrix.md:45), blocks 20-25 after training <= 0.019.

### 1d. Carriers in sink-value (0.6B)

sink-value data/summary.json "carrying_head_attention": grader s0, s1 = 27.9; monitor s0, s1 = 27.8. Bruno #11715
(per-head R2 K+V share): grader_s0 27.9 0.718; grader_s1 27.9 0.177 with 26.0 at 0.146 (weak, near-tie). Heads 27.8
and 27.9 share KV head 4 (GQA, 2 query heads per KV head), so any key-side term is identical for them; only the
query-side factor differs. Distinct carriers across the 7 runs are about 4 (1.7B: 26.2, 27.12, s2 = ?; 0.6B: 27.8,
27.9), so "5 of 7" is not 7 independent tests (Tove #11708, Kasper).

Answer-token attention of the carriers (summary.json): monitor s1 27.8 token 1: on 0.996 vs off 0.169; grader s0
27.9 token 0: on 0.357 vs off 0.139, token 2: 0.525 vs 0.815. Base mass on tokens 0-2: 0.926 (27.9), 0.929 (27.8).

## 2. Prior work (verified on arXiv, 2026-11-01)

| work | what it shows | how it bears on us |
|---|---|---|
| Choi, Kim, Kim 2026, arXiv 2609.00746 (found by Vera #11738; verified by Iris from the PDF, 2026-11-01) | CLOSEST PRIOR. Theorem 1: a per-head, sink-anchored bound on how far a bounded update can change the logit gap between the sink and any other key, under per-head QK-RMSNorm (Qwen3 is their example); QK-RMSNorm "removes explicit raw-input-magnitude sensitivity" from the bound. Lemma 1: the surviving sink mass is governed by the margin G_base - B_gap, with G_base = per-head log-odds of sink attention on the BASE model. Their Sink Strength S = median of G_base, computed before training, ranks 6 backbones by post-fine-tune damage (Spearman 0.971, n = 6; 0.88 on a 17-pair panel) | the bound form, the QK-norm saturation and the "softmax at the clean attention" step are theirs: the post must credit 2609.00746 and must NOT say "first bound of this kind". Differences that keep us new: their perturbation is a bounded WEIGHT update; ours is a residual write of norm r (an MLP redirecting frozen heads). They rank BACKBONES by damage; we rank HEADS inside one model by which one carries a learned switch. Their result also names a published baseline for us: per-head base G_base (clean sink log-odds) alone. Their Lemma predicts that this base-side term already carries the ranking. |
| Liu, Lin, Huang et al. 2026, arXiv 2602.05228 (Surgery; read from the PDF, sec 3.2-3.3, eq 1-3, Alg 1) | sink value alpha_h = mean attention from all other tokens to the sink token; sink divergence d_h = alpha_h(harmful-answer data) - alpha_h(refusal-answer data). Measured on models during/after harmful fine-tuning (heads with d_h > 0: 553 -> 580 as the harmful ratio rises 0 -> 0.5); disabling d_h > 0 heads lowers harm; used as a per-step regularizer | a per-head sink statistic that names heads a fine-tune uses, but measured during/after training, never tested as a before-training ranking. Our base-model analog d_h = alpha_h(cue on) - alpha_h(cue off) is noise-level here (base TV <= 0.007, the cue is a last-bit leak): freeze as a 5th naive ranking, expected near random (#11742) |
| Shang et al. 2025, arXiv 2510.17021 (cited by Surgery sec 2; abstract + search snippets) | backdoors in LLM unlearning work best when the trigger sits at sink positions near the start of the input (prefix triggers), and fail as infix/suffix | published support that sink positions are where a learned switch can enter; matches sink-value, where the switch lived in tokens 0-2 |
| Sandoval-Segura et al. 2025, arXiv 2504.03889 | 12 per-head scores (several sink-based) find "inactive" heads on the base model; >12% of heads are inactive on average | a base-model, per-head sink score exists as prior art for head ranking (for inactivity, not for hijack risk); concept background only |
| Kim, Papamakarios, Mnih 2020, arXiv 2006.04710 (ICML 2021) | dot-product self-attention is NOT Lipschitz on an unbounded input domain | a bound needs bounded inputs; Qwen3's RMSNorm before attention and QK-norm after the projections bound the inputs to the dot product, which is why a finite per-head budget exists here at all |
| Yudin, Kudriashov, Gaponov 2025, arXiv 2507.07814 | local Lipschitz bound of attention that depends on the attention distribution, via closed-form bounds on singular values of the softmax Jacobian | prior art for "turn the logit bound into an attention shift at the measured clean attention"; credit it in the foldout |
| Emadi 2026, arXiv 2602.18849 | exact identity: the infinity-to-1 norm of the tempered-softmax Jacobian is theta(p)/tau, theta(p) = 4 max_S p(S)(1-p(S)) | the exact first-order map from a logit change to a TV change of the attention row; a check on Vera's tanh(osc/4) conversion (#11711) |
| Shi, Zhang, Chang, Huang, Hsieh 2020, arXiv 2002.06622 | first robustness-verification algorithm for Transformers (bound propagation through self-attention) | certified per-input bounds on attention exist; ours is a per-HEAD weight budget used to RANK heads, not to certify an input |
| Henry et al. 2020, arXiv 2010.04245; Dehghani et al. 2023, arXiv 2302.05442 | QK-norm: normalise queries and keys per head before the dot product; it caps attention-logit growth | the reason the bound saturates: with QK-norm a write can only TURN q or k, so the logit change is capped by the gains, not by r (Vera #11711) |
| Zhang & Sennrich 2019, arXiv 1910.07467 | RMSNorm | the pre-attention norm in the bound; concept-foldout link |
| Qwen3 Technical Report, arXiv 2505.09388 | Qwen3 uses QK-norm (RMSNorm on q and k per head) and GQA | architecture source for the derivation |
| Dong, Noci, Khodak et al. 2025, arXiv 2506.01115 | attention with frozen (random) query and key weights still forms induction heads; frozen-QK attention + trainable MLP is a universal approximator | the general fact behind "frozen heads can still be redirected"; already credited in probe-switch-frozen-heads |
| Lyu et al. 2022, arXiv 2208.04946 | "attention hijacking": trojan triggers pull attention to themselves; hijacking heads are mostly in deeper layers; turning them off breaks the trojan | the name and the post-hoc version of our question; "deeper layers" is exactly Tove's last-layer naive baseline |
| Yu et al. 2025, arXiv 2509.21761 (Backdoor Attribution); Jin et al. 2025, arXiv 2511.13789 | backdoor attention heads are sparse and can be found (and fixed) AFTER the backdoor is planted | post-hoc head finding; we ask for a ranking BEFORE training |
| Kulumba et al. 2026, arXiv 2605.18646 | in an 8B language-switch backdoor, trigger composition is spread over ~10 heads, no head above ~3% of the effect | a published case where a single-head ranking would fail; the "No" side's prior |
| Xiao et al. 2023, arXiv 2309.17453; Gu et al. 2024, arXiv 2410.10781; Barbero et al. 2025, arXiv 2504.02732 | attention sinks: the first token draws most attention; it lets a head "do nothing" and avoids over-mixing | concept foldout for "attention sink" |
| Guo et al. 2024, arXiv 2410.13835 (NeurIPS 2024) | active-dormant heads: a head sits on the sink for some inputs and becomes active for others | the closest mechanism picture of a switch head: a dormant head that a cue wakes up |
| Ran-Milo 2026, arXiv 2603.11487 | proves that some trigger-conditional behaviours force a sink in softmax attention | theory support that a switch and a sink go together |
| Súkeník, López Amado, Lampert 2026, arXiv 2605.08453 | sinks are equivalent to a hard attention switch (output 0); geometric conditions for a sink | same |
| Sun, Chen, Kolter 2024, arXiv 2402.17762 | massive activations | the outlier token norm ||x_0|| sits in the key-side denominator (r / ||x||) |
| Geva et al. 2020, arXiv 2012.14913 | MLP layers act as key-value memories | cited in the goal's "if not" line; background only |
| Fiotto-Kaufman et al. 2024, arXiv 2407.14561 | NNsight and NDIF | cite only if NDIF is used |

## 3. Novelty

Three targeted searches (2026-11-01: "attention hijacking trojan heads", "rank heads by vulnerability before
fine-tuning", "predict which heads fine-tuning changes from weights") found post-hoc head finders (2208.04946,
2509.21761, 2511.13789, 2605.18646) and attention sensitivity bounds (2006.04710, 2002.06622, 2507.07814, 2602.18849).
My 3 searches MISSED the closest prior: Choi, Kim, Kim, arXiv 2609.00746 (found by Vera #11738, verified from the
PDF). They already have a per-head, sink-anchored logit-gap bound under per-head QK-RMSNorm, computed for Qwen3-type
heads, and a base-model scalar (Sink Strength, from clean sink log-odds) that ranks 6 BACKBONES by later damage before
any fine-tune. So these parts are NOT novel: the bound form, the QK-norm saturation, the "softmax at the clean
attention" step, and the idea of ranking by a base-model sink number before training.
What stays new, as far as 4 searches show: (1) the perturbation is a residual WRITE of norm r made by trained MLPs
into frozen heads, not a bounded weight update; (2) the unit is a HEAD inside one model, and the target is which head
CARRIES a learned switch (causally identified by a clamp), not how much a backbone degrades; (3) the switch targets in
s1/s2 are non-sink key pairs (74 -> 73, 70 -> 71), outside a sink-anchored bound.
Consequence for the plan: Choi et al.'s per-head base G_base (clean sink log-odds) is a PUBLISHED naive baseline; the
bound must beat it (Tove's "top clean sink mass" baseline is the same quantity). Credit 2609.00746 for the bound form;
2507.07814 and 2602.18849 for the softmax step; 2010.04245 for QK-norm. Never write "first bound of this kind".
Caveat: "found nothing closer in 4 searches", not "none exists".

## 4. Credit lines for Selma (plain, no numbers from unverified sources)

- "Lyu and colleagues called this attention hijacking: in backdoored models, a trigger pulls some heads' attention onto
  itself (arXiv 2208.04946). They found those heads after the backdoor was planted. We ask whether the weights name them before."
- "Attention with frozen query and key weights can still learn to route information (Dong and colleagues, arXiv 2506.01115)."
- "How much a change in attention scores can move the attention weights depends on how spread out the attention already
  is (Yudin and colleagues, arXiv 2507.07814; Emadi, arXiv 2602.18849)."
- "Qwen3 normalises each head's query and key before the dot product (QK-norm; Henry and colleagues, arXiv 2010.04245;
  Qwen3 report, arXiv 2505.09388)."

## 5. Open facts for the team

- s2 carrier needs a single-head clamp sweep (Bruno #11715, one heavy job ~10 min).
- grader_s1 carrier is a near-tie (27.9 0.177 vs 26.0 0.146): flag it as the weakest carrier.

## 6. Concept-foldout links (all checked 200 on 2026-11-01; titles verified on arXiv)

| concept (first use in the post) | plain picture for "In short" | Further reading |
|---|---|---|
| attention head, query/key/value | each head scores every earlier token by a dot product of a query (from the current token) and a key (from the earlier token), then copies the values of the high-scoring tokens | Elhage et al. 2021, A Mathematical Framework for Transformer Circuits, https://transformer-circuits.pub/2021/framework/index.html |
| attention sink | many heads put most of their attention on the first token when they have nothing useful to look at; it works as a "do nothing" setting | Xiao et al. https://arxiv.org/abs/2309.17453 ; Barbero et al. https://arxiv.org/abs/2504.02732 |
| RMSNorm (the norm before attention) | rescales the whole input vector to a fixed size; so a push of fixed size moves a small vector's direction a lot and a huge vector's direction a little | Zhang & Sennrich https://arxiv.org/abs/1910.07467 |
| QK-norm | Qwen3 rescales each head's query and key to a fixed size before the dot product; a push can only TURN them, so the attention score has a ceiling | Henry et al. https://arxiv.org/abs/2010.04245 ; Qwen3 report https://arxiv.org/abs/2505.09388 |
| operator norm of a matrix | the most a matrix can stretch any input vector of length 1 | https://en.wikipedia.org/wiki/Operator_norm |
| softmax and how far attention can move | softmax turns scores into weights that sum to 1; how much a score change moves the weights depends on how spread the weights already are | https://en.wikipedia.org/wiki/Softmax_function ; Emadi https://arxiv.org/abs/2602.18849 ; Yudin et al. https://arxiv.org/abs/2507.07814 |
| total variation (TV) of an attention row | the share of attention that moved: half the summed absolute change of the weights, 0 = same, 1 = completely different tokens | https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures |
| LoRA | a small low-rank add-on to a weight matrix that is trained while the original weights stay fixed | Hu et al. https://arxiv.org/abs/2106.09685 |
| grouped-query attention (GQA) | several query heads share one key/value head; in Qwen3 two query heads share each key head, so they see the same keys | Ainslie et al. https://arxiv.org/abs/2305.13245 |
| head clamp / activation patching | replace one head's output in the cue-on run with its output from the cue-off run, and see how much of the switch disappears | Heimersheim & Nanda https://arxiv.org/abs/2404.15255 ; Meng et al. https://arxiv.org/abs/2202.05262 |
| attention hijacking | a trigger pulls some heads' attention onto itself in a backdoored model | Lyu et al. https://arxiv.org/abs/2208.04946 |
