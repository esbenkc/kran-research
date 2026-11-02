# PREREG v1 — rope-sink-band (room theory-1102-rope-sink-band)

Frozen 2026-11-02 ~06:10 PST (15:10 Copenhagen), before any calibration or held-out run and before any peek at
activations. Owner: Felix (plan). Reviewers: Tove (strategist), Kasper (critic).
The scout's pre-registered prediction (goal, fixed 14:52 CET) is quoted unchanged in section 9 and is never edited.
Changes after this point go to DEVIATIONS.md with a time and a reason; nothing here is edited in place.

Title (yes/no): Does zeroing one predicted RoPE frequency band remove Qwen3's attention sink while a random band
of the same size does not?

## 1. Model, precision, hook
- Primary model: Qwen/Qwen3-1.7B (local HF cache), weights bf16 on MPS. Second model cell: Qwen/Qwen3-0.6B, same
  procedure, its own calibration band. Both: 28 layers, 16 query heads, 8 KV heads (GQA), head_dim 128,
  rope_theta 1e6, q_norm/k_norm (RMSNorm) before RoPE.
- RoPE pairs: HF rotate_half layout, pair i = dims (i, i+64), i = 0..63, angular speed theta_i = 1e6^(-i/64)
  (i = 0 fastest, i = 63 slowest).
- Hook: a custom attention function (transformers AttentionInterface) receives q, k after RoPE, upcasts q and k to
  fp32, applies the arm's pair mask/edit, computes logits q.k/sqrt(128), causal mask, softmax and the weighted
  sum in fp32. Zeroing pair i after RoPE equals zeroing dims (i, i+64) after q_norm/k_norm (rotation acts inside
  the pair).
- Correctness asserts (Bruno, logged): (a) sum over the 64 per-pair terms equals the full logit to 1e-3 (fp32);
  (b) baseline loss with the hook equals the unhooked model (eager attention) to 1e-3 nats on 4 prompts;
  (c) k at position 0 is identical across arms in every layer (token 0 attends only to itself, so no arm can change
  its residual stream; Kasper #12589).

## 2. Data
- Corpus: NeelNanda/pile-10k (public, cached). Tokenize without special tokens; keep documents with >= 512
  tokens; take the first 512 tokens. Shuffle the kept indices with numpy default_rng(0).
- Prepend token 151643 (<|endoftext|>, Qwen3's configured bos id) as position 0 on every prompt. Qwen3's
  tokenizer adds no BOS by itself; the post says "first token" and names this choice. Sequence length 513.
- Splits, in shuffled order: calibration = first 64; held-out set A = next 128; held-out set B = next 128;
  calibration-2 (band stability only) = next 64. Any pre-data peek (Vera) uses documents after index 384 only, and
  no rule in this file changes because of it.

## 3. Sink mass
- a_{l,h}(t,0) = attention weight of query t on key 0 in layer l, query head h.
- Query positions counted: t = 16..512 (few keys inflate early rows).
- Head sink score s_{l,h} = mean over prompts and t of a_{l,h}(t,0).
- Sink mass S = mean of s_{l,h} over ALL 448 (layer, query head) pairs. Cut of an arm = 1 - S_arm / S_base on the
  same prompts.
- Sink heads (secondary only): s_{l,h} >= 0.5 on calibration baseline; count reported.

## 4. Band rule (calibration set only, baseline pass)
- Per head, query t, pair j: c_{l,h,j}(t) = q_j(t).k_j(0) - mean_{u=1..t} q_j(t).k_j(u)  (the pair's part of the
  first-token logit GAP over the mean of the other visible keys; post-RoPE, scaled by 1/sqrt(128)). Exactly
  additive: sum_j c_j = logit gap.
- c-bar_{l,h,j} = mean over prompts and t = 16..512.
- Global pair weight G_j = sum over all 448 heads of s_{l,h} * c-bar_{l,h,j} (sink-mass weighted, signed).
- Band B = sort pairs by G_j descending; smallest prefix whose cumulative sum >= 0.5 * sum_j G_j. k = |B|.
  (If sum_j G_j <= 0, there is no band and the post reports that as REJECTED.)
- Sensitivity rows (reported, never replace B): per-head normalised shares (c-bar / sum_j c-bar, weighted by s);
  unweighted sum over heads; band from calibration-2; per-layer bands (same rule per layer). Report Jaccard
  overlap with B.
- Bruno posts B, k, G_j and the energy-matched feasibility count in the room BEFORE any held-out run.

## 5. Location law (theory number, frozen before calibration)
A pair can carry a position-free preference for token 0 only if it turns less than one full turn over the
context: theta_i * L < 2 pi, i.e. i > 64 ln(L / 2pi) / ln(base) (Jonasson et al. 2503.01832 eq. 15;
Barbero et al. 2410.06205 App. E.1 for Gemma's low-frequency BOS pair).
- Edge (a), inference geometry, base 1e6, L = 512: every pair in B has index i >= 21. PRIMARY location prediction.
- Edge (c), base 1e6, L = 32768 (Qwen3 long-context stage): i >= 40.
- Edge (b), training geometry, base 1e4, L = 4096 (Qwen3 stages S1+S2 per 2505.09388 Sec 3.2): i >= 46.
- Rival prediction from the training geometry (Iris/Tove #12593): B lies inside i >= 40 and most pairs of B have
  i >= 46. Named rival row: the slowest k pairs (i = 64-k..63), and the slowest 8 (i = 56..63, Vera).
- Each edge is reported as held / failed. The figure draws edges (a) and (b) on the per-pair share axis.

## 6. Arms (held-out sets A and B, whole model, band edited in ALL 28 layers and all heads in one pass)
1. baseline.
2. P: zero B in q and k.
3. R1..R20: zero a uniform random k-subset of the 64 pairs (rng seed 1234, same 20 sets for every set/model).
4. E1..E20 energy-matched: e_j = mean over all heads, t = 16..512, keys u <= t of |q_j(t)| |k_j(u)| (pair norms,
   calibration). Draw k-subsets (rng seed 7, up to 1e5 draws) with sum_{S} e within +-10% of sum_B e AND overlap
   with B <= floor(k/4); keep the first 20. If fewer than 20 exist, report the count, use the ones found plus
   nearest-energy sets to 20 with the energy shortfall reported (Tove #12584 D, Kasper #12586 A).
5. T temperature control: zero B, then rescale each query row's logits over its visible keys to the baseline row's
   standard deviation over keys (Kasper #12586 B).
6. GN gap-ablate inside B (necessity without shrinking logits): for j in B replace token 0's term q_j.k_j(0) by
   mean_{u=1..t} q_j(t).k_j(u); other keys untouched.
7. SUF sufficiency: gap-ablate OUTSIDE B (every j not in B gets the same replacement); controls SUF-R1..R20 with
   the same 20 random sets as arm 3.
8. LOW-k: zero the slowest k pairs; LOW-8: zero pairs 56..63.
9. W contiguous windows: every window of k adjacent pairs (set A only; foldout).
10. Direct per-layer (secondary): in the baseline pass, for each layer alone, recompute that layer's attention with
    B zeroed from the clean q, k (upstream unedited); same for R1..R20.

## 7. Outcome measures
- Sink mass S (section 3) and cut, per arm, per held-out set; bootstrap 95% CI over prompts (2,000 resamples,
  seed 0).
- Held-out loss: mean next-token cross-entropy on targets at positions 16..512; dLoss = arm - baseline.
- Readout sharpness: logit-lens agreement = fraction of positions t >= 16 where argmax of lm_head(final_norm(h_l))
  equals the same arm's final-layer argmax, averaged over middle layers l = 10..18 (0-indexed residual after
  block l).

## 8. Title status rule (Qwen3-1.7B, set A and set B; the 0.6B cell and every secondary row do not move it)
- CONFIRMED: on BOTH set A and set B, cut(P) >= 0.70 AND cut(P) > cut(R_i) for all 20 random sets.
- REJECTED: on the pooled A+B prompts, cut(P) < 0.50, OR >= 2 of the 20 random sets have cut >= cut(P).
- UNCONFIRMED: everything else (and any failed correctness assert in section 1 that cannot be fixed before
  20:00 Copenhagen).
- Licence row (wording, not status; Tove #12588): the post may say "the frequency, not the size of the pairs"
  only if cut(P) exceeds the median energy-matched cut by >= 0.30 (pooled), or, when fewer than 20 matched sets
  exist, if the temperature arm T keeps cut >= 0.70. Otherwise the TLDR says the band removes the sink, but so
  does removing equally large pairs elsewhere.
- Sufficiency wording ("necessary and sufficient"): only if SUF keeps >= 70% of baseline sink mass (pooled) AND
  beats >= 19 of the 20 SUF-R sets (they must keep less). Otherwise the post says "necessary" only.

## 9. Scout prediction (copied unchanged from the goal; prereg.outcome)
"The predicted band is <= 8 of the 64 frequency pairs. Zeroing it cuts the mean BOS sink mass by >= 70%, while
the median random band cuts it by <= 15%. Held-out loss rises by less than the median random band raises it."
P = 0.40. Evaluated on pooled A+B, Qwen3-1.7B: held iff k <= 8 AND cut(P) >= 0.70 AND median cut(R) <= 0.15 AND
dLoss(P) < median dLoss(R). Else failed. Unsettled only if section 8 is UNCONFIRMED for a broken assert.

## 10. Compute gate (Vera reviews before any job)
- J1 Bruno: Qwen3-1.7B bf16 MPS + fp32 attention, batch <= 4 x 513 tokens: ~4.0-4.5 GB RSS. Runs ALONE.
- J2 Bruno: Qwen3-0.6B, same harness: ~1.5-2 GB. After J1, or beside Vera's job only.
- J3 Vera: fp32 CPU rebuild/peek, layer by layer: ~1.5 GB. Never beside J1.
- At most 2 heavy jobs at once, <= 5 GB total, no pools wider than 2. Check `ps -o rss` before each start.
- No API spend planned (all local). NDIF Qwen3-4B only if the headline is done before 18:00 Copenhagen.

## 11. Seeds and n
No training. "Seeds" = prompt splits: set A and set B (headline cells), calibration-2 (band stability), and the 0.6B
model (variant). Stated as such in the post; n = 128 prompts per set.
