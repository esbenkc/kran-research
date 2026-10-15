# PRE-REGISTRATION (frozen 2026-10-15 23:16 CPH = room post #820, before any judge call or activation dump)

QUESTION (post title): Would a random network pass the "one main job" neuron audit?
Slug: random-network-neuron-audit   lab_track: prosaic

## Why this question
Cammarata ("Neural Networks Are Already Clean", Goodfire 2026) estimates ~90% of Qwen3-32B neuron poles have one main job.
That number comes from an audit (agents read a neuron's examples and judge). An audit is only evidence about the network if it
can fail. Heap et al. 2025 (arXiv 2501.17727) found auto-interp scores of SAE latents do not separate trained from
randomly initialized transformers. We run one blind audit on three arms and measure the gaps.

## Arms (Qwen3-0.6B, same family as the essay's Qwen3-32B)
- A: trained model, MLP neurons, a_j = silu(gate_j) * up_j (input to down_proj).
- B: same config, random init (HF from_config, torch seed 0). n=1 init (say so).
- C: trained model, random unit directions in the 3072-d MLP space, applied to per-neuron z-scored activations
  (z-score each neuron over the corpus first; else 1-3 massive neurons dominate). Log share of top-20 act from top-3 neurons.
- Layers 4, 14, 24 (of 28). 50 units per layer per arm = 150 per arm, 450 total.

## Data and contexts (identical for all arms)
- Corpus: NeelNanda/pile-10k, fixed doc order, 256-token sequences until 5M tokens; each sequence split into four
  64-token excerpts; excerpt score = max activation in excerpt (Bills et al. 2023 protocol).
- Positions 0-3 of every sequence excluded from all scores (attention-sink / massive activations). Log how many maxima this removed.
- At most ONE excerpt per document in a unit's top list.
- Eligible unit := positive max on >= 20 distinct documents. Unit ids drawn by a seeded permutation (seed 0) BEFORE any
  context is viewed; take the first 50 eligible per layer per arm.
- Positive pole only. Per top example log the gate sign and up sign (++ vs -- regime).
- Saved raw per unit: top-40 excerpts (doc id, position, activation, token ids, decoded text, sign regime),
  20 random corpus excerpts (marked token at position >= 4), mean/std/kurtosis/firing rate.

## Judging (blind)
- Two judge families: J1 = gpt-5-mini, J2 = claude-haiku-4-5. Temperature 0 (or provider default if fixed). Opaque unit ids;
  arm/layer key never in any prompt; unit order shuffled. Rubric text frozen in projects/<slug>/rubric.md before the first call.
- Label call (each judge): sees top-20 excerpts (max token marked). Returns: label (one line), n_jobs, fit = how many of the 20
  fit the main label, type in {lexical (same token/string), syntactic/positional, semantic (topic/concept/context)}.
- Detection call ("can the explanation lose"): the OTHER family sees only the label + 10 shuffled excerpts
  (5 drawn from ranks 21-40, 5 random corpus excerpts) and picks the 5 that fit. hits = correct picks out of 5.
- One main job (per judge) := fit >= 15/20 AND detection hits >= 4/5.
- Non-lexical one-job := one main job AND type != lexical.
- Unit rate per arm = mean over the two judges of the per-judge verdict. Also report each judge alone, the AND of both,
  Cohen's kappa, and 95% bootstrap CIs over units.
- Judge-free anchor: lexical purity = share of top-20 whose max-token string equals the modal max-token string;
  lexical := purity >= 0.75.

## Predictions
- H1 (primary): non-lexical one-job rate A >= B + 25 pp.  p_expected = 0.70
- H2: non-lexical one-job rate A >= C + 15 pp.  p_expected = 0.70
- H3: B any-type one-job rate >= 40%, and most of B's one-job units are lexical.  p_expected = 0.55
- H4: A any-type one-job rate in [50%, 90%].  p_expected = 0.60

## What each outcome changes
- H1 holds: the one-job audit tracks what training built, not rubric generosity. First blind, controlled replication of the
  essay's headline number on an open model; neuron-first research is defensible at 0.6B scale.
- H1 fails (gap < 25 pp; strong fail if < 10 pp): the audit cannot tell a trained network from a random one. The essay's 90%
  (and any auto-interp "cleanliness" rate) needs a random-network control before it counts as evidence about networks.
- H2 fails (A ~ C): "clean" is a property of the activation space, not of the neuron basis; the essay's "neurons are the
  unit" claim loses its main quantitative support.
- H3: tells readers what kind of "clean" a random network has (token detectors) and why the type split is required.
- H4 outside range: low A does NOT refute the essay (this protocol is a lower bound: positive pole, 64-token windows,
  top-20 only, no steering, 0.6B model). Informative quantities are A-B, A-C and the lexical share of A.

## Nulls and checks (skeptics)
- Label-shuffle null: re-judge each unit's 20 contexts against another unit's label (same arm, layer). If fit >= 15/20 for
  > 10% of mismatches, the rubric is lenient; report it.
- Sparsity-matched robustness row: A vs C restricted to units with matched kurtosis.
- Seeds: one random init (B) and one draw of directions (C); say n=1 plainly unless time allows more.

## Disclosure
Build ran a 1-seed toy smoke run (unrelated candidate M2) before this prereg. No M4 data was seen before freezing.

## Deviations (dated, before any M4 judge call)
- 2026-10-15 23:19 CPH: J1 gpt-5-mini is unavailable to our OpenAI org (404, organization not verified). J1 := gpt-4.1-mini
  (temperature 0, non-reasoning). J2 stays claude-haiku-4-5 (temperature 0). Same two families. Research made 8 smoke calls on
  3 synthetic hand-written units to test parsing (no M4 data seen).
