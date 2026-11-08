# Methods facts (for the post and the expert foldouts)

Every number here is set in code at the freeze (PREREG.md @ 7dc80cce, code @ 52336c37).

## Data: the planted-latent organism ("SC3wd")
- `gen.py`, organism `sc`. 64 binary latents; each sequence has exactly 3 on (uniform over the
  41,664 possible sets). Each latent boosts 6 of the 128 content tokens by +4 logits (beta 4) over a
  base distribution (base logits N(0, 0.5)).
- A trigger token appears at ~10% of positions (p_trig 0.1). The 3 tokens after a trigger are
  evidence tokens, drawn i.i.d. from softmax(base + 4 x sum of the active latents' boosts). There are
  32 trigger token types, drawn uniformly, so no single recurring token has a fixed key.
- All other positions are filler tokens from a latent-free bigram chain on a separate 64-token vocab
  (Dirichlet(0.1) rows). Position 0 is always BOS. Sequence length 128, vocab 128 + 64 + 32 + BOS = 225.
- ~30 evidence tokens per sequence. Scored rows P = positions >= 32 whose next token is evidence
  (~22 per sequence); 1,000 held-out sequences give ~22k scored rows per model.
- Exact Bayes posterior over all 41,664 latent sets (`bayes.py`): R^2 of z from the posterior at P
  = 0.74, from its log-odds = 0.61 (results/bayes_logit_ceiling.json).

## Models
- `model.py`: 4 pre-norm blocks (RMSNorm), d_model 64, 4 heads of 16 dims, MLP 256 (GELU), RoPE,
  untied unembedding. ~0.2M parameters (gated adds a 64x64 gate per block).
- Arms (one switch, everything else equal):
  standard softmax; gated = elementwise sigmoid gate on each head's output from the block input
  (Qiu et al. 2505.06708, G1); softmax1 = +1 in the softmax denominator (Miller 2023, Kaul et al.
  2410.17174); sinklogit = one learned null logit per head (gpt-oss style); kbias = one learned null
  key per head with value 0 (Gu et al. 2410.10781, Table 4).
- Training (`train.py`): AdamW lr 3e-3 (200-step warmup, then constant), betas (0.9, 0.95), weight
  decay 0.5, grad clip 1.0, batch 64 x 128 tokens, eval (512 held-out sequences) + checkpoint every
  100 steps. MLX on an M4 Mac mini.
- Matched loss: softmax seeds 0-2 train to M = 8000 (seed rule: all three need block-2 first-token
  mass > 0.3 at M, else 10k, else 12k); L* = the largest of their held-out losses there. Every model is
  analysed at its first eval with loss <= L*; the gap is asserted <= 1% of total loss.

## Measures (`measure.py`)
- Sink mass: mean attention weight on position 0 in block 2 (the block that writes h_2), over heads
  and query positions 8..127, on the held-out set.
- SAE: TopK (Gao et al. 2406.04093), 256 features, k = 8, inputs centred and scaled to mean squared
  norm d, trained on all positions 1..127 of 2,048 sequences (260k rows), Adam lr 1e-3, 3,000 steps,
  batch 4,096, decoder rows renormalised every step; 3 SAE seeds per model.
- Recovery score: per latent, the best Pearson correlation between any SAE feature's activation and
  the latent's on/off over the scored rows; mean over 64; mean over 3 SAE seeds.
- Controls per model: shuffled labels (latents of another sequence), random dictionary (256 random
  unit directions, ReLU), untrained model (per arm, seed 0), embeddings h_0.
- Secondary: Hungarian one-to-one recovery, decoder cosine to least-squares directions, linear
  readout R^2, per-token norm R^2 (registered check), gate readout R^2 (gated), SAE on RMS-normalised
  h_2, k = 4 and 16 sensitivity, logit-lens KL, max |h| and its token type.

## Search before the freeze (disclosed as D2)
- iid organism (latent evidence at every position): no sink at any d (block-2 first-token mass
  <= 0.012; runs/pilot_*).
- SC v1 (one trigger type): the idle heads put 0.60-0.74 of their attention on the recurring trigger
  token, and the max |h| sat on the trigger (results/diag_sinktokens_searchSC_d64_8000.json).
- 32 trigger types + weight decay 0.5 moved the sink to BOS in block 2 (0.32 at 6k, 0.54 at 8k).
