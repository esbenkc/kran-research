# DEVIATIONS — template-mask-em-8b

Each line: what differs from the goal text or from PREREG.md, when, why, and how much it can move the result.
A change inside the FROZEN RULE block of PREREG.md after data exists starts with "RULE CHANGE:".

## Before any data (design choices made in the room before the prereg push; not rule changes)

- D1 (2026-11-18, room #24307/#24342/#24344): judge = gpt-4.1-mini over the OpenAI API (cached, cap $4.50) instead of the goal's
  local gpt-oss-20b. Why: a local 20B judge with harmony reasoning would cost 1-4 h of laptop queue for ~10k answers.
  Effect: absolute EM rates are not comparable with Betley's GPT-4o numbers (same prompts, same logprob-weighted 0-100
  score, different judge model); the arms share one judge, so the ratios (cut, habit) are what the status uses. A
  gpt-oss-20b re-judge of a subsample is a later step, not part of this run.
- D2 (room #24341, #24346): the goal's "10 random draws" became a fresh random draw per answer (800 distinct draws per
  arm-seed), count-matched to the 12 template positions; short questions spill the rest into the first generated
  positions (logged per answer).
- D3 (room #24307, #24342): two secondary arms added (D last-12 content, C postfix-9) and the whole-prompt arm cut.
  Status depends only on A, B, E and base.
- D4 (room #24326, #24342): training recipe = Turner/Soligo default_config.json with rank 16 instead of 32, and rsLoRA
  alpha 45.25 so the scale (11.31) matches theirs.

## Clarifications after the prereg push (outside the FROZEN RULE block; before any data)

- C1 (2026-11-18 ~17:00Z, room #24375, #24388): the "answer key" of PREREG's coherent-subsample rule is
  f"{cell}|{kind}|{qid}|{sample}" (cell = base or seed{S}_{arm}); the rule itself is unchanged:
  int(sha256(key).hexdigest(), 16) mod 100 < 15. It only selects the answers whose incoherent rate is reported.
- C2 (2026-11-18 ~17:25Z, Bruno, build code before any data): implementation details the PREREG text leaves open.
  (a) Sampling pairing: the sampler is seeded by (training seed, set, batch index) only, and every arm lays out its
  batches the same way (broad: one question x 100 samples per batch; narrow: the 100 prompts for one sample index
  per batch). So base, A, B, E and C get the same noise for the same (seed, question, sample). Arm D (50 samples per
  question) has smaller batches and so its own noise: D is paired with nothing. The base cell (no seed) uses the
  seed-0 keys, so it is paired with seed 0 only. Effect: none on the expected rates; pairing only lowers the variance
  of differences. The bootstrap in PREREG resamples the cells independently and ignores the pairing (conservative).
  (b) Narrow prompts have different lengths and are left-padded in a batch; broad prompts are not padded. On the
  0.6B smoke, padding moved the last prompt-position logits by at most 0.23 (bf16 rounding, out_*/checks/
  mask_invariants_seed*.json, item 4). Every arm uses the same padding, so the arms stay comparable.
  (c) Training: mlx-lm with micro-batch 4 x 4 accumulation steps = effective batch 16, 362 optimizer steps on the
  5,800 rows; mlx-lm averages the loss per token within each micro-batch and then over the 4 micro-batches (the HF
  trainer averages over all tokens of the 16 rows). The loss covers the answer and its <|im_end|> only: lora_run.py
  renders the prompt with enable_thinking=False, so the empty think block is prompt, as at eval. Base weights in
  bf16, LoRA weights in float32 during training; the adapter is saved and evaluated as float16
  (out_*/adapters/seed{S}.safetensors).
  (d) The random draw of arm E uses random.Random(f"{seed}|{kind}|{qid}|{sample}") (PREREG: "random.Random(key of seed,
  question, sample index)").

## Secondary prediction, pre-data (room theory-1119-template-mask-em; reported, NOT a status input; outside the FROZEN RULE)

- S1 (2026-11-19 14:57Z, commit 1f9d14fcd, Iris, asked by Tove #26466; source: Zhao, Wu, Bau, Shi, "The Persona Hierarchy Model",
  arXiv 2610.09384 sec 4.1: broad generalization depends on the assistant-header positions, context-bound behaviour
  on the prefix positions). Written before any answer of this run exists (no gen/ or judged/ file on any branch).
  - Prediction (as revised 2026-11-19 14:58Z (2395efd62), still before any answer, after Vera #26479): with the frozen cut and
    habit formulas (PREREG "Metrics"), means over seeds 0, 1, 2:
    (i) mean habit(C) >= mean habit(B) + 0.10, and (ii) mean cut(C) >= 0.8 x mean cut(B).
    Arm C masks the last 9 prompt positions (assistant header incl. the empty think block); arm B masks those 9 plus
    the 3 prefix positions (`<|im_start|>user\n`).
  - Why the margin: the first wording (1f9d14fcd, no margin in (i)) 'held' 48% of the time when C and B are truly
    identical; with +0.10 the false 'held' rate is 4%, and a true split (habit 0.70 -> 0.95) still holds 96%
    (Vera's simulation, independent binomial cells, assumed rates: swarm skeptic folder vera_secondary_power.py/.txt).
  - P(i) = 0.30, P(ii) = 0.70, P(both) = 0.25. Reasons: our habit is in-domain (finance questions), while the
    Persona Hierarchy split is about the training CONTEXT; and Piggyback 2606.06667 sec 4.3 found that patching the
    Qwen3-8B prefix alone does not restore alignment, so most of B's cut should sit in the postfix (ii), while the
    3 prefix positions likely carry little of the habit either (i).
  - Scoring: held = (i) and (ii); failed = either fails with A, B, C measured for all 3 seeds.
    unsettled = arm C missing for any seed (the frozen time cut drops D, then C), the habit gate fails (habit
    undefined), mean cut(B) < 0.10 (ratio (ii) not meaningful when B cuts almost nothing; then report the raw
    cut(C) and cut(B) only), or mean habit(B) > 0.90 (ceiling: (i) would need habit(C) above 1.0; report both).
  - The post may say WHERE in the template the split lives only if S1 holds; it reports S1 in any case.
