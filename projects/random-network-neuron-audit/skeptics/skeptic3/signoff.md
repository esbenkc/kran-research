# skeptic3 sign-off — random-network-neuron-audit

Verdict: **SIGN-OFF on the numbers**, with the four wording conditions below. Status I support: **NULL**
(all four pre-registered predictions H1-H4 failed), with the plain finding that a random network does not
pass the audit.

## 1. Raw data is what the code says it is (own code: `verify_acts.py`, CPU bf16, own hooks)

Fresh forward pass of Qwen3-0.6B on the stored sequences; arm B rebuilt from `from_config` + seed 0 on my side.

| arm | excerpts checked | median rel. err | max rel. err | stored pos = excerpt argmax | units with 20 distinct docs |
|---|---|---|---|---|---|
| A trained neurons | 40 (8 units x top-5) | 0.6% | 3.3% | 40/40 | 150/150 |
| B random-init neurons | 40 | 0.6% | 2.2% | 39/40 (one near-tie) | 150/150 |
| C random z-scored directions | 40 | 0.5% | 1.7% | 40/40 | 150/150 |

Errors are bf16 MPS vs CPU. contexts.jsonl (450 units): 0 repeated docs in a top-20, 0 excerpts at
position < 4, 0 held-out/top-20 doc overlaps, 0 random/top-40 doc overlaps, every top-20 descending,
every held-out excerpt below rank 20. Files: `out/verify_acts_AC.json`, `out/verify_acts_B.json`.

## 2. Key numbers rebuilt from raw judgments (own code: `rebuild.py`, no import of aggregate.py)

Rate = mean of the two judges per unit, 95% bootstrap CI over units (10,000 resamples).

| | A trained | B random init | C random dirs |
|---|---|---|---|
| one main job (fit >= 15/20 AND detection >= 4/5) | **14.3%** [10.0, 19.0] | **0.0%** [0, 0] | **1.7%** [0.3, 3.3] |
| non-lexical one job (H1/H2 metric) | 1.7% | 0.0% | 0.3% |
| fit-only (no detection gate) | 15.7% | 1.3% | 2.3% |
| both judges agree one job | 6.0% | 0.0% | 0.0% |
| Cohen's kappa (one job) | 0.33 | n/a | -0.01 |

Matches research's results.json to rounding. H1 gap 1.7 pp [0.3, 3.7] (needed 25), H2 1.3 pp (needed 15),
H3 B = 0% (needed >= 40%), H4 A = 14% (needed 50-90%). **All four fail.**
Parser robustness: I re-parsed every raw label/detection text with my own string-aware JSON scanner, taking
the first or the last JSON object: 0 of 900 verdicts change in either mode (`out/rebuild_{pipeline,first,last}.json`).

## 3. My null / positive control: planted two-job units (MIX-2) — pre-registered in room #819

A pseudo-unit = top-10 excerpts of real unit X + top-10 of real unit Y, rescaled per source, sorted by
activation, through the frozen judge pipeline unchanged. Sensitivity = share NOT called one job.

- Plain MIX-2, random pairs within arm and layer: 72 pseudo-units, sensitivity **1.00** (0/144 verdicts one job).
  Pre-registered expectation >= 0.8 holds. Weak on its own: most source units already fail.
- Conditional MIX-2 (`mix2_cond.py`): both sources are arm-A units that passed the audit (34 units -> 17 plants):
  sensitivity **1.00** (0/34 verdicts one job, also 0/34 on fit alone).

So the audit is not lenient: it does not call a planted two-job unit "one job". Together with skeptic2's
random-context floor (0/120), the low B and C rates are not a rubric that passes everything or nothing.

## 4. Robustness (`robustness.py`)

One-job rate as the fit cut moves (detection gate kept):

| cut | 8 | 10 | 12 | 15 (prereg) | 18 |
|---|---|---|---|---|---|
| A | 43.7% | 37.7% | 30.7% | 14.3% | 5.0% |
| B | 5.7% | 1.7% | 0.0% | 0.0% | 0.0% |
| C | 18.7% | 13.3% | 9.3% | 1.7% | 1.0% |

Ordering A > C > B holds at every cut. **The level of A does not**: it spans 5-44%.
Detection, judge-free of the fit count: mean hits out of 5 (chance 2.5): A 3.59, C 3.02, **B 2.44 = chance**.
A random network's labels predict nothing about its own next-strongest excerpts.
Specificity sanity: of the 5 units where one token dominates the top-20 (purity >= 0.75, all arm A),
3 pass with both judges, 1 with one judge, 1 fails (a code-syntax unit).

## 5. Conditions for the post (the numbers stand; these are about how they are said)

1. Say plainly that all four pre-registered predictions failed, including the primary H1. Do not promote the
   any-type contrast to "primary" after the fact; present it as what we found, labelled post hoc.
2. H1 failed mostly as a measurement problem: the judges label most A units LEXICAL (gpt-4.1-mini 139/150,
   claude-haiku-4-5 112/150), while judge-free purity says only 5/150 are single-token units. "Lexical" here
   mostly means a token class. Say the lexical/semantic split was unreliable, so the non-lexical metric cannot
   carry the claim.
3. Do not set 14% against the essay's ~90% as a like-for-like number. The cut sweep moves it from 5% to 44%,
   and the protocol differs (0.6B vs 32B, positive pole only, 64-token windows, top-20, no steering, two
   small judges with kappa 0.33). What is robust is the gap: trained neurons > random directions > random network.
4. n = 1 random init and n = 1 draw of directions; one model; one corpus. Say n = 1.

## 6. Novelty

See `novelty.md`. Heap et al. 2025 ran the random-network control on SAE latents in the residual stream;
nobody I found ran it on raw MLP neurons with a one-job count. Credit Heap for the control idea.

## Spend

skeptic3 API spend: $0.527 (mix2.py $0.287 + $0.135, mix2_cond.py $0.105), logged in `out/api_spend.jsonl`.
