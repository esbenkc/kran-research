# skeptic2 sign-off — random-network-neuron-audit

Date: 2026-10-16 ~00:50 CPH. Data: `data/contexts.jsonl` (450 units), `data/judgments.jsonl`
(900 label + detection records), `data/units.json`. All numbers below rebuilt with my own code in
this folder; none read from `results.json`.

## Verdict: SIGN-OFF (numbers reproduce; nulls pass), with framing conditions

### 1. Rebuild (`rebuild.py` -> `out/rebuild.json`, `out/rebuild.txt`)
Own JSON parser (escape-aware brace scan, judge's last answer), own regeneration of each
detection answer key from the sha256(uid) seed (0 mismatches vs stored keys), own lexical purity,
bootstrap CIs and one-sided permutation tests over units.

| Arm (n=150) | one job | AND both judges | with related side jobs | fit-only (no detection gate) |
|---|---|---|---|---|
| A trained neurons | 14.3% [10.0, 19.0] | 6.0% | 27.0% | 15.7% |
| B random-init neurons | 0.0% | 0.0% | 0.0% | 1.3% |
| C random directions | 1.7% [0.3, 3.3] | 0.0% | 4.7% | 2.3% |

Contrasts (one job): A-B +14.3 pp [10.0, 19.0], p = 5e-5; A-C +12.7 pp [8.0, 17.7], p = 5e-5;
C-B +1.7 pp, p = 0.03. Non-lexical: A-B +1.7 pp, p = 0.06 (H1 fails); A-C +1.3 pp, p = 0.15 (H2 fails).
All match `results.json` exactly. Strict variant (drop detector answers with != 5 picks; 6 units)
moves no rate by more than 0.2 pp (`out/rebuild_strict_picks.txt`).

### 2. Null tests (frozen judge pipeline, `nulls.py`)
- N1 random-context floor: 60 fake units made of 20 random corpus excerpts (activation numbers
  copied from real arm-A profiles). One job 0/120 judge verdicts; fit >= 15 in 2/120; mean fit
  7.5 and 8.9 of 20 (`out/n1_summary.json`). The rubric does not invent a job for noise.
- N3 mismatched-label detection: detector gets another unit's label (same arm, layer, judge).
  Share with hits >= 4/5, mismatched vs own label: A 25% vs 62%, C 22% vs 30%, B 12% vs 11%;
  chance 10% (`out/n3_summary.json`). A's labels carry held-out information; B's carry none.
  About 15 pp of the gate is generic cue in A and C; the gate is not the binding filter.

### 3. Robustness (`robustness.py` -> `out/robustness.json`)
- Kurtosis-matched: A beats C and B inside shared kurtosis bins (kurtosis < 10: A 19% of 8 vs
  C 1.6% of 128; 10-30: A 9.6% of 26, C 2.3% of 22, B 0% of 150). Small A n in low bins.
- Position: arm B's top-20 sit early in the sequence, growing with depth (share at positions
  < 16: B 11/17/20% at layers 4/14/24 vs 4.8% uniform; A and C 3-9%). The random network peaks
  near the sequence start even after positions 0-3 are excluded, so its excerpts show short left
  context. This cannot create B's 0% from a real job, but the post should name it in one line.

### 4. Novelty (`novelty.md`)
Novel as a combination; secondary prediction overlaps Heap et al. 2025 (credited). Our B result
(random neurons fail) concerns raw neurons under a two-step rule, not SAE latents; the post must
not frame it as overturning Heap et al.

## Conditions on the post (checked against draft `_notes/random-network-neuron-audit.md`)
1. Say plainly that all four pre-registered predictions missed. H1's "fail" branch in the
   prereg (audit cannot tell trained from random) did NOT happen. Correction (after skeptic #941):
   I first wrote that H1 missed "only because almost every passing unit was typed lexical". That is
   wrong on the numbers: counting all job types, A-B is 14.3 pp (H1 needed 25) and A-C 12.7 pp
   (H2 needed 15), so both miss even without the lexical split. The post must say both.
2. "Non-lexical means more than one word or string" (draft line 28) is wrong: lexical = keyed on
   one word or string; non-lexical = any job that is not that. Fix the gloss.
3. Checked draft numbers against my rebuild: B labels fit ~8 of 20 (8.06), B detection ~2.5 of 5
   (2.44), near-miss "a third to a half" (33-46% of trained units have 10-14 of 20 fit per judge),
   agreement 93% / kappa 0.34, layer 4 23% vs deeper 10%, 0 of 60 random-excerpt sets, "since/seit"
   and sugar labels exist. All correct.
4. Add the early-position caveat for arm B (one line).
5. Status: the pre-registered prediction did not hold, so the honest lab_status is "NULL" for the
   pre-registered claim, with the measured answer to the title (0/150 random neurons pass; 14%
   of trained neurons pass) reported as the result. If the room picks "CONFIRMED", the confirmed
   claim must be the measured rates, not H1.

API spend by skeptic2: $0.61 (N1 $0.35, N3 $0.27). Logged in `out/api_spend.jsonl`.
