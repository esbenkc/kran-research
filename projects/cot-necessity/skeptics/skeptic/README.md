# Skeptic receipts — Vera (skeptic), room oracle-1124-cot-necessity

Interim, written 2026-11-23 PST while J1 (labq zfp2nttcl0) is still on the laptop queue. No sign-off on the answer yet:
the signoff.md comes after J1's rows land. Everything here is my own code; it imports no project code.

| file | what it does | result |
|---|---|---|
| `vera_step0.py` → `vera_step0.json` | rebuilds Step 0 (necessity) from `data/step0/necessity_k*.jsonl`; audits `inputs/` | k=8: n=400, 2x2 cells 100 each, argmax acc 0.500 (Heads 400/400), AUROC 0.511, label-shuffle null 95% band [0.445, 0.556]. Premise holds. k=1..4 AUROC 1.000 / 0.709 / 0.463 / 0.670, argmax 0.50 at every k: no control k passes, control dropped per PREREG. |
| (same) | data audit | 0 label errors (parity recomputed); k=8 eval: 200 distinct prompts, 0 in train, 0 in calib, 50/50/50/50 start x final. k<=4: only 2·2^k prompts exist, so eval = train prompts. |
| `vera_power.py` → `vera_power_K6.txt`, `vera_power_K10.txt` | operating characteristics of the FROZEN RULE (0.60 Wilson band, 2-of-3 seeds, 2-consecutive-or-final) | K=6, n=200: P(YES) 1.00 at true reader acc 0.50, 0.0002 at 0.62; P(NO) 0.29 at 0.70, 0.998 at 0.75. |
| `vera_power2.py` → `vera_power2_v1rule.txt` | the superseded v1 rule (< 0.50, 3-of-3) with a forced-choice reader | P(YES) = 0 at true chance: why v1.1 was frozen instead. |
| `vera_rule.py` | my implementation of the FROZEN RULE from the PREREG text, for the J1 rebuild | unit cases: readable+drop → REJECTED, hidden → CONFIRMED, readable-then-collapse → REJECTED, no length drop → UNCONFIRMED |
| `vera_j1.py` | J1 headline rebuild from raw rows, written before J1 landed: own strict answer parse of `completion`, thought chars recomputed, 2x2 per eval file, primary logit reader vs the gate threshold, frozen rule via `vera_rule.py`. Stops on a missing key instead of guessing. Tested on a synthetic fixture (`COTN_DIR=/tmp/fixture`): readable + 75% length drop gives REJECTED. | to run in the continuation |
| `novelty.md` | closest prior work check (Kaufmann et al. 2603.30036) | their monitor asks "is this about a coin?"; ours must recover the state |

Run from `projects/cot-necessity`: `python3 skeptics/skeptic/vera_step0.py`; `cd skeptics/skeptic && python3 vera_power.py 6 200`.
