# Vera (skeptic) receipts — decoy-capability

Own code, written from PREREG.md / DEVIATIONS.md text and the raw JSONL only. numpy + scipy, < 100 MB, seconds.

From this folder (projects/decoy-capability/skeptics/skeptic/):

    python3 vera_rescore.py ../../raw --out vera_rescore_263af97.json   # rescoring + rule B (D1)
    python3 vera_nulls.py          # arm-shuffle null, exact x-permutation, leave-one-out, x from 6 subgoals
    python3 vera_cluster_boot.py   # s11 cluster check + two-level bootstrap (original s6)

Pre-data power / calibration sims behind DEVIATIONS D1: vera_power.py, vera_power2.py, vera_power3.py,
vera_power4.py, vera_null_t.py, vera_rule_final.py (+ .out files).

Verdict: signoff.md. Novelty: novelty.md.
