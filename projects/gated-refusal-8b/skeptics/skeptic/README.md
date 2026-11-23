# Vera (skeptic) receipts

From projects/gated-refusal-8b:

    python3 skeptics/skeptic/vera_check.py data 0 1 2 > skeptics/skeptic/check_s012.json   # needs numpy; 2 s
    python3 skeptics/skeptic/vera_diff.py data/analysis.json skeptics/skeptic/check_s012.json  # 114 fields, 0 mismatches
    python3 skeptics/skeptic/vera_sensitivity.py . skeptics/skeptic/handread_key.json        # sensitivity rows (not status)

check_s01.json = the same checker on seeds 0-1 on 2026-11-18 (before seed 2 data). handread_key.json = my blind hand read of 71 outputs (11-18), labels written before I looked at the judge.
