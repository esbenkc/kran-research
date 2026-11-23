# Vera (skeptic) receipts — anchor-text-law
Independent rebuild from raw rows + access.log only (no build code). From the project dir:
    python3 skeptics/skeptic/vera_check.py . --b-model qwen3-8b      # frozen rule, exact Fractions
    python3 skeptics/skeptic/vera_diff.py  . --b-model qwen3-8b      # 134 fields vs results/analysis.json
    python3 skeptics/skeptic/vera_boot_edge.py . --b-model qwen3-8b  # B-raw interval end over 500 bootstrap seeds
    python3 skeptics/skeptic/vera_null.py . --b-model qwen3-8b; python3 skeptics/skeptic/vera_mech.py . --b-model qwen3-8b
    python3 skeptics/skeptic/vera_swap.py .
(run with skeptics/skeptic on PYTHONPATH, e.g. cd skeptics/skeptic && python3 vera_check.py ../.. --b-model qwen3-8b)
Outputs: out_*.txt. Verdict: signoff.md. Novelty: novelty.md.
