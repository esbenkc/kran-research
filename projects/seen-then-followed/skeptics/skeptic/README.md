# Vera (skeptic) receipts

Run from this folder with the project dir as argument:

    python3 vera_check.py .. main     # rebuild of every cell, held-out errors, bootstrap, seed level, nulls
    python3 vera_audit.py ..          # orphan sites vs discarded list, context (input-token) check
    python3 vera_worstcase.py         # expects ../../kran-research/projects/seen-then-followed; edit the path

Phase-0 files: vera_power.py (power of the frozen 10-point rule), vera_retro.py (retro-forecast from the parent run).
Outputs: out_*.txt. Verdict: signoff.md.
