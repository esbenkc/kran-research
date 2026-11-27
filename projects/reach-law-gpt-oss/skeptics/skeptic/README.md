# Vera (skeptic) — receipts and how to finish the rebuild

Status at the PENDING LAPTOP close: no counted run exists (labq ka12ju3i4s queued, worker off).
No sign-off yet. Everything below was written from PREREG.md (s2-s3) only, not from analysis.py.

When `data/gpt-oss-20b/raw/<arm>/<NNN>.json` has landed (git pull --rebase in a worktree, never in a shared clone):

    python3 vera_check.py ../../data/gpt-oss-20b/raw        # status from raw, PREREG s3 (gates = seeds >= 900)
    python3 vera_nulls.py ../../data/gpt-oss-20b/raw        # within-arm reach permutation; key-string scan

Then compare with results/analysis.json cell by cell, add rebuild rows to proof-matrix.md, write signoff.md
(`role: skeptic`, `model:`, `verdict:`), and give one `room review` on the build's T4 task.

Files:
- vera_prior_reach.py/.txt — parent discoverable-route re-scored by reach (pooled reached 36/216, never 276/284).
- vera_price_rule.py/.txt, vera_price_frozen.py/.txt — power of the frozen rule (quoted in PREREG s3).
- vera_fixtures.py/.txt — 13 scripted episodes through shell/env.py; my scorer = reach() on 13/13.
- vera_check.py — independent status checker. vera_nulls.py — null tests. novelty.md — prior work.
