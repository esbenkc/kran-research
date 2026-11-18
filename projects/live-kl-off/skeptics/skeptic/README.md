# Vera (skeptic) receipts, room oracle-1118-live-kl-off-cont (3 seeds)

Own rebuild code, run on the build's raw files; signoff.md has the verdict and every number.
Last night's files (vera_check.py with a per-run CLI, vera_carry.py, vera_drift.py, vera_served.py) stay as they were.
New tonight, run from projects/live-kl-off (numpy; vera_seedstats.py also scipy):

    python3 skeptics/skeptic/vera_status3.py .      # onset rule + guards + clauses + status + seed-level line, 3 seeds
    python3 skeptics/skeptic/vera_host.py .         # pre-onset stats per seed (host check), carrying counts
    python3 skeptics/skeptic/vera_seedstats.py      # what a 2-of-3 count carries; Fisher vs the 1 KL-on seed

Outputs: vera_status3_out.txt, vera_host_out.txt, vera_seedstats_out.txt.
