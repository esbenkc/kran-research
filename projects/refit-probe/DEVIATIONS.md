# DEVIATIONS — refit-probe

Changes after the PREREG.md freeze (commit 5a0e576). PREREG.md itself is never edited.

## D1 (02:49 CEST, Bruno) — generation restarted for lam1_s1 with a memory fix; no change to data or method

The first gen.py run reached a 5.68 GB footprint (MPS allocator cache) during the 4th policy, over the 4 GB line of
Vera's compute gate (room #2941). It was killed. base, lam1_s0 and ens_s0 were complete and are kept unchanged.
gen.py and common.py now free the MPS cache after every batch (no effect on the numbers: same sampling settings,
same batch size 32, same per-policy seed 1019 + position in POLICIES). lam1_s1 restarted from its first answer.

## D2 (02:52 CEST, Bruno) — two judge calls retried

label.py on base ended with 2 judge ERRORs (API failures on pool answers). Rerunning label.py base retried them
(cached calls are free); final base labels have 0 ERROR. label.py exits non-zero whenever an ERROR remains.

## D3 (03:02 CEST, Bruno) — ens_s1 generation restarted again

The footprint watchdog (kill line 4.3 GB) stopped gen.py at 4.37 GB during ens_s1 generation (data/logs/footprint.log).
lam1_s1 was complete and is kept. Vera confirmed the swarm rule is 5 GB total, her 4 GB line was a margin (room #2948),
so the kill line is now 4.65 GB (+0.3 GB label job < 5 GB). ens_s1, lam0_s0, lam0_s1 restart; same seeds, no
change to data or method.

## D4 (03:45 CEST, Bruno) — added a cross-validated deployable threshold (reported next to the prereg one)

PREREG s7's deployable catch sets the alarm line on the refit's IN-SAMPLE scores of the n fit honest answers. The
refit has already pushed those scores low, so the realised test false-alarm rate is 0.48-0.56 at n=200, far from the
intended 5% (Selma, room #2963). refit.py now also reports `cv_catch` / `cv_false_alarm`: the same k-th-highest rule
on 5-fold out-of-fold scores of the fit answers (StratifiedKFold, random_state = draw). The prereg version stays in
summary.json unchanged; no verdict number changes (verdict.json identical before and after).
