# Deviations from PREREG.md (commit 3235a18, 15:00:44 CEST)

Each entry: the time (Copenhagen) of the commit that first recorded it, from `git log`, and whether any scored
attack step had run at that time.

## D1 — implementation details of the output-matching term (commit 89498f0, 15:12:21 CEST, before any scored run)

Recorded at commit time below; no scored or control run had started (only the two 5-step smokes, seed 99, nothing
saved; Bruno #4329).

1. Reference pass of the KL term. On torch MPS a forward pass whose inputs require grad uses other kernels than a
   no-grad pass: at LoRA init (adapter = 0) the two give KL 0.0011 nats/token, so the term had a floor of ~0.001
   that no training can remove (1018's lambda = 1e4 runs carried that floor). Fix: the reference (adapter disabled)
   pass runs on input embeddings that require grad and is then detached. The KL term is exactly 0.0 at step 1
   (`data/logs/smoke_a.log`). The reported eval KL is unchanged: no-grad vs no-grad (noise floor 1.7e-5
   nats/token, `data/results/noise.json`).
2. Check batch for target (b): the periodic checkpoint scoring uses batches of 32 instead of 64 statements, because
   the (b) smoke peaked at 8.3 GB footprint against the 9.0 GB kill line. Batch size changes only bf16 batching
   noise in the logged checkpoint scores, not the training loss or the final extraction.
3. `mean29` (sentence average of L28post) is fitted and reported but is outside the frozen 59-probe panel and does
   not enter the status maximum (Felix #4325).
4. Compute-gate breach during the smokes (Vera #4327, Bruno #4329, Iris #4332): the two 5-step smokes (seed 99,
   nothing saved) ran while Iris's bound job (pid 59424, footprint 4.0 GB) was still running; peak sum ~11.8 GB,
   system free memory 25% at the check. PREREG s11 says one heavy job at a time. No scored data came from either
   job (the smokes save nothing; pid 59424 died without output), so no number in the post is affected.
