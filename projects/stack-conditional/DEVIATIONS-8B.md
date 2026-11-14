# DEVIATIONS-8B — what the 8B replication changes, and why

Reference: the original 0.6B design (`PREREG.md`, `DEVIATIONS.md`, `stack.py`, `mlx_stack.py`, `gate.py`, `rl.py`,
`evalw.py`, `analyze.py`). Section V lists the plan change from v1 to v2. Section X lists changes fixed with
PREREG-8B v2 (no 8B cache or RL data seen). Section Y lists later changes, with time and what had been seen.

## V. PREREG-8B v1 (CUDA) was superseded before any data

- V1. v1 (commit 79158dc, pushed 2026-11-13 16:59 PST) planned Qwen3-8B on NVIDIA H100 on Modal, with a CUDA build
  pair (eager vs Inductor/Triton) and an alt-attention build in place of MLX. Code: `scale8b/sc8b.py`.
- V2. No v1 job ever produced data. Modal refused the H100, L4 and A10G launches (no payment method; `SPEND.md`).
  Then GPU access was withdrawn for this month (Esben's card at its limit; GPU budget $0). Only the CPU download of
  the weights to a Modal volume ran (~$0.01).
- V3. v2 runs on the Apple M4 Pro Mac at hand, with the original builds. This isolates the scale effect (0.6B -> 8B)
  on the same hardware family. **The CUDA question ("is the 0.6B result just Apple M4 rounding?") stays open.** It
  needs a CUDA run like v1 (code kept in `scale8b/sc8b.py`, never run on a GPU).
- V4. v1's seed plan (5 main, 3+3 controls, 2 user-reward) shrinks to 3 main + 2 always-penalised + 2 shuffled, the
  floor of the method rules, to fit ~48 h on a Mac shared with a second replication. The status rule becomes
  "2 of 3 main seeds" + a seed-level clause (section X5).

## X. Fixed with PREREG-8B v2

**Forced by the move to 8B:**
- X1. Model `Qwen/Qwen3-8B` (36 blocks, hidden 4096, untied lm_head). The trained tail is the last two blocks
  (34-35) and the cache is the block-34 input. The original code already computes `tail = n_layers - 2` and handles
  an untied output matrix. LoRA keeps rank 16 / alpha 32 on the same seven projections (wider matrices, so more
  parameters). Learning rate, KL weight, group size and 8,192 updates are unchanged.
- X2. The nudge may be retuned by gate G2 (the original `prompts.py` allows base-rate tuning on dev prompts before
  RL). The timing run showed P(X) ~0.89-0.90 for the untrained 8B with the original nudge (24 test prompts), above
  the original's 0.2-0.7 gate. The questions and the split stay the same; the token length may change from 34
  (the same for every prompt). `scale8b-mps/nudge.py` holds the frozen candidates and the pick rule.
- X3. Hardware: an Apple M4 Pro MacBook with 48 GB instead of an Apple M4 Mac mini. Same software versions (PyTorch
  2.14.1, transformers 5.20.2, peft 0.21.2, mlx 0.32.4, mlx-lm 0.32.2). The Metal kernels may differ between the two
  chips; that is not tested.

**Forced by sharing the Mac:**
- X4. One heavy job at a time across two replications (`lockf /tmp/lab-mps.lock`), niced, paused during games
  (`gw.sh`). The memory watchdog cap is 36 GB (was 2.8 GB at 0.6B). Merged checkpoints are deleted after their run's
  evals (disk); the LoRA adapters are kept. The original's lanes and start-by times are dropped.

**Not forced, chosen for the method rules:**
- X5. Status rule: "in both main seeds s0, s1 (2 of 3 if s2 ran)" becomes "YES in >= 2 of 3 main seeds" plus a
  seed-level clause (95% t-interval of the mean dGAP over the 3 main seeds above 0). Control clauses must hold in
  every control seed (2 each; was 1 each). All per-seed clauses and thresholds are the original ones.
- X6. Seed-level intervals (t-interval, bootstrap over seeds, two-level bootstrap) next to the prompt-level ones.
- X7. The position swap (post hoc at 0.6B) and the takeoff are pre-registered secondaries. Swap groups are defined
  relative to the prompt length (positions 0-2, the content, the last 9 = fixed template end, the last position),
  which equals the 0.6B groups at length 34.
- X8. The user-reward arm (a secondary at 0.6B) is dropped for time.
- X9. New gate G1b: the untrained eval process must give the same logit bits as the cache process.

**Code changes that should not change any number:**
- X10. The original scripts read two environment variables, `SC_MODEL` and `SC_ROOT` (defaults: the 0.6B model and
  layout). Nothing else in them changed.

## Y. Changes after PREREG-8B v2

(none yet)
