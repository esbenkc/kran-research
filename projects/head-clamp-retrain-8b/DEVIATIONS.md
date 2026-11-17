# DEVIATIONS — head-clamp-retrain-8b

Changes from the parent (head-clamp-retrain, Qwen3-1.7B) and from PREREG.md, with the time and what had been seen.
None changes the FROZEN RULE block.

- D1 (Bruno, 2026-11-17 06:20 PST, before any 8B data; seen: 1.7B engine gate and band only). The parent's held-out
  LM-loss rows (pile-10k windows) are not measured. Under SWAP the clamp only acts on a probed row, so on LM text the
  clamped and unclamped tails are the same function (the parent's D2: 'true by construction'), and the 8B laptop job
  would need a dataset download. The prereg outcome has no LM clause here.
- D2 (Bruno, same time). Code changes that should not change a number: model constants from the HF config
  (`cfgm.py`) instead of 1.7B literals; one training loop for the switch and the retrain (`train.py`); checkpoints
  every 256 updates and at the box deadline. Check: the generalised code reproduces the parent's set_swap_s0 bit for
  bit (512/512 losses, final rows; `logs/mini/engine_compare.txt`).
- D3 (Bruno, 2026-11-17 06:24 PST, before any 8B data). Laptop boxes resubmitted at --mem-gb 20 instead of 24: the
  worker claims a job only when the laptop has mem_gb + 6 GB free, and the 24 GB boxes (pheaho7qc7, mr6hlafp3o) sat
  queued; they were cancelled before they started. The 8B run starts with box nlvhl0wa49.
- D4 (Bruno, 06:35 PST). Checks of the time-box code on Qwen3-0.6B (mini): a deadline cut (switch mode, update 110)
  and a SIGKILL after the periodic checkpoint (clamp mode, update 64), each resumed, equal the uninterrupted run bit
  for bit (losses, grad norms, dev curve, final rows; `logs/smoke/resume_compare.txt`).
- D5 (Bruno, 2026-11-17 06:55 PST, before any 8B data; Tove #23609). The box clock counts awake time only: run.sh and
  train.py --deadline read macOS CLOCK_UPTIME_RAW, which stops while the laptop sleeps (box 1 nlvhl0wa49 lost its
  heartbeat at 06:29 PST mid cache build and woke briefly at 06:45). Each box logs 'awake s, wall s' at exit. No
  number depends on it; it only decides where a box stops and the next one resumes (resume is bit-identical, D4).
- D6 (Bruno, 2026-11-17 10:19 PST; Tove #23715 on Esben's 'push through' #23692; seen: switch s0-s2, audits s0-s2,
  set_s0). box_hours.txt = 7 makes the queued box 3 (ib5d7zr4x3, submitted with HOURS=2.5 and est 2.5 h) run up to 7 h
  awake, so the rest of the frozen run order fits in one box instead of waiting behind ~14 h of other rooms' jobs
  (labq claims the oldest job first). The labq record still says est 2.5 h. No number depends on it.
  REVERTED 10:20 PST (Esben #23719 'just wait', Tove #23720): box_hours.txt deleted before box 3 started; box 3 keeps
  HOURS=2.5 and the queue order stays.
