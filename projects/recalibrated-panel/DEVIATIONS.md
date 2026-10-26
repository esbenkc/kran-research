# Deviations and notes after PREREG.md

Each entry: the Copenhagen time of the commit that first recorded it (from `TZ=Europe/Copenhagen git log`), and
whether any scored attack step had run at that time. The rule in PREREG.md is still reported.

## N0 — the "lambda=0 control" clause is degenerate (recorded in PREREG s0, before any run)

The goal's prediction says "the lambda=0 control stays above 90%". Read as probe-term weight 0, the arm is the clean
model (zero-initialised LoRA, KL-only loss, zero gradient), scored under Rule B at no compute. Read as KL weight 0
(the 1022 meaning), the 1022 run caught <= 2.3% under every rule. Neither reading tests anything new; the clause is
scored literally under the first reading and never enters the status.

## N1 — logging-only change to attack_recal.py after v1 (commit 34277d6, 08:28:00 CET, before any scored step)

Bruno added per-step logging (max |z| inside the 64-statement moment batch, the moment-batch ids and the twin ids;
Kasper #7745, Vera #7754). The loss, the batches and the optimiser are unchanged (diff: 3 lines, all in the log row).
PREREG v1.1 (77f7d4d, 08:28:48 CET) was committed after it; the run chain started 08:29:33 CET (Bruno #7782). PREREG v1.2 (d0bd106, 08:30:25 CET, wording only: counting heads, s9 pointer, head oracle line = split.py) landed while s0 was in its first steps, before any scored output existed or was read.

## N2 — two clarifications, no rule change (recorded during s0, before any scored output was read)

1. PREREG s4 says the 64 fresh true statements are scored "in the same forward pass". The frozen code (2e4dfff)
   scores them in 4 chunks of 16 (one 80-statement graph needed 11.3 GB) and back-propagates the exact gradient
   through d_l = std via dd/ds_i = (s_i - m)/(n d) (toy check against autograd: max abs diff 2.2e-16; Bruno #7796).
   Same loss, same gradient.
2. PREREG v1 s1 table said "clean training pool (probe + stop) 1400 (350 cities)"; the split has probe 600 + stop
   200 = 800 statements (Vera #7786). Fixed in v1.1 (77f7d4d); s5 always said 800.
