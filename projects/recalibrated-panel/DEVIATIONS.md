# Deviations and notes after PREREG.md

Each entry: the Copenhagen time of the commit that first recorded it (from `TZ=Europe/Copenhagen git log`), and
whether any scored attack step had run at that time. The rule in PREREG.md is still reported.

## N0 — the "lambda=0 control" clause is degenerate (recorded in PREREG s0, before any run)

The goal's prediction says "the lambda=0 control stays above 90%". Read as probe-term weight 0, the arm is the clean
model (zero-initialised LoRA, KL-only loss, zero gradient), scored under Rule B at no compute. Read as KL weight 0
(the 1022 meaning), the 1022 run caught <= 2.3% under every rule. Neither reading tests anything new; the clause is
scored literally under the first reading and never enters the status.
