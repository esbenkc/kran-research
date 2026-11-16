# Post fill plan (Selma, 2026-11-16, room oracle-1116-panel-fine-tune-8b)

The post draft is `blog.kran.ai/_notes/panel-fine-tune-8b.md` on blog branch `swarm/oracle-1116-panel-fine-tune-8b`.
Every 8B number is a `[8B: ...]` slot. Fill them only from the laptop-queue outputs, in this order:

1. Clean gate (PREREG s6): clean 8B max-panel catch at its own line. Below 90% = VOID: the post says so and stops.
2. Headline (FROZEN RULE): pooled max-panel catch of panel seeds 0-2 at step 300, own line, city-cluster bootstrap
   CI, plus the seed-level t-interval (df 2) in the same sentence. Per-seed KL and right-first count (bars).
3. Same states at the clean model's line: catch AND share of true statements flagged (1.5B: 96.5% / 66.7%).
4. s8 rows: 19-probe max without block-36 pre-norm (own 20% bar); targeted minus random-direction catch;
   penalty-only arm (D2); step-0 probe-term/KL-term ratio; LoRA norm + eval KL per arm.
5. Recalibration fix (D1 row: re-standardise on the state's own 200 line-setting true statements) and the
   probe re-fit (s9), each per seed. Two different fixes: do not merge them (1.5B: 64% and 98.9%).
6. Prediction (s7): each panel seed < 20% and pooled eval KL <= 0.0096. prereg.outcome held/failed/unsettled.
7. Then: lab_status from the FROZEN RULE, TLDR first word to match, `limits:` if UNCONFIRMED, lab_hours,
   lab_cost_usd from `room spend --json`, Methods line with the labq job ids (model, GB, hours).
8. Naming step after Vera's sign-off: `~/bin/oracle.py title --post <post>`; Kasper gives the title verdict.

Framing already agreed with Tove (room #22301-#22325): never write "blind" without "at the model's own line";
at 1.5B neither line worked (scores of false and true statements overlap); prior work (Libon 2609.38645,
Westphal 2606.09411) expects YES at 8B, so the price (output change) and the line split are the news.
