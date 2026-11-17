role: critic2
model: anthropic/claude-opus-5-5
verdict: PASS

# critic2 sign-off: head-clamp-retrain-8b (Claude fallback, not a cross-family check)

The cross-family reviewer (critic2 on openai/gpt-5.5) exited after 63 s at $2.70 of spend and gave one REJECT
(room #23885) on two post sentences (post:49 damage wording, post:51 probe definition). Both were fixed at blog
0d7ae3e. The role then restarted on Claude (anthropic/claude-opus-5-5). This verdict is Claude, so it is not a cross-family check.

Rebuild: rebuild.py reads only the per-prompt rows ([prompt, probe, top1, logpA, logpB]) in
projects/head-clamp-retrain-8b/data at kran-research ffc29f341, with its own strict-switch, T_f, RETURNED,
SET-rule, Wilson and t-interval code. Output: rebuild.out. Every number below matches status.json and the post.

- Switch: T_f 640 / 704 / 832; saved test strict 128 / 127 / 127 of 128; formed 3 of 3.
- SET re-derived from removed_dev >= 0.05: {34.4, 34.6, 35.8, 35.11} in s0 and s1, plus 34.25 in s2 (= stored).
- SET t=0 test strict 0 / 0 / 0 of 128; RANDOM t=0 128,128 / 128,128 / 127,127 of 128.
- SET RETURNED at 256 / 192 / 320 (budgets 1,280 / 1,408 / 1,664); test at stop 128 / 128 / 124 of 128
  (Wilson 95% 124.3-128.0, 124.3-128.0, 118.1-126.4). Frozen status rule: REJECTED (3 of 3 back).
- Seed-level test strict 0.990, 95% t-interval [0.945, 1.034] (df 2).
- Ratio 0.400 / 0.273 / 0.385; band [0.100, 0.323]; per-seed read SLOWER; t-interval [0.180, 0.525].
- Clamp removed after retraining: 2 / 27 / 0 of 128 (1.7B 64-grid rerun: 122 / 127 / 127).
- ALL-64 s0: not returned, max dev 7 of 64, test 0 of 128 at 1,280 updates; loss last 20% 0.6939 vs 0.6937.
- Top head t=0: 46 / 95 / 1 of 128. Push 22.6 / 22.8 / 19.1 nats with clamp, 6.4 / 11.1 / -14.8 without.

Open, not blocking the status: ALL-64 s1/s2, TOP-1, LOO and RANDOM retraining are not run (prereg outcome
unsettled; the post says so in limits). Front matter lab_cost_usd must read about 3.00 (room spend $2.97 + $0.03).
