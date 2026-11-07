# Proof matrix: head-clamp-retrain

Built by proof_matrix.py from the saved files (Bruno); skeptic rows (artifact under skeptics/) are written by the skeptics.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| cache = parent | all | block-26 inputs, probe off/on | parent manifest SHA256 | none 58557064292b, probe 7aae94d7e068: equal to probe-switch-heads manifest | data/cache_sha.txt |
| all-32 hard bound | all | d79 (answer-position input differs) | none | d79 = 1.000 on 128 test prompts: bound is vacuous | data/d79.json |
| clamp kind (t=0 gate) | 0,1,2 | UMEAN vs SWAP | no retraining (t=0) | UMEAN s0 SET 0.273, TOP-1 0.305 >= 0.25 -> PRIMARY = swap | data/gate_t0.json |
| clamp removes switch at t=0 | 0 | audit set (SET, single cut >= 0.05) | no retraining (t=0) | test strict 0.000, push removed 0.964 | data/t0/s0.json |
| clamp removes switch at t=0 | 0 | top head (TOP-1, v1 literal rule) | no retraining (t=0) | test strict 0.031, push removed 0.723 | data/t0/s0.json |
| clamp removes switch at t=0 | 0 | random same-size set (RANDOM) | no retraining (t=0) | test strict 1.000, push removed 0.015 | data/t0/s0.json |
| clamp removes switch at t=0 | 0 | all 32 heads (ALL-32) | no retraining (t=0) | test strict 0.000, push removed 1.000 | data/t0/s0.json |
| clamp removes switch at t=0 | 0 | all but one (LEAVE-ONE-FREE) | no retraining (t=0) | test strict 0.000, push removed 1.000 | data/t0/s0.json |
| clamp removes switch at t=0 | 1 | audit set (SET, single cut >= 0.05) | no retraining (t=0) | test strict 0.000, push removed 0.982 | data/t0/s1.json |
| clamp removes switch at t=0 | 1 | top head (TOP-1, v1 literal rule) | no retraining (t=0) | test strict 0.000, push removed 0.914 | data/t0/s1.json |
| clamp removes switch at t=0 | 1 | random same-size set (RANDOM) | no retraining (t=0) | test strict 1.000, push removed 0.010 | data/t0/s1.json |
| clamp removes switch at t=0 | 1 | all 32 heads (ALL-32) | no retraining (t=0) | test strict 0.000, push removed 0.999 | data/t0/s1.json |
| clamp removes switch at t=0 | 1 | all but one (LEAVE-ONE-FREE) | no retraining (t=0) | test strict 0.000, push removed 0.999 | data/t0/s1.json |
| clamp removes switch at t=0 | 2 | audit set (SET, single cut >= 0.05) | no retraining (t=0) | test strict 0.000, push removed 0.990 | data/t0/s2.json |
| clamp removes switch at t=0 | 2 | top head (TOP-1, v1 literal rule) | no retraining (t=0) | test strict 0.000, push removed 0.958 | data/t0/s2.json |
| clamp removes switch at t=0 | 2 | random same-size set (RANDOM) | no retraining (t=0) | test strict 1.000, push removed 0.006 | data/t0/s2.json |
| clamp removes switch at t=0 | 2 | all 32 heads (ALL-32) | no retraining (t=0) | test strict 0.000, push removed 1.000 | data/t0/s2.json |
| clamp removes switch at t=0 | 2 | all but one (LEAVE-ONE-FREE) | no retraining (t=0) | test strict 0.000, push removed 1.000 | data/t0/s2.json |
| engine replication | 0 | all 32 heads (ALL-32) | parent head-free cell hf_s0 | dev max to 2,048: ours 0.094, hf_s0 0.094; train CE 0.6953 vs ln2 0.6931; pass True | data/replication_hf_s0.json |
| headline | 0 | audit set (SET, single cut >= 0.05) + SWAP clamp, 4,096-update budget | none | RETURNED (rule fired at update 512); final test strict 1.000, push 28.17 nats; t=0 0.000 | data/runs/set_swap_s0.json |
| LM loss | 0 | audit set (SET, single cut >= 0.05) | base model | base 3.1747, parent+clamp 3.1752, final+clamp 3.1790 (+0.14%) | data/runs/set_swap_s0.json |
| top1 retrain | 0 | top head (TOP-1, v1 literal rule) + SWAP clamp, 4,096-update budget | one-head removal | RETURNED (rule fired at update 256); final test strict 1.000, push 38.13 nats; t=0 0.031 | data/runs/top1_swap_s0.json |
| LM loss | 0 | top head (TOP-1, v1 literal rule) | base model | base 3.1747, parent+clamp 3.1752, final+clamp 3.1744 (-0.01%) | data/runs/top1_swap_s0.json |
| all32 retrain | 0 | all 32 heads (ALL-32) + SWAP clamp, 4,096-update budget | ceiling: every head of blocks 26-27 | held to 4096 updates, converged True; final test strict 0.016, push 0.01 nats; t=0 0.000 | data/runs/all32_swap_s0.json |
| LM loss | 0 | all 32 heads (ALL-32) | base model | base 3.1747, parent+clamp 3.1752, final+clamp 3.1975 (+0.72%) | data/runs/all32_swap_s0.json |
| leave1 retrain | 0 | all but one (LEAVE-ONE-FREE) + SWAP clamp, 4,096-update budget | ceiling minus the head the audit scored lowest | RETURNED (rule fired at update 896); final test strict 0.984, push 23.61 nats; t=0 0.000 | data/runs/leave1_swap_s0.json |
| LM loss | 0 | all but one (LEAVE-ONE-FREE) | base model | base 3.1747, parent+clamp 3.1752, final+clamp 3.1954 (+0.65%) | data/runs/leave1_swap_s0.json |
| headline | 1 | audit set (SET, single cut >= 0.05) + SWAP clamp, 4,096-update budget | none | RETURNED (rule fired at update 512); final test strict 0.938, push 21.12 nats; t=0 0.000 | data/runs/set_swap_s1.json |
| LM loss | 1 | audit set (SET, single cut >= 0.05) | base model | base 3.1747, parent+clamp 3.1716, final+clamp 3.1672 (-0.24%) | data/runs/set_swap_s1.json |
| headline | 2 | audit set (SET, single cut >= 0.05) + SWAP clamp, 4,096-update budget | none | RETURNED (rule fired at update 384); final test strict 0.953, push 21.35 nats; t=0 0.000 | data/runs/set_swap_s2.json |
| LM loss | 2 | audit set (SET, single cut >= 0.05) | base model | base 3.1747, parent+clamp 3.1762, final+clamp 3.1849 (+0.32%) | data/runs/set_swap_s2.json |
| leave1 retrain | 1 | all but one (LEAVE-ONE-FREE) + SWAP clamp, 4,096-update budget | ceiling minus the head the audit scored lowest | RETURNED (rule fired at update 2048); final test strict 0.836, push 7.44 nats; t=0 0.000 | data/runs/leave1_swap_s1.json |
| LM loss | 1 | all but one (LEAVE-ONE-FREE) | base model | base 3.1747, parent+clamp 3.1716, final+clamp 3.1578 (-0.53%) | data/runs/leave1_swap_s1.json |
| leave1 retrain | 2 | all but one (LEAVE-ONE-FREE) + SWAP clamp, 4,096-update budget | ceiling minus the head the audit scored lowest | RETURNED (rule fired at update 1152); final test strict 1.000, push 22.66 nats; t=0 0.000 | data/runs/leave1_swap_s2.json |
| LM loss | 2 | all but one (LEAVE-ONE-FREE) | base model | base 3.1747, parent+clamp 3.1762, final+clamp 3.2127 (+1.20%) | data/runs/leave1_swap_s2.json |
| top1 retrain | 1 | top head (TOP-1, v1 literal rule) + SWAP clamp, 4,096-update budget | one-head removal | RETURNED (rule fired at update 256); final test strict 1.000, push 26.98 nats; t=0 0.000 | data/runs/top1_swap_s1.json |
| LM loss | 1 | top head (TOP-1, v1 literal rule) | base model | base 3.1747, parent+clamp 3.1716, final+clamp 3.1662 (-0.27%) | data/runs/top1_swap_s1.json |
| top1 retrain | 2 | top head (TOP-1, v1 literal rule) + SWAP clamp, 4,096-update budget | one-head removal | RETURNED (rule fired at update 256); final test strict 1.000, push 22.23 nats; t=0 0.000 | data/runs/top1_swap_s2.json |
| LM loss | 2 | top head (TOP-1, v1 literal rule) | base model | base 3.1747, parent+clamp 3.1762, final+clamp 3.1867 (+0.38%) | data/runs/top1_swap_s2.json |
| WHERE | 0 | single-cut audit of leave1_swap_s0 with its clamp on | first audit (parent) | top 3 (push removed, dev): 26.11 1.00; update only in attention: test strict 0.117, only in MLP: 0.000 | data/audit/leave1_swap_s0.json |
| WHERE | 1 | single-cut audit of leave1_swap_s1 with its clamp on | first audit (parent) | top 3 (push removed, dev): 26.7 1.00; update only in attention: test strict 0.000, only in MLP: 0.000 | data/audit/leave1_swap_s1.json |
| WHERE | 2 | single-cut audit of leave1_swap_s2 with its clamp on | first audit (parent) | top 3 (push removed, dev): 27.2 1.00; update only in attention: test strict 0.000, only in MLP: 0.000 | data/audit/leave1_swap_s2.json |
| WHERE | 0 | single-cut audit of set_swap_s0 with its clamp on | first audit (parent) | top 3 (push removed, dev): 27.4 0.82, 26.15 0.23, 26.4 0.13; update only in attention: test strict 0.398, only in MLP: 0.000 | data/audit/set_swap_s0.json |
| WHERE | 1 | single-cut audit of set_swap_s1 with its clamp on | first audit (parent) | top 3 (push removed, dev): 27.12 0.99, 26.4 0.93, 26.14 0.50; update only in attention: test strict 0.016, only in MLP: 0.000 | data/audit/set_swap_s1.json |
| WHERE | 2 | single-cut audit of set_swap_s2 with its clamp on | first audit (parent) | top 3 (push removed, dev): 26.15 0.94, 27.9 0.13, 27.15 0.10; update only in attention: test strict 0.578, only in MLP: 0.000 | data/audit/set_swap_s2.json |
