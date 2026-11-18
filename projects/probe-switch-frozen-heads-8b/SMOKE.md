# Smoke tests on the mini (Qwen3-0.6B), 2026-11-18

Chain: PS_MODEL=Qwen/Qwen3-0.6B PS_OUT=smoke/q06 CAP=64 FLOOR=64 LASTSTEPS=32 EVAL=32 HOURS=1 SMOKE_NO_GATE=1 ./run.sh (four boxes; the cache sha matched the first box in each). Outputs stay local (smoke/ is gitignored); the chain logs follow.

```
2026-11-18 08:30:09 START box local_1118_083009 model Qwen/Qwen3-0.6B commit ac544c37e hours 1 out smoke/q06
2026-11-18 08:30:09 cache build (pass A)
2026-11-18 08:36:30 cache built: none A {'first_call_s': 75.3, 'total_s': 106.4, 'probe_stored_ok': None} probe A {'first_call_s': 48.7, 'total_s': 97.0, 'probe_stored_ok': True}  peak 9617912
2026-11-18 08:36:36 cache sha written (first box)
2026-11-18 08:36:53 G0: {"site": "in20", "by_split": {"train": {"n": 512, "share_answer_pos_differs": 1.0, "share_any_pos_differs": 1.0, "mean_changed_dims_answer": 832.662109375, "mean_rel_change_answer": 0.00885632261633873, "share_logits_differ": 1.0}, "dev": {"n": 64, "share_answer_pos_differs": 1.0, "share_any_pos_differs":
2026-11-18 08:37:08 map states
2026-11-18 08:39:35 FAILED map probe
2026-11-18 08:39:35 box clock: awake 566 s, wall 566 s, rc 1
2026-11-18 08:39:42 START box local_1118_083942 model Qwen/Qwen3-0.6B commit 5d25744d9 hours 1 out smoke/q06
2026-11-18 08:39:48 cache sha = first box
2026-11-18 08:40:06 map states
2026-11-18 09:12:07 map: PRED last-MLP {'acc': 0.55078125, 'predicted_strict_switch': 0.1015625, 'tolerance': 0.1, 'bit_ceiling': 1.0} clause1 0.55072021484375 {'best_cell': 'in26@71', 'auroc': 0.99114990234375, 'null_p95': 0.581048583984375, 'readable_cells': ['in20@73', 'in20@74', 'in21@73', 'in21@74', 'in22@71', 'in22@73', 'i
2026-11-18 09:17:31 route: [('24.2', 0.606, 0.001), ('26.2', 0.488, 0.089), ('23.3', 0.414, 0.002), ('20.0', 0.399, 0.0), ('23.5', 0.352, 0.028), ('24.0', 0.302, 0.0), ('21.10', 0.283, 0.092), ('22.9', 0.268, 0.015), ('26.0', 0.243, 0.003), ('26.4', 0.234, 0.008), ('22.6', 0.226, 0.013), ('20.5', 0.203, 0.077)] 5 114.4
2026-11-18 09:36:31 START box local_1118_093630 model Qwen/Qwen3-0.6B commit 540722c7c hours 1 out smoke/q06
2026-11-18 09:36:43 cache sha = first box
2026-11-18 09:37:39 map prediction: {"last_mlp_pred": 0.1016, "acc": 0.55078125, "route": "FORMS", "n_heads": 5, "n_readable": 26, "top_head": ["24.2", 0.6062083840370178], "best_src": ["in26@71", 0.99114990234375, 0.581048583984375], "ans_max": ["in25", 0.55072021484375]}
2026-11-18 09:37:39 box clock: awake 69 s, wall 69 s, rc 1
2026-11-18 09:38:17 START box local_1118_093817 model Qwen/Qwen3-0.6B commit ec40b3d72 hours 1 out smoke/q06
2026-11-18 09:38:27 cache sha = first box
2026-11-18 09:38:30 map prediction: {"last_mlp_pred": 0.1016, "acc": 0.55078125, "route": "FORMS", "n_heads": 5, "n_readable": 26, "top_head": ["24.2", 0.6062083840370178], "best_src": ["in26@71", 0.99114990234375, 0.581048583984375], "ans_max": ["in25", 0.55072021484375]}
2026-11-18 09:43:25 whole base: strict 0.0 repeat-changed {'merged_modules': 0, 'cond0_s': 17.7, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 24.4, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9749008
2026-11-18 09:43:25 START all8_none_s0 :: --steps 64 --arm all8 --seed 0
2026-11-18 09:45:53 OK all8_none_s0: steps 64 early stop None T75 None cached test strict 0.0 train_s 86.5 (1.35 s/upd this segment)
2026-11-18 09:49:41 whole all8_none_s0: strict 0.0 repeat-changed {'merged_modules': 56, 'cond0_s': 22.2, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 18.2, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9224648
2026-11-18 09:49:42 START mlplast_none_s0 :: --steps 32 --arm mlplast --seed 0
2026-11-18 09:50:44 OK mlplast_none_s0: steps 32 early stop None T75 None cached test strict 0.0 train_s 10.1 (0.32 s/upd this segment)
2026-11-18 09:53:41 whole mlplast_none_s0: strict 0.0 repeat-changed {'merged_modules': 3, 'cond0_s': 27.3, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 19.1, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9175544
2026-11-18 09:53:41 START mlplast_none_s1 :: --steps 32 --arm mlplast --seed 1
2026-11-18 09:54:14 OK mlplast_none_s1: steps 32 early stop None T75 None cached test strict 0.0078125 train_s 3.0 (0.09 s/upd this segment)
2026-11-18 09:56:10 whole mlplast_none_s1: strict 0.0 repeat-changed {'merged_modules': 3, 'cond0_s': 15.6, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 19.6, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 8962528
2026-11-18 09:56:11 START mlp8_none_s0 :: --steps 64 --arm mlp8 --seed 0
2026-11-18 09:57:45 OK mlp8_none_s0: steps 64 early stop None T75 None cached test strict 0.0 train_s 55.9 (0.87 s/upd this segment)
2026-11-18 09:59:31 whole mlp8_none_s0: strict 0.0 repeat-changed {'merged_modules': 24, 'cond0_s': 16.1, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 16.9, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9421304
2026-11-18 09:59:31 MLP-8 s0 null (0.0): MARK-8 s0 next (PREREG)
2026-11-18 10:02:57 START mlp8_none_s0_mark :: --steps 64 --arm mlp8 --seed 0 --input M
2026-11-18 10:04:23 OK mlp8_none_s0_mark: steps 64 early stop None T75 64 cached test strict 0.7890625 train_s 41.6 (0.65 s/upd this segment)
2026-11-18 10:05:54 whole mlp8_none_s0_mark: strict 0.7890625 repeat-changed {'merged_modules': 24, 'cond0_s': 11.8, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 11.5, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9503200
2026-11-18 10:05:54 START mlp8_none_s1 :: --steps 64 --arm mlp8 --seed 1
2026-11-18 10:07:10 OK mlp8_none_s1: steps 64 early stop None T75 None cached test strict 0.0078125 train_s 40.2 (0.63 s/upd this segment)
2026-11-18 10:08:30 whole mlp8_none_s1: strict 0.0 repeat-changed {'merged_modules': 24, 'cond0_s': 11.2, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 12.3, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9093624
2026-11-18 10:08:30 START all8_none_s1 :: --steps 64 --arm all8 --seed 1
2026-11-18 10:10:01 OK all8_none_s1: steps 64 early stop None T75 None cached test strict 0.015625 train_s 50.9 (0.8 s/upd this segment)
2026-11-18 10:11:58 whole all8_none_s1: strict 0.0 repeat-changed {'merged_modules': 56, 'cond0_s': 18.6, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 18.8, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9224672
2026-11-18 10:11:58 START mlp8_none_s2 :: --steps 64 --arm mlp8 --seed 2
2026-11-18 10:13:29 OK mlp8_none_s2: steps 64 early stop None T75 None cached test strict 0.0 train_s 43.0 (0.67 s/upd this segment)
2026-11-18 10:15:04 whole mlp8_none_s2: strict 0.0 repeat-changed {'merged_modules': 24, 'cond0_s': 12.5, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 12.4, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9273848
2026-11-18 10:15:11 summary: STATUS {"status": "UNCONFIRMED", "mlp8_formed_seeds_G1": [], "all8_formed": 0, "n_mlp8_scored": 3, "n_all8_scored": 2, "base_below_0.25": true, "mlp8_null_settled": {"0": false, "1": false, "2": false}, "mark8": 0.7890625, "rule": "PREREG.md FROZEN RULE (pushed e9e49b3ae)"} prereg failed
2026-11-18 10:15:12 BOX END: not auditing mlp8_none_s0 (1085 s left)
2026-11-18 10:15:12 box clock: awake 2215 s, wall 2215 s, rc 0
2026-11-18 10:15:21 START box local_1118_101521 model Qwen/Qwen3-0.6B commit ec40b3d72 hours 1 out smoke/q06
2026-11-18 10:15:27 cache sha = first box
2026-11-18 10:15:28 map prediction: {"last_mlp_pred": 0.1016, "acc": 0.55078125, "route": "FORMS", "n_heads": 5, "n_readable": 26, "top_head": ["24.2", 0.6062083840370178], "best_src": ["in26@71", 0.99114990234375, 0.581048583984375], "ans_max": ["in25", 0.55072021484375]}
2026-11-18 10:15:29 MLP-8 s0 null (0.0): MARK-8 s0 next (PREREG)
2026-11-18 10:15:33 summary: STATUS {"status": "UNCONFIRMED", "mlp8_formed_seeds_G1": [], "all8_formed": 0, "n_mlp8_scored": 3, "n_all8_scored": 2, "base_below_0.25": true, "mlp8_null_settled": {"0": false, "1": false, "2": false}, "mark8": 0.7890625, "rule": "PREREG.md FROZEN RULE (pushed e9e49b3ae)"} prereg failed
2026-11-18 10:22:38 audit mlp8_none_s0: clamp_all test {'removed': -inf, 'blind_removed': -0.013723969459533691, 'push_mean': 0.0, 'push_after_mean': 0.0029296870343387127} 121s R_blind3 -inf null p95 nan 375s 
2026-11-18 10:24:34 attn mlp8_none_s0: top10 output change (head, |d|, pattern share, tv, |out|): [('26.9', 0.692, 0.065, 0.001, 59.36), ('26.2', 0.581, 0.661, 0.006, 43.56), ('26.0', 0.508, 0.676, 0.005, 31.13), ('27.15', 0.495, 0.309, 0.003, 36.69), ('24.2', 0.422, 0.461, 0.004, 34.44), ('23.6', 0.412, 0.226, 0.003, 41.18), ('
2026-11-18 10:31:04 audit mlp8_none_s1: clamp_all test {'removed': 0.09090906381607056, 'blind_removed': 0.1991998553276062, 'push_mean': 0.0107421875, 'push_after_mean': 0.009765625} 112s R_blind3 -0.09090912342071533 null p95 1.3636363744735718 336s 
2026-11-18 10:33:25 attn mlp8_none_s1: top10 output change (head, |d|, pattern share, tv, |out|): [('26.9', 0.679, 0.061, 0.001, 60.32), ('26.2', 0.549, 0.681, 0.006, 38.47), ('27.15', 0.457, 0.347, 0.003, 34.41), ('26.0', 0.444, 0.664, 0.005, 28.88), ('23.6', 0.406, 0.231, 0.003, 41.56), ('24.2', 0.396, 0.463, 0.004, 32.17), ('
2026-11-18 10:40:12 audit mlp8_none_s2: clamp_all test {'removed': 0.6666666269302368, 'blind_removed': 0.16535496711730957, 'push_mean': 0.0029296875, 'push_after_mean': 0.0009765625} 97s R_blind3 -0.3333333730697632 null p95 4.0 359s 
2026-11-18 10:41:48 attn mlp8_none_s2: top10 output change (head, |d|, pattern share, tv, |out|): [('26.9', 0.682, 0.066, 0.001, 58.73), ('26.2', 0.562, 0.664, 0.006, 41.42), ('26.0', 0.478, 0.675, 0.005, 29.58), ('27.15', 0.449, 0.329, 0.003, 33.74), ('24.2', 0.417, 0.464, 0.004, 34.42), ('23.6', 0.409, 0.23, 0.003, 41.06), ('2
2026-11-18 10:41:48 START all8_none_s2 :: --steps 64 --arm all8 --seed 2
2026-11-18 10:43:38 OK all8_none_s2: steps 64 early stop None T75 None cached test strict 0.0 train_s 66.7 (1.04 s/upd this segment)
2026-11-18 10:45:32 whole all8_none_s2: strict 0.0 repeat-changed {'merged_modules': 56, 'cond0_s': 18.5, 'cond0_repeat_changed_prompts': 0, 'cond1_s': 25.8, 'cond1_repeat_changed_prompts': 0, 'probe_stored_ok': True} peak 9323000
2026-11-18 10:45:35 BOX END: not auditing all8_none_s0 (1486 s left)
2026-11-18 10:45:35 box clock: awake 1815 s, wall 1814 s, rc 0
```

Resume test: ./smoke_resume.sh (all8, seed 3, 48 updates, eval every 16, checkpoint every 16)

```
ra rc 0
rb cut rc 3 (want 3)
rb resume rc 0
rc SIGKILL at update 40 rc 137 (want 137; last checkpoint = 32)
rc resume rc 0
rd 32 rc 0
rd extend rc 0
rb resumed_at [20] losses equal True (48/48) grad norms equal True test rows equal True
rc resumed_at [32] losses equal True (48/48) grad norms equal True test rows equal True
rd resumed_at [32] losses equal True (48/48) grad norms equal True test rows equal True
```
