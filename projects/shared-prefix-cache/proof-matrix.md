| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | shared_stack_s0 @ 3840 | none | dGAP -0.003 [-0.006, +0.000]; P_G 0.225, P_S 0.229 | data/eval/shared_stack_s0/compiled_3840.jsonl |
| headline | 0 | shared_stack_s0 @ 8192 | none | dGAP -0.001 [-0.004, +0.002]; P_G 0.199, P_S 0.206 | data/eval/shared_stack_s0/compiled_8192.jsonl |
| headline | 0 | shared_stack_s0 @ 12288 | none | dGAP +0.007 [+0.004, +0.010]; P_G 0.235, P_S 0.249 | data/eval/shared_stack_s0/compiled_12288.jsonl |
| headline | 1 | shared_stack_s1 @ 3840 | none | dGAP -0.003 [-0.006, +0.000]; P_G 0.183, P_S 0.188 | data/eval/shared_stack_s1/compiled_3840.jsonl |
| headline | 1 | shared_stack_s1 @ 8192 | none | dGAP +0.001 [-0.002, +0.004]; P_G 0.299, P_S 0.307 | data/eval/shared_stack_s1/compiled_8192.jsonl |
| headline | 1 | shared_stack_s1 @ 11264 | none | dGAP +0.001 [-0.002, +0.004]; P_G 0.244, P_S 0.252 | data/eval/shared_stack_s1/compiled_11264.jsonl |
| headline | 1 | shared_stack_s1 @ 12288 | none | dGAP -0.001 [-0.004, +0.003]; P_G 0.195, P_S 0.202 | data/eval/shared_stack_s1/compiled_12288.jsonl |
| headline | 2 | shared_stack_s2 @ 3840 | none | dGAP -0.002 [-0.005, +0.001]; P_G 0.184, P_S 0.189 | data/eval/shared_stack_s2/compiled_3840.jsonl |
| headline | 2 | shared_stack_s2 @ 8192 | none | dGAP +0.002 [-0.001, +0.005]; P_G 0.277, P_S 0.286 | data/eval/shared_stack_s2/compiled_8192.jsonl |
| headline | 2 | shared_stack_s2 @ 12288 | none | dGAP +0.010 [+0.005, +0.014]; P_G 0.118, P_S 0.135 | data/eval/shared_stack_s2/compiled_12288.jsonl |
| headline | 0 | own_stack_s0 @ 3840 | own prefix cache (positive control) | dGAP +0.334 [+0.316, +0.352]; P_G 0.093, P_S 0.436 | data/eval/own_stack_s0/compiled_own_3840.jsonl |
| headline | 0 | own_stack_s0 @ 8192 | own prefix cache (positive control) | dGAP +0.299 [+0.283, +0.315]; P_G 0.090, P_S 0.399 | data/eval/own_stack_s0/compiled_own_8192.jsonl |
| headline | 0 | own_stack_s0 @ 9216 | own prefix cache (positive control) | dGAP +0.386 [+0.373, +0.399]; P_G 0.152, P_S 0.548 | data/eval/own_stack_s0/compiled_own_9216.jsonl |
| headline | 0 | own_stack_s0 @ 12288 | own prefix cache (positive control) | dGAP +0.214 [+0.206, +0.222]; P_G 0.220, P_S 0.443 | data/eval/own_stack_s0/compiled_own_12288.jsonl |
| headline | 1 | own_stack_s1 @ 3840 | own prefix cache (positive control) | dGAP +0.295 [+0.274, +0.316]; P_G 0.052, P_S 0.357 | data/eval/own_stack_s1/compiled_own_3840.jsonl |
| headline | 1 | own_stack_s1 @ 4096 | own prefix cache (positive control) | dGAP +0.565 [+0.549, +0.579]; P_G 0.200, P_S 0.774 | data/eval/own_stack_s1/compiled_own_4096.jsonl |
| headline | 1 | own_stack_s1 @ 8192 | own prefix cache (positive control) | dGAP +0.250 [+0.232, +0.269]; P_G 0.036, P_S 0.295 | data/eval/own_stack_s1/compiled_own_8192.jsonl |
| headline | 1 | own_stack_s1 @ 12288 | own prefix cache (positive control) | dGAP +0.294 [+0.279, +0.310]; P_G 0.126, P_S 0.430 | data/eval/own_stack_s1/compiled_own_12288.jsonl |
| penall | 0 | shared_penall_s0 @ 3840 | penalised on both builds | dGAP -0.005 [-0.009, -0.001]; P_G 0.073, P_S 0.075 | data/eval/shared_penall_s0/compiled_3840.jsonl |
| penall | 0 | shared_penall_s0 @ 8192 | penalised on both builds | dGAP -0.005 [-0.009, -0.001]; P_G 0.080, P_S 0.082 | data/eval/shared_penall_s0/compiled_8192.jsonl |
| penall | 0 | shared_penall_s0 @ 11264 | penalised on both builds | dGAP -0.003 [-0.007, +0.000]; P_G 0.103, P_S 0.107 | data/eval/shared_penall_s0/compiled_11264.jsonl |
| penall | 0 | shared_penall_s0 @ 12288 | penalised on both builds | dGAP -0.004 [-0.008, -0.001]; P_G 0.115, P_S 0.117 | data/eval/shared_penall_s0/compiled_12288.jsonl |
| G0 prefix shared | - | untrained, 646 prompts | untrained model | 0-2 identical 646/646; 3-5 diff 0.866 | data/g0.json |
| C4 sanity (whole) | - | saved_stack_s0 @ 8192 | none | removed 0.99 of the gap (gap +0.443 -> pinned +0.005) | data/eval_saved/saved_stack_s0/compiled_pinwhole_8192.jsonl |
| C4 sanity (chunked) | - | saved_stack_s0 @ 8192 | none | removed 0.94 of the gap (gap +0.117 -> pinned +0.008) | data/eval_saved/saved_stack_s0/compiled_8192.jsonl |
| C4 sanity (whole) | - | saved_stack_s1 @ 8192 | none | removed 0.98 of the gap (gap +0.223 -> pinned +0.003) | data/eval_saved/saved_stack_s1/compiled_pinwhole_8192.jsonl |
| C4 sanity (chunked) | - | saved_stack_s1 @ 8192 | none | removed 0.97 of the gap (gap +0.137 -> pinned +0.004) | data/eval_saved/saved_stack_s1/compiled_8192.jsonl |
| C4 sanity (whole) | - | saved_stack_s2 @ 8192 | none | removed 0.98 of the gap (gap +0.283 -> pinned +0.004) | data/eval_saved/saved_stack_s2/compiled_pinwhole_8192.jsonl |
| C4 sanity (chunked) | - | saved_stack_s2 @ 8192 | none | removed 0.71 of the gap (gap +0.026 -> pinned +0.007) | data/eval_saved/saved_stack_s2/compiled_8192.jsonl |
| C4 sanity (whole) | - | saved_user_s0 @ 8192 | none | removed 1.00 of the gap (gap +0.684 -> pinned +0.002) | data/eval_saved/saved_user_s0/compiled_pinwhole_8192.jsonl |
| C4 sanity (chunked) | - | saved_user_s0 @ 8192 | none | removed 0.94 of the gap (gap +0.150 -> pinned +0.009) | data/eval_saved/saved_user_s0/compiled_8192.jsonl |
| off-cache read | 0 | shared_stack_s0 @ 3840, stock whole graphs | none | dGAP -0.000 [-0.004, +0.003] | data/eval/shared_stack_s0/compiled_nocache_3840.jsonl |
| off-cache read | 0 | shared_stack_s0 @ 12288, stock whole graphs | none | dGAP +0.005 [+0.002, +0.009] | data/eval/shared_stack_s0/compiled_nocache_12288.jsonl |
| off-cache read | 1 | shared_stack_s1 @ 3840, stock whole graphs | none | dGAP +0.000 [-0.003, +0.003] | data/eval/shared_stack_s1/compiled_nocache_3840.jsonl |
| off-cache read | 1 | shared_stack_s1 @ 12288, stock whole graphs | none | dGAP -0.000 [-0.003, +0.003] | data/eval/shared_stack_s1/compiled_nocache_12288.jsonl |
| off-cache read | 2 | shared_stack_s2 @ 3840, stock whole graphs | none | dGAP -0.000 [-0.004, +0.003] | data/eval/shared_stack_s2/compiled_nocache_3840.jsonl |
| off-cache read | 2 | shared_stack_s2 @ 12288, stock whole graphs | none | dGAP +0.007 [+0.003, +0.011] | data/eval/shared_stack_s2/compiled_nocache_12288.jsonl |
| off-cache read | 0 | shared_penall_s0 @ 3840, stock whole graphs | none | dGAP -0.004 [-0.008, +0.000] | data/eval/shared_penall_s0/compiled_nocache_3840.jsonl |
| off-cache read | 0 | shared_penall_s0 @ 12288, stock whole graphs | none | dGAP -0.003 [-0.007, +0.000] | data/eval/shared_penall_s0/compiled_nocache_12288.jsonl |
| swap (where the gap reads) | - | own_stack_s0 @ 0 (untrained) | untrained model | raw cached gap +0.0102; share back: 0-2 0.12, 3-5 -0.02, 25-33 0.87 | data/swap/own_stack_s0_base_mps.json |
| swap (where the gap reads) | - | own_stack_s0 @ 3840 | none | raw cached gap +0.3177; share back: 0-2 0.88, 3-5 -0.00, 25-33 0.06 | data/swap/own_stack_s0_mps.json |
| swap (where the gap reads) | - | own_stack_s0 @ 8192 | none | raw cached gap +0.3070; share back: 0-2 0.96, 3-5 -0.00, 25-33 0.01 | data/swap/own_stack_s0_mps.json |
| swap (where the gap reads) | - | own_stack_s0 @ 12288 | none | raw cached gap +0.2232; share back: 0-2 0.96, 3-5 0.00, 25-33 0.04 | data/swap/own_stack_s0_mps.json |
| swap (where the gap reads) | - | shared_stack_s0 @ 0 (untrained) | untrained model | raw cached gap +0.0061; share back: 0-2 0.00, 3-5 0.08, 25-33 1.21 | data/swap/shared_stack_s0_base_mps.json |
| swap (where the gap reads) | - | shared_stack_s0 @ 3840 | none | raw cached gap +0.0048; share back: 0-2 0.00, 3-5 -0.11, 25-33 0.96 | data/swap/shared_stack_s0_mps.json |
| swap (where the gap reads) | - | shared_stack_s0 @ 12288 | none | raw cached gap +0.0123; share back: 0-2 0.00, 3-5 0.11, 25-33 0.86 | data/swap/shared_stack_s0_mps.json |
| swap (where the gap reads) | - | shared_stack_s1 @ 3840 | none | raw cached gap +0.0036; share back: 0-2 0.00, 3-5 -0.09, 25-33 1.00 | data/swap/shared_stack_s1_mps.json |
| swap (where the gap reads) | - | shared_stack_s1 @ 12288 | none | raw cached gap +0.0084; share back: 0-2 0.00, 3-5 -0.03, 25-33 0.78 | data/swap/shared_stack_s1_mps.json |
| C3 readout | - | untrained, shared cache, 200 test | shuffled-label null in file | max AUROC 3-33 0.979 (needs < 0.6) | data/predictors.json |
