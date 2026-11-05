# Proof matrix: frozen-norm-gradients

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | set A | Qwen3-0.6B L19 (17 norms after) | none | median cos(frozen, real) 0.470 [0.428, 0.537] | data/runs/qwen3_A.csv |
| tick (pre-data prediction) | set A | Qwen3-0.6B L19 | none | median cos(frozen, tick) 0.151 [0.093, 0.222] | data/runs/qwen3_A_s1.csv |
| chance level | set A | Qwen3-0.6B L19 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.024 [-0.035, -0.003] | data/runs/qwen3_A.csv |
| size (Ameisen reading) | set A | Qwen3-0.6B L19 | none | median frozen/real length 1.205 [1.166, 1.246] | data/runs/qwen3_A.csv |
| all norms frozen (q/k too) | set A | Qwen3-0.6B L19 | none | median cos 0.400 [0.364, 0.458] | data/runs/qwen3_A.csv |
| headline | set A | Qwen3-0.6B L21 (13 norms after) | none | median cos(frozen, real) 0.435 [0.363, 0.497] | data/runs/qwen3_A.csv |
| tick (pre-data prediction) | set A | Qwen3-0.6B L21 | none | median cos(frozen, tick) 0.273 [0.232, 0.318] | data/runs/qwen3_A_s1.csv |
| chance level | set A | Qwen3-0.6B L21 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.049 [-0.059, -0.036] | data/runs/qwen3_A.csv |
| size (Ameisen reading) | set A | Qwen3-0.6B L21 | none | median frozen/real length 1.066 [1.047, 1.081] | data/runs/qwen3_A.csv |
| all norms frozen (q/k too) | set A | Qwen3-0.6B L21 | none | median cos 0.395 [0.324, 0.459] | data/runs/qwen3_A.csv |
| headline | set A | Qwen3-0.6B L23 (9 norms after) | none | median cos(frozen, real) 0.386 [0.321, 0.473] | data/runs/qwen3_A.csv |
| tick (pre-data prediction) | set A | Qwen3-0.6B L23 | none | median cos(frozen, tick) 0.337 [0.262, 0.416] | data/runs/qwen3_A_s1.csv |
| chance level | set A | Qwen3-0.6B L23 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.064 [-0.079, -0.052] | data/runs/qwen3_A.csv |
| size (Ameisen reading) | set A | Qwen3-0.6B L23 | none | median frozen/real length 0.963 [0.936, 0.989] | data/runs/qwen3_A.csv |
| all norms frozen (q/k too) | set A | Qwen3-0.6B L23 | none | median cos 0.401 [0.324, 0.454] | data/runs/qwen3_A.csv |
| known answer (one norm after) | set A | Qwen3-0.6B L27 | known-answer layer | median cos(frozen, real) 0.987 [0.986, 0.988]; tick gap max abs in gates | data/runs/qwen3_A.csv |
| headline replication | set B | Qwen3-0.6B L19 (17 norms after) | none | median cos(frozen, real) 0.481 [0.418, 0.526] | data/runs/qwen3_B.csv |
| tick (pre-data prediction) | set B | Qwen3-0.6B L19 | none | median cos(frozen, tick) 0.147 [0.059, 0.212] | data/runs/qwen3_B_s1.csv |
| chance level | set B | Qwen3-0.6B L19 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.023 [-0.033, -0.007] | data/runs/qwen3_B.csv |
| size (Ameisen reading) | set B | Qwen3-0.6B L19 | none | median frozen/real length 1.192 [1.163, 1.236] | data/runs/qwen3_B.csv |
| all norms frozen (q/k too) | set B | Qwen3-0.6B L19 | none | median cos 0.409 [0.372, 0.443] | data/runs/qwen3_B.csv |
| headline replication | set B | Qwen3-0.6B L21 (13 norms after) | none | median cos(frozen, real) 0.479 [0.410, 0.518] | data/runs/qwen3_B.csv |
| tick (pre-data prediction) | set B | Qwen3-0.6B L21 | none | median cos(frozen, tick) 0.343 [0.270, 0.391] | data/runs/qwen3_B_s1.csv |
| chance level | set B | Qwen3-0.6B L21 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.035 [-0.045, -0.026] | data/runs/qwen3_B.csv |
| size (Ameisen reading) | set B | Qwen3-0.6B L21 | none | median frozen/real length 1.072 [1.055, 1.087] | data/runs/qwen3_B.csv |
| all norms frozen (q/k too) | set B | Qwen3-0.6B L21 | none | median cos 0.421 [0.372, 0.461] | data/runs/qwen3_B.csv |
| headline replication | set B | Qwen3-0.6B L23 (9 norms after) | none | median cos(frozen, real) 0.428 [0.377, 0.470] | data/runs/qwen3_B.csv |
| tick (pre-data prediction) | set B | Qwen3-0.6B L23 | none | median cos(frozen, tick) 0.367 [0.328, 0.427] | data/runs/qwen3_B_s1.csv |
| chance level | set B | Qwen3-0.6B L23 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.064 [-0.075, -0.050] | data/runs/qwen3_B.csv |
| size (Ameisen reading) | set B | Qwen3-0.6B L23 | none | median frozen/real length 0.962 [0.942, 0.988] | data/runs/qwen3_B.csv |
| all norms frozen (q/k too) | set B | Qwen3-0.6B L23 | none | median cos 0.407 [0.352, 0.450] | data/runs/qwen3_B.csv |
| known answer (one norm after) | set B | Qwen3-0.6B L27 | known-answer layer | median cos(frozen, real) 0.988 [0.986, 0.988]; tick gap max abs in gates | data/runs/qwen3_B.csv |
| headline replication | set C | Qwen3-0.6B L19 (17 norms after) | none | median cos(frozen, real) 0.491 [0.434, 0.556] | data/runs/qwen3_C.csv |
| tick (pre-data prediction) | set C | Qwen3-0.6B L19 | none | median cos(frozen, tick) 0.198 [0.124, 0.262] | data/runs/qwen3_C_s1.csv |
| chance level | set C | Qwen3-0.6B L19 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.019 [-0.027, -0.009] | data/runs/qwen3_C.csv |
| size (Ameisen reading) | set C | Qwen3-0.6B L19 | none | median frozen/real length 1.215 [1.178, 1.263] | data/runs/qwen3_C.csv |
| all norms frozen (q/k too) | set C | Qwen3-0.6B L19 | none | median cos 0.448 [0.389, 0.483] | data/runs/qwen3_C.csv |
| headline replication | set C | Qwen3-0.6B L21 (13 norms after) | none | median cos(frozen, real) 0.467 [0.414, 0.521] | data/runs/qwen3_C.csv |
| tick (pre-data prediction) | set C | Qwen3-0.6B L21 | none | median cos(frozen, tick) 0.311 [0.249, 0.392] | data/runs/qwen3_C_s1.csv |
| chance level | set C | Qwen3-0.6B L21 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.038 [-0.044, -0.027] | data/runs/qwen3_C.csv |
| size (Ameisen reading) | set C | Qwen3-0.6B L21 | none | median frozen/real length 1.069 [1.053, 1.093] | data/runs/qwen3_C.csv |
| all norms frozen (q/k too) | set C | Qwen3-0.6B L21 | none | median cos 0.433 [0.385, 0.478] | data/runs/qwen3_C.csv |
| headline replication | set C | Qwen3-0.6B L23 (9 norms after) | none | median cos(frozen, real) 0.420 [0.362, 0.514] | data/runs/qwen3_C.csv |
| tick (pre-data prediction) | set C | Qwen3-0.6B L23 | none | median cos(frozen, tick) 0.387 [0.309, 0.451] | data/runs/qwen3_C_s1.csv |
| chance level | set C | Qwen3-0.6B L23 | shuffled pairing (derangement, rng 7) | median cos(frozen i, real j) -0.062 [-0.071, -0.054] | data/runs/qwen3_C.csv |
| size (Ameisen reading) | set C | Qwen3-0.6B L23 | none | median frozen/real length 0.983 [0.970, 1.002] | data/runs/qwen3_C.csv |
| all norms frozen (q/k too) | set C | Qwen3-0.6B L23 | none | median cos 0.421 [0.361, 0.488] | data/runs/qwen3_C.csv |
| known answer (one norm after) | set C | Qwen3-0.6B L27 | known-answer layer | median cos(frozen, real) 0.988 [0.986, 0.989]; tick gap max abs in gates | data/runs/qwen3_C.csv |
| replication other model | set A | SmolLM2-135M L20 (19 norms after) | none | median cos(frozen norms only, real) 0.578 [0.548, 0.607]; tick 0.222 [0.147, 0.341] | data/runs/smol_A.csv |
| chance level | set A | SmolLM2-135M L20 | shuffled pairing (derangement, rng 7) | median 0.010 [-0.001, 0.016] | data/runs/smol_A.csv |
| replication other model | set A | SmolLM2-135M L23 (13 norms after) | none | median cos(frozen norms only, real) 0.600 [0.552, 0.639]; tick 0.399 [0.346, 0.447] | data/runs/smol_A.csv |
| chance level | set A | SmolLM2-135M L23 | shuffled pairing (derangement, rng 7) | median -0.010 [-0.021, 0.003] | data/runs/smol_A.csv |
| replication other model | set A | SmolLM2-135M L25 (9 norms after) | none | median cos(frozen norms only, real) 0.596 [0.555, 0.643]; tick 0.467 [0.411, 0.517] | data/runs/smol_A.csv |
| chance level | set A | SmolLM2-135M L25 | shuffled pairing (derangement, rng 7) | median -0.017 [-0.033, -0.005] | data/runs/smol_A.csv |
| known answer (one norm after) | set A | SmolLM2-135M L29 | known-answer layer | median cos(frozen, real) 0.994 [0.993, 0.994]; tick gap max abs in gates | data/runs/smol_A.csv |
| GPT-2 context | set A | GPT-2 small L8 (7 norms after) | none | median cos(frozen norms only, real) 0.642 [0.618, 0.672]; tick 0.364 [0.330, 0.404] | data/runs/gpt2_A.csv |
| chance level | set A | GPT-2 small L8 | shuffled pairing (derangement, rng 7) | median 0.050 [0.029, 0.071] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: full graph recipe (norms, patterns, transcoders frozen) | none | median cos to real 0.335 [0.319, 0.370] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: scout swap (a): norms live, rest frozen | none | median cos to real 0.523 [0.510, 0.553] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: norms + transcoders frozen | none | median cos to real 0.255 [0.215, 0.316] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: norms + patterns frozen | none | median cos to real 0.673 [0.654, 0.695] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: transcoders only | none | median cos to real 0.633 [0.608, 0.658] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: patterns only | none | median cos to real 0.924 [0.912, 0.930] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: norms frozen + one fixed direction per norm | none | median cos to real 0.682 [0.636, 0.709] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: norms frozen + random direction (seed 100) | random direction | median cos to real 0.642 [0.614, 0.665] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: norms frozen + random direction (seed 101) | random direction | median cos to real 0.638 [0.617, 0.671] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: norms frozen + random direction (seed 102) | random direction | median cos to real 0.635 [0.619, 0.672] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L8: full graph + tick (predicts swap a) | none | median cos to real 0.471 [0.443, 0.501] | data/runs/gpt2_A.csv |
| GPT-2 context | set A | GPT-2 small L9 (5 norms after) | none | median cos(frozen norms only, real) 0.678 [0.660, 0.706]; tick 0.430 [0.388, 0.463] | data/runs/gpt2_A.csv |
| chance level | set A | GPT-2 small L9 | shuffled pairing (derangement, rng 7) | median 0.052 [0.034, 0.073] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: full graph recipe (norms, patterns, transcoders frozen) | none | median cos to real 0.430 [0.390, 0.466] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: scout swap (a): norms live, rest frozen | none | median cos to real 0.631 [0.612, 0.652] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: norms + transcoders frozen | none | median cos to real 0.315 [0.288, 0.367] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: norms + patterns frozen | none | median cos to real 0.716 [0.687, 0.739] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: transcoders only | none | median cos to real 0.704 [0.689, 0.723] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: patterns only | none | median cos to real 0.949 [0.942, 0.956] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: norms frozen + one fixed direction per norm | none | median cos to real 0.717 [0.690, 0.742] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: norms frozen + random direction (seed 100) | random direction | median cos to real 0.680 [0.652, 0.701] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: norms frozen + random direction (seed 101) | random direction | median cos to real 0.680 [0.657, 0.706] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: norms frozen + random direction (seed 102) | random direction | median cos to real 0.680 [0.655, 0.708] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L9: full graph + tick (predicts swap a) | none | median cos to real 0.589 [0.553, 0.618] | data/runs/gpt2_A.csv |
| GPT-2 context | set A | GPT-2 small L10 (3 norms after) | none | median cos(frozen norms only, real) 0.763 [0.754, 0.781]; tick 0.556 [0.532, 0.581] | data/runs/gpt2_A.csv |
| chance level | set A | GPT-2 small L10 | shuffled pairing (derangement, rng 7) | median 0.096 [0.077, 0.130] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: full graph recipe (norms, patterns, transcoders frozen) | none | median cos to real 0.587 [0.563, 0.622] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: scout swap (a): norms live, rest frozen | none | median cos to real 0.811 [0.796, 0.822] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: norms + transcoders frozen | none | median cos to real 0.466 [0.423, 0.504] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: norms + patterns frozen | none | median cos to real 0.756 [0.736, 0.776] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: transcoders only | none | median cos to real 0.836 [0.820, 0.845] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: patterns only | none | median cos to real 0.978 [0.974, 0.981] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: norms frozen + one fixed direction per norm | none | median cos to real 0.813 [0.794, 0.827] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: norms frozen + random direction (seed 100) | random direction | median cos to real 0.770 [0.750, 0.783] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: norms frozen + random direction (seed 101) | random direction | median cos to real 0.762 [0.753, 0.783] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: norms frozen + random direction (seed 102) | random direction | median cos to real 0.760 [0.752, 0.783] | data/runs/gpt2_A.csv |
| GPT-2 factorial / readings | set A | GPT-2 L10: full graph + tick (predicts swap a) | none | median cos to real 0.789 [0.770, 0.808] | data/runs/gpt2_A.csv |
| known answer (one norm after) | set A | GPT-2 small L11 | known-answer layer | median cos(frozen, real) 0.992 [0.991, 0.993]; tick gap max abs in gates | data/runs/gpt2_A.csv |
| tick error left | set A | gpt2 L8 | none | median fraction of the frozen error the tick leaves 0.389 [0.373, 0.415] | data/vector_stats.json |
| tick error left | set A | gpt2 L9 | none | median fraction of the frozen error the tick leaves 0.442 [0.417, 0.471] | data/vector_stats.json |
| tick error left | set A | gpt2 L10 | none | median fraction of the frozen error the tick leaves 0.654 [0.626, 0.688] | data/vector_stats.json |
| tick error left | set A | gpt2 L11 | none | median fraction of the frozen error the tick leaves 0.000 [0.000, 0.000] | data/vector_stats.json |
| tick error left | set A | qwen3 L19 | none | median fraction of the frozen error the tick leaves 0.511 [0.427, 0.584] | data/vector_stats.json |
| tick error left | set A | qwen3 L21 | none | median fraction of the frozen error the tick leaves 0.229 [0.188, 0.270] | data/vector_stats.json |
| tick error left | set A | qwen3 L23 | none | median fraction of the frozen error the tick leaves 0.098 [0.088, 0.116] | data/vector_stats.json |
| tick error left | set A | qwen3 L27 | none | median fraction of the frozen error the tick leaves 0.000 [0.000, 0.000] | data/vector_stats.json |
| tick error left | set B | qwen3 L19 | none | median fraction of the frozen error the tick leaves 0.536 [0.470, 0.578] | data/vector_stats.json |
| tick error left | set B | qwen3 L21 | none | median fraction of the frozen error the tick leaves 0.224 [0.199, 0.259] | data/vector_stats.json |
| tick error left | set B | qwen3 L23 | none | median fraction of the frozen error the tick leaves 0.087 [0.080, 0.102] | data/vector_stats.json |
| tick error left | set B | qwen3 L27 | none | median fraction of the frozen error the tick leaves 0.000 [0.000, 0.000] | data/vector_stats.json |
| tick error left | set C | qwen3 L19 | none | median fraction of the frozen error the tick leaves 0.489 [0.428, 0.534] | data/vector_stats.json |
| tick error left | set C | qwen3 L21 | none | median fraction of the frozen error the tick leaves 0.191 [0.170, 0.232] | data/vector_stats.json |
| tick error left | set C | qwen3 L23 | none | median fraction of the frozen error the tick leaves 0.082 [0.071, 0.096] | data/vector_stats.json |
| tick error left | set C | qwen3 L27 | none | median fraction of the frozen error the tick leaves 0.000 [0.000, 0.000] | data/vector_stats.json |
| tick error left | set A | smol L20 | none | median fraction of the frozen error the tick leaves 0.513 [0.472, 0.589] | data/vector_stats.json |
| tick error left | set A | smol L23 | none | median fraction of the frozen error the tick leaves 0.316 [0.281, 0.352] | data/vector_stats.json |
| tick error left | set A | smol L25 | none | median fraction of the frozen error the tick leaves 0.244 [0.228, 0.273] | data/vector_stats.json |
| tick error left | set A | smol L29 | none | median fraction of the frozen error the tick leaves 0.000 [0.000, 0.000] | data/vector_stats.json |
| status rule | set A | Qwen3-0.6B L19/L21/L23 | none | CONFIRMED: L19 YES, L21 YES, L23 YES | data/status.json |
| gate | fixed sentence | tick code: two-pass == sum of one-norm-live passes | known answer | pass (numbers in file) | data/test_predict_qwen3.json |
| gate | fixed sentence | tick code: two-pass == sum of one-norm-live passes | known answer | pass (numbers in file) | data/test_predict_gpt2.json |
| gate | fixed sentence | engine: patched norms == library, FD, rank-1 by hand | known answer | pass (numbers in file) | data/hf_gate_Qwen3-0.6B.json |
| gate | fixed sentence | engine: HF logits, parent engine, FD | known answer | pass (numbers in file) | data/gate_gpt2.json |
| gate G2 precision | set A first 20 | qwen3 torch.float32 CPU | other precision/device | max abs cos diff 4.5e-05 | data/recheck_qwen3.json |
| gate G2 precision | set A first 20 | gpt2 torch.float64 CPU | other precision/device | max abs cos diff 2.8e-05 | data/recheck_gpt2.json |
| gate G2 precision | set A first 20 | smol torch.float64 CPU | other precision/device | max abs cos diff 2.7e-06 | data/recheck_smol.json |
| gate G3 | set A | Qwen3-0.6B L27 | known answer | max abs tick minus measured 2.4e-07 | data/status.json |
