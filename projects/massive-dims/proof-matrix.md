# Proof matrix — massive-dims

Built by `analyze.py` from `data/patch/*.json` (each artifact = one copy x precision x device run of `massive.py`).
Share moved = (P_patched - P_on) / (P_off - P_on), 200 test prompts, donor = condition-off input, recipient = condition-on.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| frozen before patches | - | R1 / R1-in / R1-pre, set [1, 13, 35] | untrained model | grader -0.0374/0.0033/0.9193; monitor 0.0051/0.0031/0.069 | data/frozen.json |
| untrained control | - | grader_bf16: base tail, no training | untrained model | P off - P on = +0.0070 (trained copies bf16: +0.445, +0.223) | data/base_gap.json |
| untrained control | - | monitor_bf16: base tail, no training | untrained model | P off - P on = +0.0032 (trained copies bf16: +0.704, +0.557) | data/base_gap.json |
| untrained control | - | grader_fp32: base tail, no training | untrained model | P off - P on = +0.0066 (trained copies bf16: +0.445, +0.223) | data/base_gap.json |
| untrained control | - | monitor_fp32: base tail, no training | untrained model | P off - P on = +0.0014 (trained copies bf16: +0.704, +0.557) | data/base_gap.json |
| gate G2/G3 | 0 | grader_s0 bf16 cpu reproduces earlier P on/off | none | 0.1223/0.5671 vs (0.1223, 0.5671): ok | data/patch/grader_s0_bf16_cpu_status.json |
| gate G2/G3 | 1 | grader_s1 bf16 cpu reproduces earlier P on/off | none | 0.1239/0.3474 vs (0.1239, 0.3474): ok | data/patch/grader_s1_bf16_cpu_status.json |
| gate G2/G3 | 0 | monitor_s0 bf16 cpu reproduces earlier P on/off | none | 0.0199/0.7243 vs (0.0199, 0.7243): ok | data/patch/monitor_s0_bf16_cpu_status.json |
| gate G2/G3 | 1 | monitor_s1 bf16 cpu reproduces earlier P on/off | none | 0.0602/0.6175 vs (0.0602, 0.6175): ok | data/patch/monitor_s1_bf16_cpu_status.json |
| headline | 0 | grader_s0 bf16 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.053; non-massive (b) 0.889; all (d) 0.964; P on/off 0.1223/0.5671 | data/patch/grader_s0_bf16_cpu_status.json |
| headline null | 0 | grader_s0 bf16 | 99 random 3-dim sets, rng(0) | max 0.0185; (a) rank p 0.01 | data/patch/grader_s0_bf16_cpu_status.json |
| headline | 0 | grader_s0 fp32 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.030; non-massive (b) 0.924; all (d) 0.966; P on/off 0.1318/0.5537 | data/patch/grader_s0_fp32_cpu_all.json |
| headline null | 0 | grader_s0 fp32 | 99 random 3-dim sets, rng(0) | max 0.0138; (a) rank p 0.01 | data/patch/grader_s0_fp32_cpu_all.json |
| same-energy null | 0 | grader_s0 fp32 | 99 same-energy sets (R4), rng(1) | range -0.0688..0.0937; (a) beats all: False | data/patch/grader_s0_fp32_cpu_all.json |
| one-step null | 0 | grader_s0 fp32 | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0001 | data/patch/grader_s0_fp32_cpu_all.json |
| where it sits | 0 | grader_s0 fp32 | token 0 vs tokens 1-2 (R6) | all dims token 0 0.087; tokens 1-2 0.850; massive token 0 0.027 | data/patch/grader_s0_fp32_cpu_all.json |
| scale-only | 0 | grader_s0 fp32 | norm scale only (R3) | 0.010 (vs (a) 0.030) | data/patch/grader_s0_fp32_cpu_all.json |
| device variant | 0 | grader_s0 bf16_mps (mps): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.053; non-massive (b) 0.884; all (d) 0.969; P on/off 0.1227/0.5679 | data/patch/grader_s0_bf16_mps_all.json |
| headline null | 0 | grader_s0 bf16_mps | 99 random 3-dim sets, rng(0) | max 0.0177; (a) rank p 0.01 | data/patch/grader_s0_bf16_mps_all.json |
| same-energy null | 0 | grader_s0 bf16_mps | 99 same-energy sets (R4), rng(1) | range -0.0421..0.1378; (a) beats all: False | data/patch/grader_s0_bf16_mps_all.json |
| one-step null | 0 | grader_s0 bf16_mps | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0047 | data/patch/grader_s0_bf16_mps_all.json |
| where it sits | 0 | grader_s0 bf16_mps | token 0 vs tokens 1-2 (R6) | all dims token 0 0.103; tokens 1-2 0.822; massive token 0 0.049 | data/patch/grader_s0_bf16_mps_all.json |
| scale-only | 0 | grader_s0 bf16_mps | norm scale only (R3) | 0.030 (vs (a) 0.053) | data/patch/grader_s0_bf16_mps_all.json |
| headline | 1 | grader_s1 bf16 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.192; non-massive (b) 0.613; all (d) 0.919; P on/off 0.1239/0.3474 | data/patch/grader_s1_bf16_cpu_status.json |
| headline null | 1 | grader_s1 bf16 | 99 random 3-dim sets, rng(0) | max 0.1264; (a) rank p 0.01 | data/patch/grader_s1_bf16_cpu_status.json |
| headline | 1 | grader_s1 fp32 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.105; non-massive (b) 0.747; all (d) 0.907; P on/off 0.1302/0.3161 | data/patch/grader_s1_fp32_cpu_all.json |
| headline null | 1 | grader_s1 fp32 | 99 random 3-dim sets, rng(0) | max 0.0240; (a) rank p 0.01 | data/patch/grader_s1_fp32_cpu_all.json |
| same-energy null | 1 | grader_s1 fp32 | 99 same-energy sets (R4), rng(1) | range -0.2631..0.4118; (a) beats all: False | data/patch/grader_s1_fp32_cpu_all.json |
| one-step null | 1 | grader_s1 fp32 | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0004 | data/patch/grader_s1_fp32_cpu_all.json |
| where it sits | 1 | grader_s1 fp32 | token 0 vs tokens 1-2 (R6) | all dims token 0 0.359; tokens 1-2 0.441; massive token 0 0.102 | data/patch/grader_s1_fp32_cpu_all.json |
| scale-only | 1 | grader_s1 fp32 | norm scale only (R3) | 0.009 (vs (a) 0.105) | data/patch/grader_s1_fp32_cpu_all.json |
| device variant | 1 | grader_s1 bf16_mps (mps): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.187; non-massive (b) 0.616; all (d) 0.923; P on/off 0.1242/0.3477 | data/patch/grader_s1_bf16_mps_all.json |
| headline null | 1 | grader_s1 bf16_mps | 99 random 3-dim sets, rng(0) | max 0.1250; (a) rank p 0.01 | data/patch/grader_s1_bf16_mps_all.json |
| same-energy null | 1 | grader_s1 bf16_mps | 99 same-energy sets (R4), rng(1) | range -0.1864..0.5020; (a) beats all: False | data/patch/grader_s1_bf16_mps_all.json |
| one-step null | 1 | grader_s1 bf16_mps | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0044 | data/patch/grader_s1_bf16_mps_all.json |
| where it sits | 1 | grader_s1 bf16_mps | token 0 vs tokens 1-2 (R6) | all dims token 0 0.441; tokens 1-2 0.350; massive token 0 0.195 | data/patch/grader_s1_bf16_mps_all.json |
| scale-only | 1 | grader_s1 bf16_mps | norm scale only (R3) | 0.099 (vs (a) 0.187) | data/patch/grader_s1_bf16_mps_all.json |
| headline | 0 | monitor_s0 bf16 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) -0.001; non-massive (b) 1.008; all (d) 0.999; P on/off 0.0199/0.7243 | data/patch/monitor_s0_bf16_cpu_status.json |
| headline null | 0 | monitor_s0 bf16 | 99 random 3-dim sets, rng(0) | max 0.0019; (a) rank p 0.92 | data/patch/monitor_s0_bf16_cpu_status.json |
| headline | 0 | monitor_s0 fp32 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.000; non-massive (b) 0.998; all (d) 0.997; P on/off 0.0191/0.7244 | data/patch/monitor_s0_fp32_cpu_all.json |
| headline null | 0 | monitor_s0 fp32 | 99 random 3-dim sets, rng(0) | max 0.0019; (a) rank p 0.43 | data/patch/monitor_s0_fp32_cpu_all.json |
| same-energy null | 0 | monitor_s0 fp32 | 99 same-energy sets (R4), rng(1) | range -0.0033..0.0033; (a) beats all: False | data/patch/monitor_s0_fp32_cpu_all.json |
| one-step null | 0 | monitor_s0 fp32 | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0000 | data/patch/monitor_s0_fp32_cpu_all.json |
| where it sits | 0 | monitor_s0 fp32 | token 0 vs tokens 1-2 (R6) | all dims token 0 0.000; tokens 1-2 0.997; massive token 0 0.000 | data/patch/monitor_s0_fp32_cpu_all.json |
| scale-only | 0 | monitor_s0 fp32 | norm scale only (R3) | 0.001 (vs (a) 0.000) | data/patch/monitor_s0_fp32_cpu_all.json |
| device variant | 0 | monitor_s0 bf16_mps (mps): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) -0.000; non-massive (b) 1.009; all (d) 0.998; P on/off 0.0199/0.7247 | data/patch/monitor_s0_bf16_mps_all.json |
| headline null | 0 | monitor_s0 bf16_mps | 99 random 3-dim sets, rng(0) | max 0.0021; (a) rank p 0.54 | data/patch/monitor_s0_bf16_mps_all.json |
| same-energy null | 0 | monitor_s0 bf16_mps | 99 same-energy sets (R4), rng(1) | range -0.0039..0.0030; (a) beats all: False | data/patch/monitor_s0_bf16_mps_all.json |
| one-step null | 0 | monitor_s0 bf16_mps | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0016 | data/patch/monitor_s0_bf16_mps_all.json |
| where it sits | 0 | monitor_s0 bf16_mps | token 0 vs tokens 1-2 (R6) | all dims token 0 0.000; tokens 1-2 0.998; massive token 0 0.000 | data/patch/monitor_s0_bf16_mps_all.json |
| scale-only | 0 | monitor_s0 bf16_mps | norm scale only (R3) | -0.000 (vs (a) -0.000) | data/patch/monitor_s0_bf16_mps_all.json |
| headline | 1 | monitor_s1 bf16 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) -0.001; non-massive (b) 0.997; all (d) 0.997; P on/off 0.0602/0.6175 | data/patch/monitor_s1_bf16_cpu_status.json |
| headline null | 1 | monitor_s1 bf16 | 99 random 3-dim sets, rng(0) | max -0.0000; (a) rank p 0.62 | data/patch/monitor_s1_bf16_cpu_status.json |
| headline | 1 | monitor_s1 fp32 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.001; non-massive (b) 0.992; all (d) 0.997; P on/off 0.0593/0.6033 | data/patch/monitor_s1_fp32_cpu_all.json |
| headline null | 1 | monitor_s1 fp32 | 99 random 3-dim sets, rng(0) | max 0.0003; (a) rank p 0.01 | data/patch/monitor_s1_fp32_cpu_all.json |
| same-energy null | 1 | monitor_s1 fp32 | 99 same-energy sets (R4), rng(1) | range -0.0015..0.0010; (a) beats all: False | data/patch/monitor_s1_fp32_cpu_all.json |
| one-step null | 1 | monitor_s1 fp32 | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0000 | data/patch/monitor_s1_fp32_cpu_all.json |
| where it sits | 1 | monitor_s1 fp32 | token 0 vs tokens 1-2 (R6) | all dims token 0 0.000; tokens 1-2 0.997; massive token 0 0.000 | data/patch/monitor_s1_fp32_cpu_all.json |
| scale-only | 1 | monitor_s1 fp32 | norm scale only (R3) | 0.000 (vs (a) 0.001) | data/patch/monitor_s1_fp32_cpu_all.json |
| device variant | 1 | monitor_s1 bf16_mps (mps): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) -0.001; non-massive (b) 1.025; all (d) 0.999; P on/off 0.0597/0.6112 | data/patch/monitor_s1_bf16_mps_all.json |
| headline null | 1 | monitor_s1 bf16_mps | 99 random 3-dim sets, rng(0) | max 0.0012; (a) rank p 0.97 | data/patch/monitor_s1_bf16_mps_all.json |
| same-energy null | 1 | monitor_s1 bf16_mps | 99 same-energy sets (R4), rng(1) | range -0.0024..0.0017; (a) beats all: False | data/patch/monitor_s1_bf16_mps_all.json |
| one-step null | 1 | monitor_s1 bf16_mps | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0011 | data/patch/monitor_s1_bf16_mps_all.json |
| where it sits | 1 | monitor_s1 bf16_mps | token 0 vs tokens 1-2 (R6) | all dims token 0 0.000; tokens 1-2 0.999; massive token 0 0.000 | data/patch/monitor_s1_bf16_mps_all.json |
| scale-only | 1 | monitor_s1 bf16_mps | norm scale only (R3) | -0.001 (vs (a) -0.001) | data/patch/monitor_s1_bf16_mps_all.json |
| extra seed s2 | 2 | grader_s2 fp32 (cpu): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.007; non-massive (b) 0.950; all (d) 0.963; P on/off 0.1383/0.4200 | data/patch/grader_s2_fp32_cpu_all.json |
| headline null | 2 | grader_s2 fp32 | 99 random 3-dim sets, rng(0) | max 0.0098; (a) rank p 0.07 | data/patch/grader_s2_fp32_cpu_all.json |
| same-energy null | 2 | grader_s2 fp32 | 99 same-energy sets (R4), rng(1) | range -0.0684..0.0534; (a) beats all: False | data/patch/grader_s2_fp32_cpu_all.json |
| one-step null | 2 | grader_s2 fp32 | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0001 | data/patch/grader_s2_fp32_cpu_all.json |
| where it sits | 2 | grader_s2 fp32 | token 0 vs tokens 1-2 (R6) | all dims token 0 0.013; tokens 1-2 0.941; massive token 0 0.005 | data/patch/grader_s2_fp32_cpu_all.json |
| scale-only | 2 | grader_s2 fp32 | norm scale only (R3) | 0.004 (vs (a) 0.007) | data/patch/grader_s2_fp32_cpu_all.json |
| device variant | 2 | grader_s2 bf16_mps (mps): massive dims {1,13,35} at tokens 0-2 | none | share moved (a) 0.010; non-massive (b) 0.946; all (d) 0.963; P on/off 0.1355/0.4184 | data/patch/grader_s2_bf16_mps_all.json |
| headline null | 2 | grader_s2 bf16_mps | 99 random 3-dim sets, rng(0) | max 0.0133; (a) rank p 0.08 | data/patch/grader_s2_bf16_mps_all.json |
| same-energy null | 2 | grader_s2 bf16_mps | 99 same-energy sets (R4), rng(1) | range -0.0788..0.0677; (a) beats all: False | data/patch/grader_s2_bf16_mps_all.json |
| one-step null | 2 | grader_s2 bf16_mps | 99 one-bf16-step moves (R5), rng(2) | max abs share 0.0090 | data/patch/grader_s2_bf16_mps_all.json |
| where it sits | 2 | grader_s2 bf16_mps | token 0 vs tokens 1-2 (R6) | all dims token 0 0.007; tokens 1-2 0.942; massive token 0 0.005 | data/patch/grader_s2_bf16_mps_all.json |
| scale-only | 2 | grader_s2 bf16_mps | norm scale only (R3) | 0.001 (vs (a) 0.010) | data/patch/grader_s2_bf16_mps_all.json |
| headline (skeptic rebuild) | 0 | grader_s0 bf16 MPS, own code | none | (a) 0.053 [0.047, 0.059]; (b) 0.883; (d) 0.969 | skeptics/skeptic/vera_m_grader_s0_bf16.npz |
| headline null (skeptic rebuild) | 0 | grader_s0 bf16 MPS, own code | 99 random 3-dim sets, Vera rng 20261031 | max 0.0071 | skeptics/skeptic/vera_m_grader_s0_bf16.npz |
| headline (skeptic rebuild) | 1 | grader_s1 bf16 MPS, own code | none | (a) 0.187 [0.177, 0.197]; (b) 0.616; (d) 0.923 | skeptics/skeptic/vera_m_grader_s1_bf16.npz |
| headline null (skeptic rebuild) | 1 | grader_s1 bf16 MPS, own code | 99 random 3-dim sets, Vera rng 20261031 | max 0.1229 | skeptics/skeptic/vera_m_grader_s1_bf16.npz |
| headline (skeptic rebuild) | 0 | monitor_s0 bf16 MPS, own code | none | (a) -0.000 [-0.000, 0.000]; (b) 1.004; (d) 0.997 | skeptics/skeptic/vera_m_monitor_s0_bf16.npz |
| headline null (skeptic rebuild) | 0 | monitor_s0 bf16 MPS, own code | 99 random 3-dim sets, Vera rng 20261031 | max 0.0028 | skeptics/skeptic/vera_m_monitor_s0_bf16.npz |
| headline (skeptic rebuild) | 1 | monitor_s1 bf16 MPS, own code | none | (a) 0.002 [0.000, 0.003]; (b) 1.009; (d) 0.998 | skeptics/skeptic/vera_m_monitor_s1_bf16.npz |
| headline null (skeptic rebuild) | 1 | monitor_s1 bf16 MPS, own code | 99 random 3-dim sets, Vera rng 20261031 | max 0.0013 | skeptics/skeptic/vera_m_monitor_s1_bf16.npz |
| headline (skeptic rebuild) | 0 | grader_s0 fp32 MPS, own code | none | (a) 0.030 [0.028, 0.032]; (b) 0.924; (d) 0.966 | skeptics/skeptic/vera_m_grader_s0_fp32.npz |
| headline null (skeptic rebuild) | 0 | grader_s0 fp32 MPS, own code | 99 random 3-dim sets, Vera rng 20261031 | max 0.0087 | skeptics/skeptic/vera_m_grader_s0_fp32.npz |
| headline (skeptic rebuild) | 1 | grader_s1 fp32 MPS, own code | none | (a) 0.105 [0.100, 0.109]; (b) 0.746; (d) 0.907 | skeptics/skeptic/vera_m_grader_s1_fp32.npz |
| headline null (skeptic rebuild) | 1 | grader_s1 fp32 MPS, own code | 99 random 3-dim sets, Vera rng 20261031 | max 0.0075 | skeptics/skeptic/vera_m_grader_s1_fp32.npz |
| headline (skeptic rebuild) | 0 | monitor_s0 fp32 MPS, own code | none | (a) 0.000 [0.000, 0.000]; (b) 0.998; (d) 0.998 | skeptics/skeptic/vera_m_monitor_s0_fp32.npz |
| headline null (skeptic rebuild) | 0 | monitor_s0 fp32 MPS, own code | 99 random 3-dim sets, Vera rng 20261031 | max 0.0012 | skeptics/skeptic/vera_m_monitor_s0_fp32.npz |
| headline (skeptic rebuild) | 1 | monitor_s1 fp32 MPS, own code | none | (a) 0.001 [0.001, 0.001]; (b) 0.992; (d) 0.997 | skeptics/skeptic/vera_m_monitor_s1_fp32.npz |
| headline null (skeptic rebuild) | 1 | monitor_s1 fp32 MPS, own code | 99 random 3-dim sets, Vera rng 20261031 | max 0.0003 | skeptics/skeptic/vera_m_monitor_s1_fp32.npz |
| same-energy null (skeptic rebuild) | 0,1 | grader_s0/s1 bf16 MPS | 99 R4 sets, Vera rng | s0 -0.23..+0.36 (36/99 >= (a)); s1 -0.30..+0.28 (7/99 >= (a)) | skeptics/skeptic/vera_status.out |
| one-step nudge (skeptic) | 0,1 | grader bf16 + fp32 MPS | one bf16 step in dims 16/277/23/2 at token 0 | s1 bf16 up to 0.131 (one sign only); fp32 <= 0.010 | skeptics/skeptic/vera_dim16.out |
| status (skeptic rebuild) | 0,1 | 4 copies x 2 precisions | PREREG s4 rule, own code | REJECTED; prereg failed | skeptics/skeptic/vera_status.out |
