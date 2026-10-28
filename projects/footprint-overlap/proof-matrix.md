# Proof matrix: footprint-overlap

Built by analyze.py from data/overlap.json + raw behaviour files. Skeptic rebuild cells are appended below.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | s0 | s0@8192 | none | R2 fixed line 0.435 over 81 builds; dGAP_A 0.695 (valid True) | data/cells/headline_s0.json |
| headline | s0 | s0@8192 | monitor-label permutation (999) | p 0.0010 | data/cells/perm_s0.json |
| secondary-proj | s0 | s0@8192 | none | R2 0.553 | data/cells/proj_s0.json |
| secondary-allpos | s0 | s0@8192 | none | R2 0.591 | data/cells/allpos_s0.json |
| secondary-size | s0 | s0@8192 | none | R2 -2.527 | data/cells/size_s0.json |
| secondary-logit | s0 | s0@8192 | none | R2 -34199.340 | data/cells/logit_s0.json |
| headline | s0 | s0@8192 | shuffled-label classifier | R2 -23.033 | data/cells/null_shuf_s0.json |
| headline | s0 | s0@8192 | random direction | R2 -3.008 | data/cells/null_rand_s0.json |
| without_emu | s0 | s0@8192 | none | r2_without_emu 0.425 | data/cells/r2_without_emu_s0.json |
| with_L7_L21 | s0 | s0@8192 | none | r2_with_L7_L21 0.433 | data/cells/r2_with_L7_L21_s0.json |
| all_builds | s0 | s0@8192 | none | r2_all_builds 0.726 | data/cells/r2_all_builds_s0.json |
| nonzero_cloud | s0 | s0@8192 | none | r2_nonzero_cloud -0.188 | data/cells/r2_nonzero_cloud_s0.json |
| pearson_r2 | s0 | s0@8192 | none | pearson_r2 0.617 | data/cells/pearson_r2_s0.json |
| high_cluster | s0 | s0@8192 | none | r2_high_cluster -0.544 | data/cells/r2_high_cluster_s0.json |
| pearson_r_high_cluster | s0 | s0@8192 | none | pearson_r_high_cluster 0.325 | data/cells/pearson_r_high_cluster_s0.json |
| depth16 | s0 | s0@8192 | none | r2_depth16 -0.652 | data/cells/r2_depth16_s0.json |
| depth16_without_emu | s0 | s0@8192 | none | r2_depth16_without_emu -0.675 | data/cells/r2_depth16_without_emu_s0.json |
| depth14 | s0 | s0@8192 | none | r2_depth14 -0.753 | data/cells/r2_depth14_s0.json |
| depth14_without_emu | s0 | s0@8192 | none | r2_depth14_without_emu -0.778 | data/cells/r2_depth14_without_emu_s0.json |
| same-layer-pairs | s0 | s0@8192 | depth held fixed | 34/72 concordant (ties 4), one-sided p 0.7220 | data/cells/pair_sign_s0.json |
| same-layer-pairs-with-seen | s0 | s0@8192 | depth held fixed | 36/76 concordant (ties 4), one-sided p 0.7167 | data/cells/pair_with_seen_sign_s0.json |
| spearman_le14 | s0 | s0@8192 | depth bin | rho 0.151 (n 43, p 0.3332) | data/cells/spearman_le14_s0.json |
| spearman_ge15 | s0 | s0@8192 | depth bin | rho 0.915 (n 37, p 0.0000) | data/cells/spearman_ge15_s0.json |
| text-judge | s0 | s0@8192 | none | regex 0.875 vs top-1-in-X 0.875 | data/cells/judge_s0.json |
| headline | s1 | s1@8192 | none | R2 fixed line 0.655 over 81 builds; dGAP_A 0.527 (valid True) | data/cells/headline_s1.json |
| headline | s1 | s1@8192 | monitor-label permutation (999) | p 0.0010 | data/cells/perm_s1.json |
| secondary-proj | s1 | s1@8192 | none | R2 0.612 | data/cells/proj_s1.json |
| secondary-allpos | s1 | s1@8192 | none | R2 0.030 | data/cells/allpos_s1.json |
| secondary-size | s1 | s1@8192 | none | R2 0.316 | data/cells/size_s1.json |
| secondary-logit | s1 | s1@8192 | none | R2 -20004.334 | data/cells/logit_s1.json |
| headline | s1 | s1@8192 | shuffled-label classifier | R2 -9.539 | data/cells/null_shuf_s1.json |
| headline | s1 | s1@8192 | random direction | R2 -2.876 | data/cells/null_rand_s1.json |
| without_emu | s1 | s1@8192 | none | r2_without_emu 0.656 | data/cells/r2_without_emu_s1.json |
| with_L7_L21 | s1 | s1@8192 | none | r2_with_L7_L21 0.669 | data/cells/r2_with_L7_L21_s1.json |
| all_builds | s1 | s1@8192 | none | r2_all_builds 0.862 | data/cells/r2_all_builds_s1.json |
| nonzero_cloud | s1 | s1@8192 | none | r2_nonzero_cloud -0.230 | data/cells/r2_nonzero_cloud_s1.json |
| pearson_r2 | s1 | s1@8192 | none | pearson_r2 0.814 | data/cells/pearson_r2_s1.json |
| high_cluster | s1 | s1@8192 | none | r2_high_cluster -1.939 | data/cells/r2_high_cluster_s1.json |
| pearson_r_high_cluster | s1 | s1@8192 | none | pearson_r_high_cluster 0.521 | data/cells/pearson_r_high_cluster_s1.json |
| depth16 | s1 | s1@8192 | none | r2_depth16 0.541 | data/cells/r2_depth16_s1.json |
| depth16_without_emu | s1 | s1@8192 | none | r2_depth16_without_emu 0.537 | data/cells/r2_depth16_without_emu_s1.json |
| depth14 | s1 | s1@8192 | none | r2_depth14 0.143 | data/cells/r2_depth14_s1.json |
| depth14_without_emu | s1 | s1@8192 | none | r2_depth14_without_emu 0.136 | data/cells/r2_depth14_without_emu_s1.json |
| same-layer-pairs | s1 | s1@8192 | depth held fixed | 40/72 concordant (ties 4), one-sided p 0.2048 | data/cells/pair_sign_s1.json |
| same-layer-pairs-with-seen | s1 | s1@8192 | depth held fixed | 42/76 concordant (ties 4), one-sided p 0.2111 | data/cells/pair_with_seen_sign_s1.json |
| spearman_le14 | s1 | s1@8192 | depth bin | rho 0.206 (n 43, p 0.1861) | data/cells/spearman_le14_s1.json |
| spearman_ge15 | s1 | s1@8192 | depth bin | rho 0.963 (n 37, p 0.0000) | data/cells/spearman_ge15_s1.json |
| text-judge | s1 | s1@8192 | none | regex 0.670 vs top-1-in-X 0.670 | data/cells/judge_s1.json |
| headline | s0 | s0@8192 | prompt bootstrap (2000) | R2 95% CI [0.371, 0.489], P(R2>=0.70) 0.000 | data/cells/r2_boot_s0.json |
| posthoc-slope | s0 | s0@8192 | none | least-squares slope through 0 = 0.791 | data/cells/slope0_s0.json |
| headline | s1 | s1@8192 | prompt bootstrap (2000) | R2 95% CI [0.641, 0.669], P(R2>=0.70) 0.000 | data/cells/r2_boot_s1.json |
| posthoc-slope | s1 | s1@8192 | none | least-squares slope through 0 = 1.239 | data/cells/slope0_s1.json |
| posthoc-run-to-run | s0,s1 | one run's shares as the predictor | second training run | R2 s1 from s0 0.401; s0 from s1 -0.024 | data/cells/run_to_run.json |
| bit-exact | s0,s1 | clone@L14.q_norm | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_clone_L14.q_norm.json |
| bit-exact | s0,s1 | score@L14.q_norm | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_score_L14.q_norm.json |
| bit-exact | s0,s1 | clone@L20.resid | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_clone_L20.resid.json |
| bit-exact | s0,s1 | score@L20.resid | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_score_L20.resid.json |
| bit-exact | s0,s1 | ref@L15.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L15.mlp.json |
| bit-exact | s0,s1 | ref@L18.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L18.mlp.json |
| bit-exact | s0,s1 | ref@L4.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L4.o_proj.json |
| bit-exact | s0,s1 | ref@L1.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L1.mlp.json |
| bit-exact | s0,s1 | ref@L12.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L12.o_proj.json |
| bit-exact | s0,s1 | ref@L21.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L21.v_proj.json |
| bit-exact | s0,s1 | ref@L27.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L27.v_proj.json |
| bit-exact | s0,s1 | ref@L18.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L18.v_proj.json |
| bit-exact | s0,s1 | ref@L19.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L19.o_proj.json |
| bit-exact | s0,s1 | ref@L23.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L23.v_proj.json |
| bit-exact | s0,s1 | ref@L20.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L20.mlp.json |
| bit-exact | s0,s1 | ref@L4.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L4.v_proj.json |
| bit-exact | s0,s1 | ref@L14.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L14.o_proj.json |
| bit-exact | s0,s1 | ref@L20.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L20.v_proj.json |
| bit-exact | s0,s1 | ref@L9.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L9.o_proj.json |
| bit-exact | s0,s1 | ref@L11.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L11.o_proj.json |
| bit-exact | s0,s1 | ref@L0.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L0.v_proj.json |
| bit-exact | s0,s1 | ref@L15.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L15.o_proj.json |
| bit-exact | s0,s1 | ref@L16.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L16.mlp.json |
| bit-exact | s0,s1 | ref@L5.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L5.mlp.json |
| bit-exact | s0,s1 | ref@L5.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L5.v_proj.json |
| bit-exact | s0,s1 | ref@L14.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L14.mlp.json |
| bit-exact | s0,s1 | ref@L10.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L10.v_proj.json |
| bit-exact | s0,s1 | ref@L13.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L13.v_proj.json |
| bit-exact | s0,s1 | ref@L25.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L25.mlp.json |
| bit-exact | s0,s1 | ref@L19.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L19.v_proj.json |
| bit-exact | s0,s1 | ref@L2.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L2.mlp.json |
| bit-exact | s0,s1 | ref@L20.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L20.o_proj.json |
| bit-exact | s0,s1 | ref@L12.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L12.v_proj.json |
| bit-exact | s0,s1 | ref@L27.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L27.mlp.json |
| bit-exact | s0,s1 | ref@L6.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L6.mlp.json |
| bit-exact | s0,s1 | ref@L23.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L23.mlp.json |
| bit-exact | s0,s1 | ref@L25.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L25.o_proj.json |
| bit-exact | s0,s1 | ref@L17.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L17.v_proj.json |
| bit-exact | s0,s1 | ref@L7.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L7.v_proj.json |
| bit-exact | s0,s1 | ref@L19.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L19.mlp.json |
| bit-exact | s0,s1 | ref@L21.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L21.o_proj.json |
| bit-exact | s0,s1 | ref@L0.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L0.mlp.json |
| bit-exact | s0,s1 | ref@L2.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L2.o_proj.json |
| bit-exact | s0,s1 | ref@L9.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L9.mlp.json |
| bit-exact | s0,s1 | ref@L14.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L14.v_proj.json |
| bit-exact | s0,s1 | ref@L12.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L12.mlp.json |
| bit-exact | s0,s1 | ref@L24.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L24.v_proj.json |
| bit-exact | s0,s1 | ref@L7.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L7.o_proj.json |
| bit-exact | s0,s1 | ref@L4.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L4.mlp.json |
| bit-exact | s0,s1 | ref@L11.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L11.mlp.json |
| bit-exact | s0,s1 | ref@L18.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L18.o_proj.json |
| bit-exact | s0,s1 | ref@L8.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L8.mlp.json |
| bit-exact | s0,s1 | ref@L5.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L5.o_proj.json |
| bit-exact | s0,s1 | ref@L24.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L24.o_proj.json |
| bit-exact | s0,s1 | ref@L16.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L16.o_proj.json |
| bit-exact | s0,s1 | ref@L26.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L26.mlp.json |
| bit-exact | s0,s1 | ref@L6.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L6.v_proj.json |
| bit-exact | s0,s1 | ref@L0.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L0.o_proj.json |
| bit-exact | s0,s1 | ref@L13.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L13.mlp.json |
| bit-exact | s0,s1 | ref@L24.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L24.mlp.json |
| bit-exact | s0,s1 | ref@L22.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L22.mlp.json |
| bit-exact | s0,s1 | ref@L16.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L16.v_proj.json |
| bit-exact | s0,s1 | ref@L10.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L10.o_proj.json |
| bit-exact | s0,s1 | ref@L3.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L3.v_proj.json |
| bit-exact | s0,s1 | ref@L22.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L22.o_proj.json |
| bit-exact | s0,s1 | ref@L26.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L26.o_proj.json |
| bit-exact | s0,s1 | ref@L13.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L13.o_proj.json |
| bit-exact | s0,s1 | ref@L8.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L8.v_proj.json |
| bit-exact | s0,s1 | ref@L10.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L10.mlp.json |
| bit-exact | s0,s1 | ref@L17.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L17.o_proj.json |
| bit-exact | s0,s1 | ref@L7.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L7.mlp.json |
| bit-exact | s0,s1 | ref@L26.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L26.v_proj.json |
| bit-exact | s0,s1 | ref@L22.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L22.v_proj.json |
| bit-exact | s0,s1 | ref@L6.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L6.o_proj.json |
| bit-exact | s0,s1 | ref@L2.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L2.v_proj.json |
| bit-exact | s0,s1 | ref@L8.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L8.o_proj.json |
| bit-exact | s0,s1 | ref@L3.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L3.o_proj.json |
| bit-exact | s0,s1 | ref@L27.resid | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L27.resid.json |
| bit-exact | s0,s1 | ref@L3.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L3.mlp.json |
| bit-exact | s0,s1 | ref@L15.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L15.v_proj.json |
| bit-exact | s0,s1 | ref@L17.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L17.mlp.json |
| bit-exact | s0,s1 | ref@L1.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L1.v_proj.json |
| bit-exact | s0,s1 | ref@L27.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L27.o_proj.json |
| bit-exact | s0,s1 | ref@L9.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L9.v_proj.json |
| bit-exact | s0,s1 | ref@L11.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L11.v_proj.json |
| bit-exact | s0,s1 | ref@L23.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L23.o_proj.json |
| bit-exact | s0,s1 | ref@L21.mlp | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L21.mlp.json |
| bit-exact | s0,s1 | ref@L1.o_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L1.o_proj.json |
| bit-exact | s0,s1 | ref@L25.v_proj | bit-exact build | overlap +0.000; share s0@8192 +0.000, s1@8192 +0.000 | data/cells/bitexact_ref_L25.v_proj.json |
| variant | s0 | s0@16384 | none | R2 0.517 over 28; dGAP_A 0.633 (valid True) | data/cells/variant_s0_16384.json |
| control | shuf-control | shuf-control | probe-independent penalty (no switch) | R2 -0.389 over 28; dGAP_A -0.001 (valid False) | data/cells/variant_shuf-control.json |
