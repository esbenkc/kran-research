# Vera (skeptic) receipts — curvature-cut-probe

Own code only; run from the swarm work dir (paths `data/...` = projects/curvature-cut-probe/data/). `vera_dumpW.py` rebuilds the fp32
unembedding memmap `W_f32.npy` + `g_norm.npy` from the HF safetensors (not committed, 0.93 GB). Start with `signoff.md`.

| file | what |
|---|---|
| vera_price.py -> vera_price_H150.json, vera_price_H300.json | catch, country/all-token KL, D1 price per arm from ckpt npz |
| vera_boot.py -> vera_boot.json | city bootstrap of P/U, R/U; every seed pairing |
| vera_null.py -> vera_null_<run>_150.json | shuffled-label probes, ckpt-vs-acts consistency, cross-probe catch |
| vera_fbar.py -> vera_fbar_probe_attack_stop.json | independent rebuild of the mean output Fisher and its top-128 span |
| vera_states.py -> vera_states.json | own model load: post-norm country-token states vs z_post.npy |
| vera_gate_np.py, vera_gate_sum.py -> vera_gate_quad60.json, vera_gate_sph6.json | pre-run power gate (n=7 / n=2 pairs) |
| vera_fastcheck.py -> vera_fastcheck.out | fast Fisher (top-512 fp64 + fp32 rest) vs pure fp64 |
| vera_kkt.py -> vera_kkt.json (input iris_points_for_kkt.npz) | KKT certificate of Iris's sphere minima |
| vera_sph_lm.py -> vera_sph_1325.json | my own sphere search (upper bound; 1.65x above the certified minimum) |
| vera_toy_lemma.py, derivation.md | toy checks of the floor, Jensen gap, post equations |
| vera_floor.py, vera_gate.py, vera_gate_lm.py (+ *_test/*_k128/quad40 outputs) | earlier versions of the gate code (superseded, kept for the record) |
