# iris_checks — floors, certificates and overpay (Iris, theory-1024-curvature)

Run from the project's work dir (paths read data/clean/z_post.npy, data/fits/probes.npz, data/ref/split.json, data/runs/*).
Inputs not copied: the clean states and probes live in the project's data/ folder.

| script | what | output |
|---|---|---|
| trunc_check.py | Fisher truncation to top-K tokens biases the floor down | trunc_check.json |
| emb_reader.py | low-RSS plain-read bf16 unembedding (bit-exact vs safetensors) | - |
| exact_min.py | exact free-space minimum (Newton) + Donsker-Varadhan certificate | exact_min*.json |
| ball_min.py | exact minimum over the solid RMSNorm ellipsoid + certificate + KKT points | ball_min_hugo.json, ball_min_points_hugo.npz |
| cert_tilt2.py / cert_floor.py | linear-tilt certificate curves (tight for U/R, loose for P) | cert2_k128p.npz |
| efficiency.py | measured country-token KL / certified minimum for the shift each run made | efficiency_k128p_step150.json |
| make_band.py | pre-run prediction file for the figure | ../data/floor/band.json |
| floor_bound.py, ellip_pairs.py | early ellipsoid searches; the Adam search was WEAK and is retracted (room #6109) | floor_bound_*.json |

research.md = literature, credit, novelty and the numbers above, with room post ids.
