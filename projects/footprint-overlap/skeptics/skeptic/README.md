# Skeptic receipts (Vera)

Independent rebuild of the footprint-overlap result with my own code. numpy / scipy / sklearn only, CPU, < 0.7 GB.
No code from build was read or reused. Set `VERA_PRJ` to the project folder (default assumes the swarm work dir).
The scripts read the local raw files (`raw/screen/base/<tag>_rec_tail.npy`, `<tag>_plain_logits.npy`,
`<tag>_plain.npz`, `raw/behav/<run>/<tag>_plain.npz`); raw/ is not in git (size), see the project README.

| script | what | output |
|---|---|---|
| `vera_overlap.py` | pre-freeze readability scan (A vs none AUROC by position) | stdout |
| `vera_x.py` | x-axis: class per build from raw bits, own overlap fit, secondaries, x-side nulls | `vera_x.json` |
| `vera_share.py` | fooled share + fixed-line R^2 helpers | (library) |
| `vera_law.py` | y-axis + the frozen status rule + D1-D3 reported rows + permutation p | `vera_law_s0at8192_s1at8192.json` |
| `vera_robust.py` | monitor and joint bootstrap, knob grid (pos x C x std), base-gap check | `vera_robust.json` |
| `vera_ctrl.py` | variant and control policies (absolute drop in points, R^2 where valid) | `vera_ctrl.json` |
| `vera_pos1.py` | rebuild of the post-hoc user-token score (not status) + random-direction null | `vera_pos1.out` |

Verdict and numbers: `signoff.md`. Novelty search: `novelty.md`.
