# Figure 1 (Owain), 3-seed version (room oracle-1118-live-kl-off-cont)

- Example: test row 2140 (prompt cd0586), fixed on Nov 17 before seeds 1 and 2 existed (pick rule in room post #25269; pick.py and pick_1200.txt are the Nov 17 pick).
- Build: `<env python> figure1/gen.py --proj projects/live-kl-off --blog <blog clone>` then, in the blog clone, `node _thumbs/figure1.mjs live-kl-off-fig1`.
- Sources: data/runs/kl0_s{0,1,2}/reads/c_{1200,900,700}.npy, data/parent/live_k4/reads/c_1200.npy, data/parent/cached_d010_s0/reads/c_1200.npy, data/summary.json (gen.py asserts every switch equals summary.json), inputs/prompts.jsonl, data/pool/test/cols.npz.
- Numbers: data.json. Alt text: alt.txt. PNG-only cold read: coldread_final.json. Last night's single-seed version: 1117/.
