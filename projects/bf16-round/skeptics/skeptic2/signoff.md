# skeptic2 sign-off — bf16-round

Verdict: REJECTED.

I rebuilt the title answer from saved raw data with my own scripts:
- `checker.py` reads only saved `.npy` arrays. It counts a prompt as changed when any saved uint16 logit/hidden bit differs between probed and unprobed runs. It treats `last.npy` as `(pass, prompt, vocab)`, so warm-up pass 0 cannot hide warm passes 1-3.
- `detector_check.py` reads detector JSON rows. It recomputes accuracy from each row's `correct` field, prompt-pair bit identity from the saved SHA256 rows, and separation from probe-minus-no-probe log-odds.

Key rebuilt numbers:
- Validity positive control: stock R0 changes 64/64 dev prompts on all-position logits. `checker_output.md:9`.
- Title arm R2 `round16`: changes 64/64 dev prompts on all-position logits, and 64/64 on hidden/last arrays. `checker_output.md:10`, `checker_output.md:21`, `checker_output.md:33`.
- Determinism floor: every present A/B pair is 0/64 changed, including R2 probed and unprobed. `checker_output.md:35-59`.
- Metal gate: R2 source contains the q_norm weight-product bfloat cast before rope. Build/Vera checked `round16_noprobe_A.py:866-867`; I separately read the same source during the run.
- Fresh detector trained on R2: 233/256 correct = 91.02%, with 0/128 SHA-identical prompt pairs and 10.509 nats separation. `detector_check.md:12`.
- Null test: under a chance-label detector, `P[Bin(256, 0.5) >= 233] = 3.28e-45`. `detector_check.md:17`.
- Saved detectors on R2: 205/256, 153/256, 176/256. `detector_check.md:9-11`.
- Precision-emulation setting is clean for the saved detectors: each saved detector is exactly 128/256 with 128/128 SHA-identical prompt pairs. `detector_check.md:13-15`.
- Speed clause is not part of title status, but timing has median R2 93.63 ms vs R0 93.97 ms, cost -0.37%, in `raw/timing/timing.json:104-110`.

Definition check:
- The frozen REJECTED rule was: title arm passes Metal gate, changes at least one prompt, and a fresh detector scores at least 75%.
- The title arm passes the Metal gate, changes 64/64 prompts, and the fresh detector scores 91.02%.
- Therefore the answer to the title is no: this explicit 16-bit round does not make the compiled read-only probe invisible to a trained detector on this stack.

Novelty check:
- See `novelty.md`. I found public evidence for Sauers' compiled-probe detector and mechanism audit, but no public test of this specific fix: a split-point 16-bit round applied to both probed and unprobed graphs and checked bit-for-bit on this MPS stack.
