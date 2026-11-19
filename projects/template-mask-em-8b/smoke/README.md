# 0.6B smoke of the mask code (not data of this run)

Qwen/Qwen3-0.6B with an 8-step LoRA (32 training rows) on the Mac mini, only to test the code path.
- mask_invariants_qwen3-0.6b_seed0.json: `test_mask.py --model Qwen/Qwen3-0.6B` on the committed gen.py: all checks pass
  (all-masked == base 0.0; prefix-masked positions 0-2 == base 0.0; padded batch: short row's prefix == base 0.0,
  padded-vs-alone gap 0.39 vs template-vs-full gap 14.69).
- mask_invariants_qwen3-0.6b_UNSHIFTED_MUTANT.json: the same test on a copy of gen.py whose mask ignores the left pad
  (`mk[b, j, 0]` instead of `mk[b, pads[b] + j, 0]`): check 5b fails (padded-vs-alone gap 13.11), "pass": false.

## Continuation smoke (room theory-1119-template-mask-em, T1): `bash smoke/cont_smoke.sh <scratch>`

Qwen/Qwen3-0.6B with a random rank-16 adapter (`smoke/mk_adapter.py`, lora_b ~ N(0, 0.01^2)), Mac mini. Code-path tests of the
memory-fixed, chunked gen.py; not data.
- `cont_identity_cmp.log`: gen.py vs the same file with the memory fixes off (cache grown in 256-slot steps, vocab projection at
  every prompt position, no mx.clear_cache): 8/8 answer files byte-identical (base, full, template, rand; broad + narrow).
- `cont_memory.log`: one 50-row x 600-token batch per set. Peak 5.12 GiB (broad), 5.16 GiB (narrow); weights + adapter
  1.13 GiB active. KV formula: 50 rows x 645 slots x 114,688 B/token (28 layers x 8 KV heads x 128 x 2 x bf16) = 3.44 GiB,
  so 4.57 GiB predicted + ~0.57 GiB decode transients. Same formula at Qwen3-8B (147,456 B/token, 15.26 GiB weights):
  19.8 GiB + transients, ~21-22 GiB; the 2026-11-18 layout (100 rows, cache grown by concatenation) needed 31.5-39 GB.
- `cont_resume_cmp.log`: a run of all six cells (base, full, template, rand, postfix, lastk) killed with SIGKILL after 40 s
  (finished batches kept in gen/<cell>/<kind>.parts/), then restarted: 12/12 answer files byte-identical to an
  uninterrupted run, no .parts dir left.
