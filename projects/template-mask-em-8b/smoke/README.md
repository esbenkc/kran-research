# 0.6B smoke of the mask code (not data of this run)

Qwen/Qwen3-0.6B with an 8-step LoRA (32 training rows) on the Mac mini, only to test the code path.
- mask_invariants_qwen3-0.6b_seed0.json: `test_mask.py --model Qwen/Qwen3-0.6B` on the committed gen.py: all checks pass
  (all-masked == base 0.0; prefix-masked positions 0-2 == base 0.0; padded batch: short row's prefix == base 0.0,
  padded-vs-alone gap 0.39 vs template-vs-full gap 14.69).
- mask_invariants_qwen3-0.6b_UNSHIFTED_MUTANT.json: the same test on a copy of gen.py whose mask ignores the left pad
  (`mk[b, j, 0]` instead of `mk[b, pads[b] + j, 0]`): check 5b fails (padded-vs-alone gap 13.11), "pass": false.
