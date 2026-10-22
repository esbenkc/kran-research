# Step 1 rebuild — skeptic2

Manifest files checked: 84; SHA256 mismatches: 0.

## compile
- none vs none in-process: [0, 0]/64 prompts changed; fresh process 0/64; separate compile 0/64.
- recorder effect on logits: 64/64, frac 0.826687, max_abs 0.5.
- probe: logits 64/64, frac 0.898040, max_abs 1.140625; tail 64/64, frac 0.810401, max_abs 16.0.
  - repeat stability: 0/64 changed in-process.
  - constant changed tail positions top: [{'changed': True, 'max_deviation_across_prompts': 0.0, 'norm': 13.921406745910645, 'position': 2}, {'changed': True, 'max_deviation_across_prompts': 0.0, 'norm': 10.071552276611328, 'position': 1}].
- copy: logits 0/64, frac 0.000000, max_abs 0.0; tail 0/64, frac 0.000000, max_abs 0.0.
  - repeat stability: 0/64 changed in-process.
- twok: logits 64/64, frac 0.898040, max_abs 1.140625; tail 64/64, frac 0.810401, max_abs 16.0.
  - constant changed tail positions top: [{'changed': True, 'max_deviation_across_prompts': 0.0, 'norm': 13.921406745910645, 'position': 2}, {'changed': True, 'max_deviation_across_prompts': 0.0, 'norm': 10.071552276611328, 'position': 1}].
- fp32: logits 64/64, frac 0.896052, max_abs 0.765625; tail 64/64, frac 0.813582, max_abs 16.0.
  - constant changed tail positions top: [{'changed': True, 'max_deviation_across_prompts': 0.0, 'norm': 13.581108093261719, 'position': 1}, {'changed': True, 'max_deviation_across_prompts': 0.0, 'norm': 10.74691104888916, 'position': 2}].

## eager
- none vs none in-process: [0, 0]/64 prompts changed; fresh process 0/64; separate compile 0/64.
- recorder effect on logits: 0/64, frac 0.000000, max_abs 0.0.
- probe: logits 0/64, frac 0.000000, max_abs 0.0; tail 0/64, frac 0.000000, max_abs 0.0.
  - repeat stability: 0/64 changed in-process.
- copy: logits 0/64, frac 0.000000, max_abs 0.0; tail 0/64, frac 0.000000, max_abs 0.0.
  - repeat stability: 0/64 changed in-process.
- twok: logits 0/64, frac 0.000000, max_abs 0.0; tail 0/64, frac 0.000000, max_abs 0.0.
- fp32: logits 64/64, frac 0.919527, max_abs 0.75; tail 64/64, frac 0.848502, max_abs 18.0.
  - constant changed tail positions top: [{'changed': True, 'max_deviation_across_prompts': 0.0, 'norm': 16.835359573364258, 'position': 2}, {'changed': True, 'max_deviation_across_prompts': 0.0, 'norm': 13.7689847946167, 'position': 1}].
