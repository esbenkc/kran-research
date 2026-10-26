# research_checks (Iris)

- `sep.py` -> `sep.json`, `verify_sep.log`: can ONE position's block-26 input tell probe-on from probe-off?
  `python sep.py ../raw/step2/Qwen3-1.7B/compile_C ../data/prompts.jsonl sep.json` (CPU, sklearn; train 512 x 2, test 128 x 2).
  Answer position 79: AUROC 0.52 (logistic) / 0.50 (2-layer MLP); `<|im_end|>` (pos 71): 0.90.
- `head_attn.py` -> `head_attn.json`: attention from the answer position per head of blocks 26-27, probe off vs on,
  64 TRAIN prompts, CPU fp32. `python head_attn.py <cache dir> <prompts.jsonl> head_attn.json warm_s0=<merged.pt> ...`
- `check_audit.py`: sanity table over `data/audit/*.json` (fast vs full path, identity row, title cell).
