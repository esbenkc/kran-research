# Vera 1121 receipts (room theory-1121-template-mask-em; seeds 0, 1 judged; seed 2 pending)
Run from skeptics/skeptic (P=../..). No build code is imported.
```
uv run --no-project --with numpy python vera_calls.py 1121/answers.jsonl $P/out_p1 $P/out_p2      # add $P/out_p3 for seed 2
uv run --no-project --with scipy --with numpy python vera_check.py 1121/answers.jsonl --out 1121/vera_check_out.json
python3 1121/vera_diff.py 1121/answers.jsonl 1121/vera_check_out.json $P/analysis.json > 1121/vera_diff.txt
uv run --no-project --with numpy python 1121/vera_extra.py 1121/answers.jsonl $P/out_p1/gen $P/out_p2/gen > 1121/vera_extra.txt
cd 1121 && uv run --no-project --with numpy python vera_whatif.py vera_check_out.json > vera_whatif.txt
uv run --no-project --with tiktoken --with pyyaml --with numpy python vera_reserve.py first_plot_questions.yaml narrow_prompt.txt $P/out_p1 $P/out_p2
```
vera_reserve.py needs the frozen judge yaml (POST-TODO step: curl + sha256 97df0b13...) and the narrow prompt from
PREREG (narrow_prompt.txt). Results: ../signoff.md.
