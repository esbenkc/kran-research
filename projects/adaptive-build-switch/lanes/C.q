# CPU lane (heldout push search, CPU fp32 + MPS bf16 read), heavy job 2, watchdog 2.2 GB (Vera #13795)
hn_base !$PY watchdog.py --cap-gb 2.2 --log logs/hn_base.log -- $PY heldout_nudge.py --models base=base --chunk 2
hn_stack_s0 !$PY watchdog.py --cap-gb 2.2 --log logs/hn_stack_s0.log -- $PY heldout_nudge.py --models stack_s0=raw/runs/stack_s0_merged.pt --chunk 2
hn_stack_s1 !$PY watchdog.py --cap-gb 2.2 --log logs/hn_stack_s1.log -- $PY heldout_nudge.py --models stack_s1=raw/runs/stack_s1_merged.pt --chunk 2
