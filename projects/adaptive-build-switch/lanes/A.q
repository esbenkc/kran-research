# MPS lane: training (one heavy job), then MPS evals between runs. Cap multiple 1.0 (D8: pilot_cap1_s0 dGap >= 0.10)
pilot_cap1_s0 done
adv_mid_s0 --arm stack --seed 0 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
vera_slot !rm -f lanes/vera_done; for i in $(seq 1 120); do [ -f lanes/vera_done ] && break; sleep 5; done; echo "vera slot over $(date +%T)"
# held-out push search for untrained + twins before adv_mid_s1 (lane B is the critical path; device timing: base cpu, twins mps)
hn_base !$PY watchdog.py --cap-gb 2.2 --log logs/hn_base.log -- $PY heldout_nudge.py --models base=base --chunk 1 --threads 4 >> logs/hn_base.log 2>&1
hn_stack_s0 !for i in $(seq 1 360); do [ -f raw/runs/stack_s0_merged.pt ] && break; sleep 10; done; sleep 30; $PY watchdog.py --cap-gb 2.2 --log logs/hn_stack_s0.log -- $PY heldout_nudge.py --models stack_s0=raw/runs/stack_s0_merged.pt --chunk 1 --threads 4 --device mps >> logs/hn_stack_s0.log 2>&1
hn_stack_s1 !for i in $(seq 1 360); do [ -f raw/runs/stack_s1_merged.pt ] && break; sleep 10; done; sleep 30; $PY watchdog.py --cap-gb 2.2 --log logs/hn_stack_s1.log -- $PY heldout_nudge.py --models stack_s1=raw/runs/stack_s1_merged.pt --chunk 1 --threads 4 --device mps >> logs/hn_stack_s1.log 2>&1
adv_mid_s1 --arm stack --seed 1 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
# after adv_mid_s1: held-out push search (device per the timing above)
hn_adv_mid_s0 !for i in $(seq 1 360); do [ -f raw/runs/adv_mid_s0_merged.pt ] && break; sleep 10; done; sleep 30; $PY watchdog.py --cap-gb 2.2 --log logs/hn_adv_mid_s0.log -- $PY heldout_nudge.py --models adv_mid_s0=raw/runs/adv_mid_s0_merged.pt --chunk 1 --threads 4 >> logs/hn_adv_mid_s0.log 2>&1
hn_adv_mid_s1 !for i in $(seq 1 360); do [ -f raw/runs/adv_mid_s1_merged.pt ] && break; sleep 10; done; sleep 30; $PY watchdog.py --cap-gb 2.2 --log logs/hn_adv_mid_s1.log -- $PY heldout_nudge.py --models adv_mid_s1=raw/runs/adv_mid_s1_merged.pt --chunk 1 --threads 4 >> logs/hn_adv_mid_s1.log 2>&1
hn_cln_s0 !for i in $(seq 1 360); do [ -f raw/runs/cln_s0_merged.pt ] && break; sleep 10; done; sleep 30; $PY watchdog.py --cap-gb 2.2 --log logs/hn_cln_s0.log -- $PY heldout_nudge.py --models cln_s0=raw/runs/cln_s0_merged.pt --chunk 1 --threads 4 >> logs/hn_cln_s0.log 2>&1
