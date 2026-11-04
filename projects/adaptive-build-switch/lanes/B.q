# MPS lane B: matched cleans (2 MPS training lanes at ~2.0 GB each; CPU search lane paused, Bruno 17:39)
cln_s0 --arm penall --seed 0 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
cln_s1 --arm penall --seed 1 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
cln_s2 --arm penall --seed 2 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
# Felix #13858: layouts for cln_s1/s2 first, then searches cleans
evall !./eval_batch.sh "base:0 cln_s1:8192 cln_s2:8192" "cln_s1=raw/runs/cln_s1_merged.pt,cln_s2=raw/runs/cln_s2_merged.pt" evall; $PY watchdog.py --cap-gb 2.5 --log logs/pool_check.log -- $PY pool_check.py --runs cln_s1:stack_s1,cln_s2:stack_s0 >> logs/pool_check.log 2>&1
# auditor SKIPPED (Felix #13952: D14 start-by-20:20 not met, D17 fallback not needed); panel/reads/audit removed
