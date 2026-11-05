# lane A (PREREG v1 Compute table)
P_s0 --seed 0 --arm stack --steps 16384 --kl 1 --lam 1 --p-nudge 0.3333 --search-starts 8 --search-steps 60 --nfit 16 --search-chunk 1 --nudge-every 2048 --save-at 1024,2048,3072,4096,5120,6144,7168,8192,9216,10240,11264,12288,13312,14336,15360
alane_A !./alane.sh A > logs/alane_A.out 2>&1
