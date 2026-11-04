# MPS lane: training (one heavy job), then MPS evals between runs. Cap multiple 1.0 (D8: pilot_cap1_s0 dGap >= 0.10)
pilot_cap1_s0 done
adv_mid_s0 --arm stack --seed 0 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
cln_s0 --arm penall --seed 0 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
adv_mid_s1 --arm stack --seed 1 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
cln_s1 --arm penall --seed 1 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
cln_s2 --arm penall --seed 2 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
cln_s3 --arm penall --seed 3 --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,2048,3072,4096,5120,6144,7168
