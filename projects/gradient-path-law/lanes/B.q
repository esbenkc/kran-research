# lane B (gradient-path-law). D4 (Felix #15771).
go_B|for k in $(seq 720); do [ -e logs/go ] && break; sleep 5; done
T_s1|./runline.sh T_s1 cue 1 raw/cache/train26_L14
fed_s3|./runline.sh fed_s3 fed 3
Tc_s1|EXTRA='--sampler trainer' ./runline.sh Tc_s1 cue 1 raw/cache/train26_L14
rec_s3|./runline.sh rec_s3 standard 3
Tb_s1|EXTRA='--tis 2' ./runline.sh Tb_s1 cue 1 raw/cache/train26_L14
stackrec_s0|cd stack && ./stackline.sh stackrec_s0 recompute 0
