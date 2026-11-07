# lane B (gradient-path-law). Gate: logs/go exists.
go_B|for k in $(seq 720); do [ -e logs/go ] && break; sleep 5; done
T_s1|./runline.sh T_s1 cue 1 raw/cache/train26_L14
fed_s3|./runline.sh fed_s3 fed 3
rec_s3|./runline.sh rec_s3 standard 3
stackrec_s0|cd stack && ./stackline.sh stackrec_s0 recompute 0
