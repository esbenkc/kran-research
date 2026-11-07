# lane A (gradient-path-law). Gate: logs/go exists (PREREG committed + Vera compute pass).
go_A|for k in $(seq 720); do [ -e logs/go ] && break; sleep 5; done
T_s0|./runline.sh T_s0 cue 0 raw/cache/train26_L14
fed_s2|./runline.sh fed_s2 fed 2
rec_s2|./runline.sh rec_s2 standard 2
decoy_s2|./runline.sh decoy_s2 cue 2 raw/cache/decoy26_L14
stackfed_s3|cd stack && ./stackline.sh stackfed_s3 fed 3
