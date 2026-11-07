# lane A (gradient-path-law). D4 (Felix #15771): decoy_s2 and stackfed_s3 cut; Tc/Tb/Ta added.
go_A|for k in $(seq 720); do [ -e logs/go ] && break; sleep 5; done
T_s0|./runline.sh T_s0 cue 0 raw/cache/train26_L14
fed_s2|./runline.sh fed_s2 fed 2
Tc_s0|EXTRA='--sampler trainer' ./runline.sh Tc_s0 cue 0 raw/cache/train26_L14
rec_s2|./runline.sh rec_s2 standard 2
Tb_s0|EXTRA='--tis 2' ./runline.sh Tb_s0 cue 0 raw/cache/train26_L14
Ta_s0|./runline.sh Ta_s0 cue 0 raw/cache/train26_L14_none2
