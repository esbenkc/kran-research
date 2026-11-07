# lane A (gradient-path-law). D4 (Felix #15771): decoy_s2 and stackfed_s3 cut; Tc/Tb/Ta added.
go_A|for k in $(seq 720); do [ -e logs/go ] && break; sleep 5; done
T_s0|./runline.sh T_s0 cue 0 raw/cache/train26_L14
fed_s2|./runline.sh fed_s2 fed 2
Tc_s0|EXTRA='--sampler trainer' ./runline.sh Tc_s0 cue 0 raw/cache/train26_L14
rec_s2|./runline.sh rec_s2 standard 2
fedread_s2|../../../env/.venv/bin/python watchdog.py --cap-gb 2.8 --log logs/status_fed_s2.log -- ../../../env/.venv/bin/python cache.py --builds none,ref,score --passes plain --splits test --out raw/status/fed_s2 --merged raw/runs/fed_s2_merged.pt && ../../../env/.venv/bin/python status.py fed_s2 >> logs/status_fed_s2.log 2>&1 && ../../../env/.venv/bin/python watchdog.py --cap-gb 2.2 --log logs/carry_fed_s2.log -- ../../../env/.venv/bin/python carry.py fed_s2 --trainer-cache raw/cache/base26_L14
randprefix|../../../env/.venv/bin/python watchdog.py --cap-gb 2.2 --log logs/randprefix.log -- ../../../env/.venv/bin/python randprefix.py --runs Tc_s0,Tc_s1,fed_s2,fed_s3,T_s1 --k 8
Tb_s0|EXTRA='--tis 2' ./runline.sh Tb_s0 cue 0 raw/cache/train26_L14
Ta_s0|./runline.sh Ta_s0 cue 0 raw/cache/train26_L14_none2
