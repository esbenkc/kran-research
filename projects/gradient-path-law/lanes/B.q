# lane B (gradient-path-law). D4 (Felix #15771).
go_B|for k in $(seq 720); do [ -e logs/go ] && break; sleep 5; done
T_s1|./runline.sh T_s1 cue 1 raw/cache/train26_L14
fed_s3|./runline.sh fed_s3 fed 3
Tc_s1|EXTRA='--sampler trainer' ./runline.sh Tc_s1 cue 1 raw/cache/train26_L14
rec_s3|./runline.sh rec_s3 standard 3
fedread_s3|../../../env/.venv/bin/python watchdog.py --cap-gb 2.8 --log logs/status_fed_s3.log -- ../../../env/.venv/bin/python cache.py --builds none,ref,score --passes plain --splits test --out raw/status/fed_s3 --merged raw/runs/fed_s3_merged.pt && ../../../env/.venv/bin/python status.py fed_s3 >> logs/status_fed_s3.log 2>&1 && ../../../env/.venv/bin/python watchdog.py --cap-gb 2.2 --log logs/carry_fed_s3.log -- ../../../env/.venv/bin/python carry.py fed_s3 --trainer-cache raw/cache/base26_L14
Tb_s1|EXTRA='--tis 2' ./runline.sh Tb_s1 cue 1 raw/cache/train26_L14
stackrec_s0|cd stack && ./stackline.sh stackrec_s0 recompute 0
