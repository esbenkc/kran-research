#!/bin/bash
# The one heavy lane, serial: gate (start read + calib) -> wait for v1.1 (logs/go holds N) -> planted s0, k16 s0,
# k16 s1 (N updates each), k0 s0 (100) -> planted-text reads for the C contrast. Never edit while it runs.
set -u
cd "$(dirname "$0")"
until [ -f data/runs/pilot/adapter.pt ]; do sleep 10; done
echo "lane: pilot done $(date '+%F %T')"
./gate.sh
echo "lane: gate done, waiting for logs/go (N) $(date '+%F %T')"
until [ -f logs/go ]; do sleep 10; done
N=$(cat logs/go)
echo "lane: T0 $(date '+%F %T') N=$N"
./runline.sh planted_s0 planted 0 "$N"
./runline.sh k16_s0 k16 0 "$N"
./runline.sh k16_s1 k16 1 "$N"
./runline.sh k0_s0 k0 0 100
./gate_c.sh
echo "lane: all done $(date '+%F %T')"
