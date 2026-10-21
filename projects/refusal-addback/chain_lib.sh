step() {
  name=$1; shift
  while pgrep -f "python (addback|strip).py" > /dev/null; do sleep 10; done  # strictly one model process
  echo "=== $name $(date +%H:%M:%S)" >> logs/chain.log
  $PY "$@" > logs/$name.log 2>&1 &
  pid=$!
  peak=0; rpeak=0
  while kill -0 $pid 2>/dev/null; do
    fp=$(footprint $pid 2>/dev/null | awk '/phys_footprint:/{v=$2; u=$3; if(u=="KB")v=v/1024; if(u=="GB")v=v*1024; if(u=="B")v=0; printf "%d", v; exit}')
    if [ -n "$fp" ] && [ "$fp" -gt "$peak" ]; then peak=$fp; fi
    rss=$(ps -o rss= -p $pid 2>/dev/null | awk '{printf "%d", $1/1024}')
    if [ -n "$rss" ] && [ "$rss" -gt "$rpeak" ]; then rpeak=$rss; fi
    if [ -n "$fp" ] && [ "$fp" -gt 6000 ]; then echo "KILL $name footprint $fp MB" >> logs/chain.log; kill $pid; fi
    sleep 5
  done
  wait $pid; rc=$?
  echo "$name peak_footprint_MB=$peak peak_rss_MB=$rpeak rc=$rc $(date +%H:%M:%S)" >> logs/footprint.log
  [ $rc -eq 0 ] || { echo "FAIL $name rc=$rc" >> logs/chain.log; exit 1; }
}
