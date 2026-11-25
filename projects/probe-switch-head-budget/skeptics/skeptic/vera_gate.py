# Vera: recompute the frozen MONITOR HEALTH gate (first re-pick, relative top-2 gap < 1e-5 x |top|) for every k6 seed
# from the saved first REPICK record (data/runs/<run>.json picks[0], or the laptop log for a killed run).
import json, re, sys, os
proj = sys.argv[1]
def first(run):
    f = os.path.join(proj, 'data/runs', run + '.json')
    if os.path.exists(f):
        return json.load(open(f))['picks'][0], 'data/runs/%s.json' % run
    L = open(os.path.join(proj, 'data/laptop/99ig3z70yx', run + '.log')).read()
    return json.loads(re.search(r'REPICK (\{.*\})', L).group(1)), 'data/laptop/99ig3z70yx/%s.log' % run
for run in ('k6_s0', 'k6_s1', 'k6_s2'):
    p, src = first(run)
    rel = [(b - r) / abs(b) for b, r in p['greedy_best_runnerup']]
    cum = [b / p['D_mon'] for b, _ in p['greedy_best_runnerup']]
    ties = [x < 1e-5 for x in rel]
    print(f"{run} ({src}): heads {p['heads']}; rel gaps " + ' '.join('%.1e' % x for x in rel)
          + f"; ties {ties}; cumulative share removed " + ' '.join('%.5f' % x for x in cum)
          + f"; HEALTHY={not any(ties)}; k<=3 healthy={not any(ties[:3])}")
