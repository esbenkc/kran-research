"""Vera: s8 pair per Felix #10759: say rise and observed drop over ALL runs (s7 sets); say-gap factor
(trip|not - trip|says) within the prompt-on stratum on exposed window runs (s5 unit). Strata with >=5 per say-group
(exposed window runs, prompt on). Also the run-count bars for the strip."""
import json, numpy as np
from collections import defaultdict
R = [json.loads(l) for l in open('vera_counted_scores.jsonl')]
B = {(r['model'], r['cell'], r['seed']): r for r in map(json.loads, open('../../kran-research/projects/honeypot-talk/results/scores.jsonl'))}
def run(sayf, label):
    C = defaultdict(lambda: defaultdict(list))
    for r in R: C[(r['model'], r['decoy'])][r['prompt']].append(r)
    pred, obs, ks = [], [], []
    for k, c in sorted(C.items()):
        w = [r for r in c['on'] if r['exposed'] and r['window_reply_exists']]
        s = np.array([sayf(r) for r in w]); t = np.array([r['trip'] for r in w])
        if s.sum() >= 5 and (~s).sum() >= 5:
            rise = np.mean([sayf(r) for r in c['on']]) - np.mean([sayf(r) for r in c['off']])
            pred.append(rise * (t[~s].mean() - t[s].mean()))
            obs.append(np.mean([r['trip'] for r in c['off']]) - np.mean([r['trip'] for r in c['on']])); ks.append(k)
    print(f"{label}: {len(ks)} pairs, predicted {np.mean(pred):.3f}, observed {np.mean(obs):.3f}, diff {np.mean(obs)-np.mean(pred):+.3f}")
run(lambda r: bool(r['says_v13']) and r['exposed'] and r['window_reply_exists'], 'Vera says')
run(lambda r: bool(B[(r['model'], r['cell'], r['seed'])].get('says')) and r['exposed'] and r['window_reply_exists'], 'scorer.py says')
off = [r for r in R if r['prompt'] == 'off']; on = [r for r in R if r['prompt'] == 'on']
offx = [r for r in off if r['exposed'] and r['window_reply_exists']]
ns = [r for r in on if r['exposed'] and r['window_reply_exists'] and not B[(r['model'], r['cell'], r['seed'])].get('says')]
print(f"bars: off all {sum(r['trip'] for r in off)}/{len(off)}, off exposed {sum(r['trip'] for r in offx)}/{len(offx)}, on all {sum(r['trip'] for r in on)}/{len(on)}, on non-sayers (scorer.py says) {sum(r['trip'] for r in ns)}/{len(ns)}")
