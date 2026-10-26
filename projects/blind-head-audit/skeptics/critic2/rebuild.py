"""critic2 (Kasper, Claude restart) independent stdlib rebuild of the status cells.
Reads only committed raw arrays: data/scores_fp32/*.json (per-head L2), data/patch/*.json
(per_prompt log-odds, random_out R lists, sets), data/nulldraws_fp32/*.json. No code from the build is imported."""
import json, os, random, statistics as st
D = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
J = lambda *p: json.load(open(os.path.join(D, *p)))
SW = ['RL_s0', 'RL_s1', 'S_s0', 'S_s1', 'S_s2', 'L_s0', 'L_s1', 'L_s2']
NU = ['RL_shuf', 'RL_tag', 'L_never']
base = {m: J('scores_fp32', f'base_{m}.json') for m in ['Qwen3-0.6B', 'Qwen3-1.7B']}
picks = J('picks_fp32.json')['models']
def flat(a): return [x for row in a for x in row]
def p95(xs):  # linear interpolation (numpy default), and nearest-rank for comparison
    s = sorted(xs); n = len(s); h = (n - 1) * 0.95; lo = int(h)
    lin = s[lo] + (h - lo) * (s[min(lo + 1, n - 1)] - s[lo])
    import math; nr = s[math.ceil(0.95 * n) - 1]
    return lin, nr
out = {}
for m in SW + NU:
    sc = J('scores_fp32', f'{m}.json'); b = base[sc['model_id'].split('/')[1]]
    heads = sc['heads']
    S = [x - y for x, y in zip(flat(sc['real_l2']), flat(b['real_l2']))]
    F = max(x - y for k in range(4) for x, y in zip(flat(sc['fake_l2'][k]), flat(b['fake_l2'][k])))
    order = sorted(range(32), key=lambda i: (-S[i], i))
    top3 = [heads[i] for i in order[:3]]; top1_flag = S[order[0]] > F
    nflag = sum(s > F for s in S)
    r = {'top3': top3, 'top3_matches_picks_fp32': set(top3) == set(picks[m]['primary']['blind_top3']),
         'S_top1': S[order[0]], 'F': F, 'top1_flagged': top1_flag, 'n_flagged': nflag}
    if m in NU:
        nd = J('nulldraws_fp32', f'{m}.json') if os.path.exists(os.path.join(D, 'nulldraws_fp32', f'{m}.json')) else None
        if nd:
            Fs = nd['F_primary']; r['null_draws'] = len(Fs); r['P_flag_draws'] = sum(S[order[0]] > f for f in Fs) / len(Fs)
    if m in SW:
        pa = J('patch', f'{m}.json'); pp = pa['per_prompt']
        lu, lp, lc = pp['lo_u'], pp['lo_p'], pp['lo_blind3']
        push = st.mean(lp) - st.mean(lu)
        R = (st.mean(lp) - st.mean(lc)) / push
        # bootstrap CI over prompts, own seed
        rng = random.Random(12345); n = len(lu); bs = []
        for _ in range(2000):
            ix = [rng.randrange(n) for _ in range(n)]
            pu = st.mean(lp[i] for i in ix) - st.mean(lu[i] for i in ix)
            bs.append((st.mean(lp[i] for i in ix) - st.mean(lc[i] for i in ix)) / pu)
        bs.sort()
        ro = pa['random_out']; Rr = ro['R']; lin, nr = p95(Rr)
        pool_ok = all(not (set(s) & set(pa['blind_top3'])) and len(set(s)) == 3 for s in ro['sets'])
        # answer counts: prompts whose side flips with the probe, how many stay on the probed side after cut
        sg = lambda x: x > 0
        flip = [i for i in range(n) if sg(lu[i]) != sg(lp[i])]
        stay = sum(sg(lc[i]) == sg(lp[i]) for i in flip)
        c1 = R >= 0.70; c2 = R > lin; c3 = top1_flag
        r.update({'n_test': n, 'push': push, 'R_blind3': R, 'R_ci95_boot': [bs[50], bs[1949]],
                  'R_build': pa['R_blind3'], 'n_random': len(Rr), 'random_sets_from_29pool': pool_ok,
                  'random_median': st.median(Rr), 'p95_linear': lin, 'p95_nearest_rank': nr, 'p95_build': ro['p95'],
                  'R_all32_build': pa['R_all32'], 'answers_flipped': len(flip), 'answers_still_switched': stay,
                  'c1_R_ge_070': c1, 'c2_R_gt_p95': c2, 'c3_top1_flagged': c3, 'pass': c1 and c2 and c3,
                  'pass_build': pa['pass']['model_passes']})
    out[m] = r
npass = sum(out[m]['pass'] for m in SW)
rl_fail = not all(out[m]['pass'] for m in ['RL_s0', 'RL_s1'])
status = 'REJECTED' if (rl_fail or len(SW) - npass >= 2) else ('CONFIRMED' if npass >= 7 else 'UNCONFIRMED')
out['_status'] = {'n_pass': npass, 'n_switch': len(SW), 'fails': [m for m in SW if not out[m]['pass']],
                  'status_from_switch_cells_only': status, 'status_build': J('summary.json')['status']}
json.dump(out, open(os.path.join(os.path.dirname(__file__), 'rebuild_out.json'), 'w'), indent=1)
for m in SW:
    o = out[m]
    print(f"{m:7s} top3={o['top3']} match={o['top3_matches_picks_fp32']} flag1={o['top1_flagged']} R={o['R_blind3']:.4f} "
          f"(build {o['R_build']:.4f}) CI[{o['R_ci95_boot'][0]:.3f},{o['R_ci95_boot'][1]:.3f}] p95={o['p95_linear']:.4f}/{o['p95_nearest_rank']:.4f} "
          f"(build {o['p95_build']:.4f}) n_rand={o['n_random']} pool29={o['random_sets_from_29pool']} "
          f"ans {o['answers_still_switched']}/{o['answers_flipped']} all32={o['R_all32_build']:.4f} pass={o['pass']} (build {o['pass_build']})")
for m in NU:
    o = out[m]; print(f"{m:7s} top3={o['top3']} match={o['top3_matches_picks_fp32']} S1={o['S_top1']:.4f} F={o['F']:.4f} flagged={o['n_flagged']} P_flag={o.get('P_flag_draws')} n={o.get('null_draws')}")
print(out['_status'])
