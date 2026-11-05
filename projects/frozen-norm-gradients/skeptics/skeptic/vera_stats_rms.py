"""Status rule of PREREG 0b4e178b s4 applied to Vera's own rows; P2 null; P3 ratio; tick band.
Usage: python vera_stats_rms.py TAG [TAG ...]   (TAG e.g. qwen3_A)"""
import sys, json, numpy as np, torch


def boot_med(v, B=10000, seed=0):
    v = np.asarray(v); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (B, len(v)))
    m = np.median(v[idx], 1)
    return float(np.median(v)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def derangement(n, seed=7):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p


if __name__ == '__main__':
    out = {}
    for tag in sys.argv[1:]:
        rows = [json.loads(l) for l in open(f'vera_rows_{tag}.jsonl')]
        meta = json.load(open(f'vera_meta_{tag}.json'))
        read_L = meta['read_L']; status_L = read_L[:3]
        vec = torch.load(f'vera_vec_{tag}.pt')
        o = {'n': len(rows), 'redraws_total': sum(r['redraws'] for r in rows), 'docs_with_redraw': sum(r['redraws'] > 0 for r in rows),
             'skipped': meta['skipped'], 'layers': {}}
        verdicts = []
        for L in read_L:
            c = [r[f'c_{L}'] for r in rows]
            m, lo, hi = boot_med(c)
            v = 'YES' if hi < 0.90 else ('NO' if lo >= 0.90 else 'unsettled')
            if L in status_L: verdicts.append(v)
            d = {'median_c': m, 'ci': [lo, hi], 'verdict': v,
                 'q25_q75': [float(np.percentile(c, 25)), float(np.percentile(c, 75))],
                 'median_ratio': float(np.median([r[f'ratio_{L}'] for r in rows]))}
            if rows[0].get(f'cpred_{L}') is not None:
                cp = [r[f'cpred_{L}'] for r in rows]
                d['median_cpred'] = float(np.median(cp)); d['tick_gap'] = d['median_cpred'] - m
                d['tick_in_band_0.11'] = abs(d['tick_gap']) <= 0.11
                d['median_cos_pred_real'] = float(np.median([r[f'cpredreal_{L}'] for r in rows]))
                d['median_abs_tok_gap'] = float(np.median(np.abs(np.array(cp) - np.array(c))))
                from scipy.stats import spearmanr
                d['spearman_cpred_c'] = float(spearmanr(cp, c)[0])
            # P2 null: cos(g_frz[i], g_real[j]) for a fixed derangement
            gf = vec[L]['frz'].double(); gr = vec[L]['real'].double(); p = derangement(len(gf))
            nul = torch.nn.functional.cosine_similarity(gf, gr[p], 1).numpy()
            d['null_median'] = float(np.median(nul)); d['null_ci'] = list(boot_med(nul)[1:])
            # extra null: random-direction baseline in d dims
            o['layers'][L] = d
        ny = verdicts.count('YES'); nn = verdicts.count('NO')
        o['status'] = 'CONFIRMED' if ny >= 2 else ('REJECTED' if nn >= 2 else 'UNCONFIRMED')
        o['verdicts'] = dict(zip(status_L, verdicts))
        out[tag] = o
        print(tag, json.dumps(o, indent=1))
    json.dump(out, open('vera_stats_' + '_'.join(sys.argv[1:]) + '.json', 'w'), indent=1)
