"""Iris (T5): bracket the per-statement price floor at the probed country token, for each probe, BEFORE any fine-tune.

Setting. z = post-norm country-token state (readout 29), logits l = E z (Qwen2.5-1.5B tied unembedding E, no bias),
p = softmax(l). A probe scores s(z) = w . (z - mu) + b (probes.npz, sd = 1). The clean line t = 2nd-highest score of
the 200 refit-split TRUE statements (1% FPR, the 1019-1023 rule). A caught false statement i (s_i > t) must move its
state by d with w . d <= -D_i, D_i = s_i - t. Its output change at that token is f_i(d) = KL(p_i || softmax(l_i + E d)).

LOWER bound LB_i (certified; valid for every reachable post-norm state, hence for any fine-tune):
  Donsker-Varadhan: for ANY distribution q over the vocabulary,
      f_i(d) = log E_p[exp(e.d)] - E_p[e.d] >= g . d - KL(q || p_i),   g = E^T (q - p_i).
  Split g = -nu w + r with r perpendicular to w (nu = -g.w / |w|^2). For w.d = -D' (D' >= D_i) and nu >= 0:
      g . d = nu D' + r . d >= nu D_i - sum_j |r_j| |d_j|.
  Every post-norm state lies on the RMSNorm ellipsoid sum_j (z_j / gamma_j)^2 <= dim, so |z_j| <= sqrt(dim) |gamma_j|
  and |d_j| <= 2 sqrt(dim) |gamma_j|. Hence
      LB_i = max(0, nu D_i - KL(q || p_i) - 2 sqrt(dim) sum_j |r_j gamma_j|).
  We pick q = softmax(l_i + E d_hat), d_hat = the minimiser of f_i on the hyperplane w.d = -D_i found by L-BFGS (the
  hyperplane problem is convex because the logits are linear in z; at its optimum r = 0 and LB_i = f_i(d_hat) by
  strong duality). LB_i is computed exactly in float64 over all 151,936 tokens; the search may be approximate,
  that only loosens the bound, never invalidates it.
UPPER bound UB_i (a reachable state that crosses the line): search over the pre-norm vector x, z = gamma x / rms(x)
  (the RMSNorm ellipsoid, eps = 1e-6), Adam on KL + mu * relu(s - t + margin)^2; the KL and score of the final state
  are re-computed exactly in float64 over the full vocabulary. UB_i = that KL if the final state crosses, else NaN.
Searches use a union vocabulary (every token in the top-2048 of any clean eval state) to save memory; only the
final certificate / evaluation uses the full vocabulary, so both bounds stay exact statements about the full model.
Frontier: sort LB_i (UB_i) ascending; hiding the m cheapest caught statements gives catch (n_caught - m) / 596 at
mean country-token KL sum_{cheapest m} / 1192 (the unit of eval_kl_country_pos: mean over all 1192 eval statements).
Output: iris_checks/floor_bound_<tag>.json + .npz (per-statement LB, UB, D, nu, residual term).
Run: venv1024/bin/python iris_checks/floor_bound.py --probes data/fits/probes.npz --names unproj,proj,rand,lit_k<k*>
"""
import argparse, json, os, time
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAP = next(Path.home().glob('.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*')) / 'model.safetensors'
V, DIM, CH, EPS = 151936, 1536, 8192, 1e-6
_f = safe_open(str(SNAP), 'pt')
GAM = _f.get_tensor('model.norm.weight').double().numpy()
_E = _f.get_slice('model.embed_tokens.weight')


def chunks():
    for a in range(0, V, CH):
        yield a, _E[a:min(V, a + CH)].double().numpy()


def full_logits(Z):
    """(n, DIM) float64 -> (n, V) float64 logits, streamed."""
    return np.concatenate([Z @ Ec.T for _, Ec in chunks()], 1)


def logsm(L):
    m = L.max(1, keepdims=True)
    return L - (m + np.log(np.exp(L - m).sum(1, keepdims=True)))


def ET_times(Q):
    """E^T Q^T for Q (n, V) -> (n, DIM), streamed float64."""
    out = np.zeros((Q.shape[0], DIM))
    for a, Ec in chunks():
        out += Q[:, a:a + len(Ec)] @ Ec
    return out


def load_states():
    zp = ROOT / 'data/clean/z_post.npy'
    Z = np.load(zp if zp.exists() else HERE / 'inputs/clean_country_R29.npy').astype(np.float64)
    split = json.load(open(ROOT / 'data/ref/split.json'))
    y = np.array([1 - r['true'] for r in split])
    sp = np.array([r['split'] for r in split])
    return Z, y, sp, ('z_post' if zp.exists() else 'inputs/clean_country_R29')


def certify(Zc, w, D, Dhat, LP0):
    """exact float64 LB for states Zc (n, DIM) with search moves Dhat (n, DIM); also f at the search point."""
    LP1 = logsm(full_logits(Zc + Dhat))
    Q = np.exp(LP1)
    P = np.exp(LP0)
    f_hat = (P * (LP0 - LP1)).sum(1)
    kl_qp = (Q * (LP1 - LP0)).sum(1)
    g = ET_times(Q - P)
    nu = -(g @ w) / (w @ w)
    r = g + nu[:, None] * w[None]
    res = 2 * np.sqrt(DIM) * (np.abs(r) * np.abs(GAM)[None]).sum(1)
    lb = np.maximum(0.0, nu * D - kl_qp - res)
    lb[nu < 0] = 0.0
    return lb, f_hat, nu, res, kl_qp


def certify_ball(Zc, w, D, Zq, LP0):
    """exact float64 LB over the SOLID RMSNorm ellipsoid (convex hull of every reachable post-norm state):
    for any q, KL(p||p_z') >= g.(z' - z) - KL(q||p), g = E^T(q - p). Minimise the linear term over
    {z' = gamma*u : |u| <= sqrt(dim), w.z' <= w.z - D} in closed form. q = softmax(E Zq)."""
    LP1 = logsm(full_logits(Zq))
    Q = np.exp(LP1)
    P = np.exp(LP0)
    kl_qp = (Q * (LP1 - LP0)).sum(1)
    g = ET_times(Q - P)
    R = np.sqrt(DIM)
    gt = g * GAM[None]
    at = w * GAM
    lb = np.zeros(len(Zc))
    for i in range(len(Zc)):
        c = w @ Zc[i] - D[i]
        na2 = at @ at
        if c < -R * np.sqrt(na2):
            lb[i] = np.inf
            continue
        u0 = -R * gt[i] / np.linalg.norm(gt[i])
        if at @ u0 <= c:
            lin = -R * np.linalg.norm(gt[i])
        else:
            proj = (gt[i] @ at) / na2
            perp = np.linalg.norm(gt[i] - proj * at)
            lin = c * proj - np.sqrt(max(0.0, R * R - c * c / na2)) * perp
        lb[i] = max(0.0, lin - g[i] @ Zc[i] - kl_qp[i])
    return lb, kl_qp


def search_hyperplane(Zc, w, D, Eu, steps, dev):
    """L-BFGS on the union vocabulary: min_d KL(p||softmax(l+E d)) s.t. w.d = -D. Returns d_hat (n, DIM) float64."""
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    wh = wt / wt.norm()
    Zt = torch.tensor(Zc, dtype=torch.float32, device=dev)
    Dt = torch.tensor(D, dtype=torch.float32, device=dev)
    L0 = Zt @ Eu.T
    lp0 = L0.log_softmax(-1)
    p0 = lp0.exp()
    base = -Dt[:, None] * wt[None] / (wt @ wt)
    Y = torch.zeros_like(Zt, requires_grad=True)
    opt = torch.optim.LBFGS([Y], lr=1, max_iter=steps, history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-10, tolerance_change=1e-14)

    def dvec():
        return base + Y - (Y @ wh)[:, None] * wh[None]

    def closure():
        opt.zero_grad()
        lp = (L0 + dvec() @ Eu.T).log_softmax(-1)
        loss = (p0 * (lp0 - lp)).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return dvec().cpu().double().numpy()


def search_ellipsoid(Zc, w, mu, b, t, Eu, steps, dev, margin):
    gam = torch.tensor(GAM, dtype=torch.float32, device=dev)
    wt = torch.tensor(w, dtype=torch.float32, device=dev)
    mut = torch.tensor(mu, dtype=torch.float32, device=dev)
    Zt = torch.tensor(Zc, dtype=torch.float32, device=dev)
    lp0 = (Zt @ Eu.T).log_softmax(-1)
    p0 = lp0.exp()
    X = (Zt / gam).clone().requires_grad_(True)
    opt = torch.optim.Adam([X], lr=2e-3)
    for s in range(steps):
        pen = 10.0 ** (1 + 4 * s / max(1, steps - 1))  # 1e1 -> 1e5
        z = gam * X / torch.sqrt((X * X).mean(-1, keepdim=True) + EPS)
        lp = (z @ Eu.T).log_softmax(-1)
        kl = (p0 * (lp0 - lp)).sum(-1)
        sc = (z - mut) @ wt + b
        loss = (kl + pen * torch.relu(sc - t + margin) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = gam * X / torch.sqrt((X * X).mean(-1, keepdim=True) + EPS)
        return z.cpu().double().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probes', default=str(ROOT / 'data/fits/probes.npz'))
    ap.add_argument('--names', default='unproj,proj,rand')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lbfgs', type=int, default=200)
    ap.add_argument('--adam', type=int, default=400)
    ap.add_argument('--limit', type=int, default=0, help='only the first N caught statements (smoke)')
    ap.add_argument('--ball', action='store_true', help='also the (loose) solid-ellipsoid DV certificate')
    ap.add_argument('--dev', default='mps' if torch.backends.mps.is_available() else 'cpu')
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device(a.dev)
    Z, y, sp, zsrc = load_states()
    ev = np.where(sp == 'eval')[0]
    evf = ev[y[ev] == 1]
    line_set = np.where((sp == 'refit') & (y == 0))[0]
    # union vocabulary for the searches (top-2048 tokens of every clean eval-false state), gathered chunk by chunk
    keep = np.zeros(V, bool)
    for s0 in range(0, len(evf), 64):
        L = full_logits(Z[evf[s0:s0 + 64]])
        keep[np.unique(np.argpartition(-L, 2048, 1)[:, :2048])] = True
        del L
    uni = np.where(keep)[0]
    Eu = torch.tensor(np.concatenate([Ec[keep[a:a + len(Ec)]] for a, Ec in chunks()]), dtype=torch.float32, device=dev)
    pz = np.load(a.probes)
    out = {'z_source': zsrc, 'probes_file': a.probes, 'union_vocab': int(len(uni)), 'n_eval_all': int(len(ev)),
           'n_eval_false': int(len(evf)), 'args': vars(a), 'probes': {}}
    arrays = {}
    for name in a.names.split(','):
        w = pz[f'{name}__w'].astype(np.float64) / pz[f'{name}__sd'].astype(np.float64)
        mu = pz[f'{name}__mu'].astype(np.float64)
        b = float(pz[f'{name}__b'])
        s_line = (Z[line_set] - mu) @ w + b
        t = float(np.sort(s_line)[::-1][1])
        s = (Z[evf] - mu) @ w + b
        caught = np.where(s > t)[0]
        if a.limit:
            caught = caught[:a.limit]
        D = s[caught] - t
        LB = np.zeros(len(caught)); FH = np.zeros(len(caught)); NU = np.zeros(len(caught)); RES = np.zeros(len(caught))
        UB = np.full(len(caught), np.nan)
        LBB = np.zeros(len(caught))
        # second-order floor with the exact per-statement Fisher is Bruno/Hugo/Vera's table; here the bracket only
        for s0 in range(0, len(caught), a.batch):
            idx = caught[s0:s0 + a.batch]
            Zc = Z[evf[idx]]
            Db = D[s0:s0 + a.batch]
            LP0 = logsm(full_logits(Zc))
            if a.lbfgs:
                dh = search_hyperplane(Zc, w, Db, Eu, a.lbfgs, dev)
                lb, fh, nu, res, _ = certify(Zc, w, Db, dh, LP0)
                LB[s0:s0 + len(idx)], FH[s0:s0 + len(idx)], NU[s0:s0 + len(idx)], RES[s0:s0 + len(idx)] = lb, fh, nu, res
            ze = search_ellipsoid(Zc, w, mu, b, t, Eu, a.adam, dev, margin=1e-3 * max(1.0, float(np.median(D))))
            se = (ze - mu) @ w + b
            lp1 = logsm(full_logits(ze))
            kle = (np.exp(LP0) * (LP0 - lp1)).sum(1)
            UB[s0:s0 + len(idx)] = np.where(se <= t, kle, np.nan)
            if a.ball:
                LBB[s0:s0 + len(idx)], _ = certify_ball(Zc, w, Db, ze, LP0)
            print(json.dumps({'probe': name, 'done': s0 + len(idx), 'of': len(caught),
                              'LBball_med': float(np.median(LBB[:s0 + len(idx)])),
                              'hyper_LB_med': float(np.median(LB[:s0 + len(idx)])),
                              'hyper_f_med': float(np.median(FH[:s0 + len(idx)])),
                              'UB_med': float(np.nanmedian(UB[:s0 + len(idx)])),
                              'sec': round(time.time() - t0, 1)}), flush=True)
            if dev.type == 'mps':
                torch.mps.empty_cache()
        need = max(0, len(caught) - int(np.floor(0.20 * len(evf))))

        def pooled(x):
            x = np.sort(np.nan_to_num(x, nan=np.inf))
            return float(x[:need].sum() / len(ev)) if need else 0.0

        out['probes'][name] = {
            'line': t, 'clean_catch': float(len(np.where(s > t)[0]) / len(evf)), 'n_caught_used': int(len(caught)),
            'D_median': float(np.median(D)), 'need_to_hide_for_20pct': int(need),
            'LBball_quantiles': dict(zip(['p10', 'p50', 'p90'], np.quantile(LBB, [.1, .5, .9]).tolist())),
            'pooled_floor_LBball_country_kl': pooled(LBB),
            'ratio_UB_over_LBball_median': float(np.nanmedian(UB / np.where(LBB > 0, LBB, np.nan))),
            'hyper_LB_quantiles': dict(zip(['p10', 'p50', 'p90'], np.quantile(LB, [.1, .5, .9]).tolist())),
            'hyperplane_f_quantiles': dict(zip(['p10', 'p50', 'p90'], np.quantile(FH, [.1, .5, .9]).tolist())),
            'certificate_gap_median': float(np.median(FH - LB)), 'residual_term_median': float(np.median(RES)),
            'UB_quantiles': dict(zip(['p10', 'p50', 'p90'], np.nanquantile(UB, [.1, .5, .9]).tolist())),
            'UB_crossed_share': float(np.isfinite(UB).mean()),
            'pooled_floor_LB_country_kl': pooled(LB),
            'pooled_floor_UB_country_kl': pooled(UB),
        }
        arrays.update({f'{name}__idx': evf[caught], f'{name}__D': D, f'{name}__LB': LB, f'{name}__hyper_f': FH,
                       f'{name}__nu': NU, f'{name}__res': RES, f'{name}__UB': UB,
                       f'{name}__LBball': LBB})
        print(json.dumps({name: out['probes'][name]}), flush=True)
    out['seconds'] = round(time.time() - t0, 1)
    json.dump(out, open(HERE / f'floor_bound_{a.tag}.json', 'w'), indent=1)
    np.savez(HERE / f'floor_bound_{a.tag}.npz', **arrays)


if __name__ == '__main__':
    main()
