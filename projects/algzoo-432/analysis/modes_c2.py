"""C2 addendum (Kasper #1545): two tests of Felix's oscillator-bank story (denseB1.md s.2).

(a) Event-time write rule: does the max event write a FIXED phase into the max clock (mode k2,
    |lambda| 0.951, 0.746 rad/step)? For samples whose running max arrives at position p and is never
    beaten, measure the phase of z_k2 = (left eigvec k2) . h_{p+d} for d = 1..(10-p). If the event writes
    a fixed phase, the circular concentration R = |mean exp(i phase)| is ~1 within each p, and the phase
    advances by ~0.746 per step of d. Also: does the phase depend on the gap M - (previous max) or x_p?
(b) 'Readout = phase decoder': in the mode basis W_oh = C L (C = W_oh R, complex). For each clock mode k,
    the claim is C[j,k] = a_k * exp(i (phi_k - angle_k * (9 - j))): a rotating template matched to
    'steps since event'. Replace C[:,k] for the claimed clock modes by that 2-parameter (complex a_k)
    template (least squares to C, i.e. to weights, not to model outputs) and measure agreement.

Run: .venv/bin/python modes_c2.py  -> modes_c2.json. Seeds: 41 (pattern), 43 (tests). Never 432432.
"""
import json

import numpy as np

from core import agreement, load_raw, run, sample, second_argmax

W_hi, W_hh, W_oh = load_raw()


def dominant_pattern_modes(seed=41, n=100_000):
    _, H = run(sample(n, seed), W_hi, W_hh, W_oh, return_states=True)
    D = H[:, 6:] > 0
    pats, cnt = np.unique(D.reshape(-1, 16), axis=0, return_counts=True)
    m = pats[cnt.argmax()]
    ev, R = np.linalg.eig(W_hh * m[:, None])
    o = np.argsort(-np.abs(ev))
    ev, R = ev[o], R[:, o]
    return m, ev, R, np.linalg.inv(R), float(cnt.max() / cnt.sum())


def circ_R(ph):
    return float(np.abs(np.mean(np.exp(1j * ph)))) if len(ph) else float("nan")


def main():
    m, ev, R, L, share = dominant_pattern_modes()
    k_clock = 2  # Felix's k2: 0.951 at +0.746 rad/step
    print(f"pattern off={np.where(~m)[0].tolist()} share={share:.3f}; k2 lambda={abs(ev[2]):.3f} "
          f"angle={np.angle(ev[2]):+.3f}")
    x = sample(300_000, 43)
    l, H = run(x, W_hi, W_hh, W_oh, return_states=True)
    p_model = l.argmax(1)
    err = p_model != second_argmax(x)
    p1 = x.argmax(1)
    out = {"pattern_off": np.where(~m)[0].tolist(), "pattern_share": share,
           "k2_lambda_abs": float(abs(ev[2])), "k2_angle": float(np.angle(ev[2])), "event_write": []}

    # (a) phase of the max clock after the max event
    z = np.einsum("k j, b t j -> b t k", L[[k_clock]], H)[:, :, 0]  # (B, 11)
    print("\n(a) max clock phase after max event at position p (max positive, never beaten):")
    print("  p  d  n      R(phase)  mean_phase  R within gap terciles     corr(phase, gap)")
    for p in range(0, 9):
        sel = (p1 == p) & (x.max(1) > 0)
        prevmax = np.maximum(0, x[:, :p].max(1)) if p > 0 else np.zeros(len(x))
        gap = x[:, p] - prevmax
        for d in range(1, 11 - p):
            ph = np.angle(z[sel, p + d])
            g = gap[sel]
            q = np.quantile(g, [1 / 3, 2 / 3])
            Rs = [circ_R(ph[g <= q[0]]), circ_R(ph[(g > q[0]) & (g <= q[1])]), circ_R(ph[g > q[1]])]
            mph = float(np.angle(np.mean(np.exp(1j * ph))))
            # linear-circular association: |corr(gap, cos)|, |corr(gap, sin)|
            c = max(abs(np.corrcoef(g, np.cos(ph - mph))[0, 1]), abs(np.corrcoef(g, np.sin(ph - mph))[0, 1]))
            out["event_write"].append({"p": p, "d": d, "n": int(sel.sum()), "R": circ_R(ph), "mean_phase": mph,
                                       "R_gap_terciles": Rs, "assoc_gap": float(c)})
            if d <= 3 or d == 10 - p:
                print(f"  {p}  {d:2d} {sel.sum():6d}  {circ_R(ph):.3f}   {mph:+.3f}     "
                      f"{Rs[0]:.2f} {Rs[1]:.2f} {Rs[2]:.2f}            {c:.2f}")

    # phase advance per step (from mean phases)
    adv = []
    for p in range(0, 8):
        rows = [r for r in out["event_write"] if r["p"] == p]
        for a, b in zip(rows[1:-1], rows[2:]):
            adv.append(np.angle(np.exp(1j * (b["mean_phase"] - a["mean_phase"]))))
    out["mean_phase_advance_per_step_d>=2"] = float(np.mean(adv))
    print(f"  mean phase advance per step (d>=2): {np.mean(adv):+.3f} rad (mode angle {np.angle(ev[2]):+.3f})")

    # (b) readout as phase decoder
    C = W_oh @ R  # (10,16) complex; W_oh = Re(C L) exactly
    assert np.allclose((C @ L).real, W_oh, atol=1e-8)
    j = np.arange(10)

    def template_fit(col, ang):
        basis = np.exp(-1j * ang * (9 - j))
        a = np.vdot(basis, col) / np.vdot(basis, basis)
        return a * basis

    groups = {"max clock k2,k3": [2, 3], "2nd clocks k4,k5,k7,k8": [4, 5, 7, 8],
              "all complex clocks": [2, 3, 4, 5, 7, 8]}
    out["phase_decoder"] = {}
    print("\n(b) replace readout of clock modes by rotating template a_k exp(-i angle_k (9-j)):")
    for name, ks in groups.items():
        C2 = C.copy()
        rel = []
        for k in ks:
            C2[:, k] = template_fit(C[:, k], np.angle(ev[k]))
            rel.append(float(np.linalg.norm(C2[:, k] - C[:, k]) / np.linalg.norm(C[:, k])))
        W2 = (C2 @ L).real
        q = run(x, W_hi, W_hh, W2).argmax(1)
        a, e = agreement(q, p_model), agreement(q[err], p_model[err])
        out["phase_decoder"][name] = {"agree": a, "agree_err_set": e, "rel_fit_err": rel}
        print(f"  {name:24s} agree {a:.4f} (err-set {e:.4f})  rel fit error per mode {np.round(rel, 2)}")

    # (b2) phase decoder only for early positions j <= J (late positions are read via subcircuit A),
    # with amplitude growth |lambda|^-(9-j) (compensates the decay since the event).
    def template_fit_early(col, lam, J):
        basis = np.exp(-1j * np.angle(lam) * (9 - j)) * np.abs(lam) ** (-(9 - j))
        sel = j <= J
        a = np.vdot(basis[sel], col[sel]) / np.vdot(basis[sel], basis[sel])
        new = col.copy()
        new[sel] = a * basis[sel]
        return new, float(np.linalg.norm(new - col) / np.linalg.norm(col[sel]))

    print("\n(b2) template only for early positions j<=J, amplitude ~ |lambda|^-(9-j); late j exact:")
    out["phase_decoder_early"] = {}
    for name, ks in groups.items():
        for J in (3, 4, 5):
            C2 = C.copy()
            rel = []
            for k in ks:
                C2[:, k], r = template_fit_early(C[:, k], ev[k], J)
                rel.append(r)
            q = run(x, W_hi, W_hh, (C2 @ L).real).argmax(1)
            a, e = agreement(q, p_model), agreement(q[err], p_model[err])
            out["phase_decoder_early"][f"{name} J={J}"] = {"agree": a, "agree_err_set": e, "rel_fit_err": rel}
            print(f"  {name:24s} J={J} agree {a:.4f} (err-set {e:.4f})  rel fit err {np.round(rel, 2)}")
    json.dump(out, open("modes_c2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
