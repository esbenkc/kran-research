"""Check VERA_FAST fp32-chunk Fisher against the fp64 Fisher on 3 states: floors for 4 probes."""
import os, sys, numpy as np, time, importlib
sys.argv = ["x", "data/fits/probes.npz", "unproj,proj_k128,rand_k128,lit_k128", "0", "0", "0", "/dev/null"]
import vera_gate_np as G
for i in [321, 1634, 1319]:
    z = np.asarray(G.Z[i], dtype=np.float64)
    G.FAST = False; t = time.time(); F64, p = G.fisher(z); t64 = time.time() - t
    G.FAST = True; t = time.time(); F32, _ = G.fisher(z); t32 = time.time() - t
    e64, Q64 = np.linalg.eigh(F64); e32, Q32 = np.linalg.eigh(F32)
    out = []
    for nm in G.names:
        a = G.A[nm]["a"]
        f = [float(((Q.T @ a) ** 2 / (np.maximum(e, 0) + 1e-8 * e.max())).sum()) for e, Q in ((e64, Q64), (e32, Q32))]
        out.append(f"{nm} g64 {f[0]:.5g} g32 {f[1]:.5g} ratio {f[1]/f[0]:.5f}")
    print(i, f"t64 {t64:.1f}s t32 {t32:.1f}s |F32-F64|/|F64| {np.linalg.norm(F32-F64)/np.linalg.norm(F64):.2e}", "; ".join(out), flush=True)
