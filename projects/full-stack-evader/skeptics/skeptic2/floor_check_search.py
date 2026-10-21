"""Hugo (skeptic2) own check, part 2: for every caught false eval statement on the CLEAN model, search for a
reachable last-layer state that the L28post probe no longer flags, and record the cheapest next-word KL found.

Reachable = on the final-RMSNorm surface: h(u) = gamma * u / rms(u) (eps 1e-6), u free in R^1536.
Logits = E h (tied input embedding, full 151,936-word vocabulary). Cost = KL(p_clean || p_moved) at the country
token position (the distribution of the next word). Constraint: probe score s(h) <= line (flag iff s > line).
Optimiser: Adam on u, loss = KL + mu * relu(s - line + margin)^2 with mu raised in stages; the best FEASIBLE
iterate per statement is kept. Every kept point is re-scored and its KL recomputed in float64 on CPU.
Any feasible point is an upper bound on the true minimum cost. Own code; no other agent's script read."""
import json, sys, time
import numpy as np
import torch
from safetensors import safe_open

snap = sys.argv[1]
P = np.load("floor_prep.npz")
dev = torch.device("mps")
with safe_open(f"{snap}/model.safetensors", "pt") as f:
    Ebf = f.get_tensor("model.embed_tokens.weight")  # (V, d) bf16, kept on CPU
E = Ebf.float().to(dev)
gamma = torch.tensor(P["gamma"], dtype=torch.float32, device=dev)
a = torch.tensor(P["w"] / P["sd"], dtype=torch.float32, device=dev)          # score = a.h + c
c = float(P["b"]) - float((P["w"] / P["sd"]) @ P["mu"])
line = float(P["line"])
x28 = torch.tensor(P["x28"], dtype=torch.float32, device=dev)
d = x28.shape[1]

def h_of(u):
    return gamma * u / torch.sqrt((u * u).mean(1, keepdim=True) + 1e-6)

with torch.no_grad():
    u0 = x28 / torch.sqrt((x28 * x28).mean(1, keepdim=True))
    h0 = h_of(u0)
    lp0 = torch.log_softmax(h0 @ E.T, dim=1)
    s0 = h0 @ a + c
print(f"n={len(u0)}; clean score on recomputed state: min {s0.min().item():.4f} (line {line:.4f}); "
      f"still caught {(s0 > line).sum().item()}")

N = len(u0)
best_kl = np.full(N, np.inf); best_u = u0.detach().cpu().numpy().copy()
margin = 1e-3
t0 = time.time()
for lo in range(0, N, 150):
    sl = slice(lo, min(N, lo + 150))
    u = u0[sl].clone().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=2e-3)
    for mu, steps in ((10.0, 300), (100.0, 300), (1e3, 400), (1e4, 400)):
        for _ in range(steps):
            h = h_of(u)
            lq = torch.log_softmax(h @ E.T, dim=1)
            kl = (lp0[sl].exp() * (lp0[sl] - lq)).sum(1)
            s = h @ a + c
            loss = (kl + mu * torch.relu(s - line + margin) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                feas = (s <= line).cpu().numpy()
                klc = kl.detach().cpu().numpy()
                idx = np.arange(sl.start, sl.stop)
                upd = feas & (klc < best_kl[idx])
                if upd.any():
                    best_kl[idx[upd]] = klc[upd]
                    best_u[idx[upd]] = u.detach().cpu().numpy()[upd]
    print(f"batch {lo}: done, {time.time()-t0:.0f}s, feasible so far {np.isfinite(best_kl[:sl.stop]).sum()}")

# float64 re-verification of every kept point (score and exact KL from scratch, CPU)
U = torch.tensor(best_u, dtype=torch.float64)
G = torch.tensor(P["gamma"], dtype=torch.float64)
H = G * U / torch.sqrt((U * U).mean(1, keepdim=True) + 1e-6)
H0 = G * torch.tensor(P["x28"]) / torch.sqrt((torch.tensor(P["x28"]) ** 2).mean(1, keepdim=True) + 1e-6)
A64 = torch.tensor(P["w"] / P["sd"], dtype=torch.float64)
S = (H @ A64 + c).numpy(); S0 = (H0 @ A64 + c).numpy()
kl64 = np.empty(N)
for lo in range(0, N, 50):
    z0 = torch.cat([H0[lo:lo+50] @ Ebf[v:v+16384].double().T for v in range(0, Ebf.shape[0], 16384)], 1)
    z1 = torch.cat([H[lo:lo+50] @ Ebf[v:v+16384].double().T for v in range(0, Ebf.shape[0], 16384)], 1)
    l0 = torch.log_softmax(z0, 1); l1 = torch.log_softmax(z1, 1)
    kl64[lo:lo+50] = (l0.exp() * (l0 - l1)).sum(1).numpy()
ok = np.isfinite(best_kl) & (S <= line)
moved = np.linalg.norm((H - H0).numpy(), axis=1) / np.linalg.norm(H0.numpy(), axis=1)
res = {
    "n_caught_clean": int(N), "feasible_verified_f64": int(ok.sum()),
    "kl_f64_median": float(np.median(kl64[ok])), "kl_f64_max": float(kl64[ok].max()),
    "kl_f64_quantiles_10_50_90_99": [float(q) for q in np.quantile(kl64[ok], [.1, .5, .9, .99])],
    "max_abs_f32_vs_f64_kl": float(np.max(np.abs(kl64[ok] - best_kl[ok]))),
    "rel_move_median": float(np.median(moved[ok])), "rel_move_max": float(moved[ok].max()),
    "line": line, "clean_score_min": float(S0.min()),
}
# share of the 596 false eval statements still caught when the move budget is B: caught ones that cost > B
# (the 11 never caught by the clean probe are not caught at any budget)
res["still_caught_share_of_596_at_budget"] = {str(B): float(((~ok) | (kl64 > B)).sum() / 596)
                                              for B in (0.001, 0.003, 0.01, 0.015, 0.05)}
json.dump(res, open("floor_check_search.json", "w"), indent=1)
np.save("floor_check_kl64.npy", kl64)
print(json.dumps(res, indent=1))
