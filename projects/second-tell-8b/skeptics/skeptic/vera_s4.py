"""Vera's own S4 read (PREREG section 3): principal angles between output subspaces of LoRA deltas.
dW = (alpha/r) B @ A per module (alpha 32, r 16). dW1 = MAIN stage-1 delta (B = 0 at init). dW2 = stage-2 end minus
stage-2 start (stage-1 end; FRESH: zero). Output subspace = top-16 left singular vectors. Modules: o_proj, down_proj
of tail layers 0 and 1 (= blocks 34, 35). Reports smallest and mean of the 16 principal angles (degrees).
Usage: python3 vera_s4.py <adapters_dir> <stage2_suffix e.g. k4>
"""
import sys, os, glob, json, numpy as np, torch

MODS = [f"base_model.model.layers.{l}.{m}" for l in (0, 1) for m in ("self_attn.o_proj", "mlp.down_proj")]
SC = 32 / 16


def factors(sd, mod):
    A = sd[f"{mod}.lora_A.default.weight"].double().numpy()
    B = sd[f"{mod}.lora_B.default.weight"].double().numpy()
    return SC * B, A


def U_lowrank(Bs, As, k=16):
    # left singular vectors of sum_i B_i A_i without forming the dense matrix: QR of [B_1 .. B_n], then small SVD
    Bc = np.concatenate(Bs, 1)
    Ac = np.concatenate(As, 0)
    Q, R = np.linalg.qr(Bc)
    u, s, vt = np.linalg.svd(R @ Ac, full_matrices=False)
    return (Q @ u)[:, :k], s


def angles(U1, U2):
    c = np.clip(np.linalg.svd(U1.T @ U2, compute_uv=False), -1, 1)
    a = np.degrees(np.arccos(c))
    return float(a.min()), float(a.mean()), [float(x) for x in np.sort(a)]


def main(d, suf):
    ld = lambda f: torch.load(f, map_location="cpu", weights_only=False)
    out = {}
    for s in range(5, 10):
        f1 = os.path.join(d, f"main_s{s}_s1.pt")
        if not os.path.exists(f1):
            continue
        s1m = ld(f1)
        row = {}
        for arm in ("main", "shuf", "coin", "fresh"):
            fe = glob.glob(os.path.join(d, f"{arm}_s{s}_*{suf}*.pt"))
            if not fe:
                continue
            end = ld(fe[0])
            start = ld(os.path.join(d, f"{arm}_s{s}_s1.pt")) if arm != "fresh" else None
            per = []
            for mod in MODS:
                B1, A1 = factors(s1m, mod)
                Be, Ae = factors(end, mod)
                if start is not None:
                    Bs, As = factors(start, mod)
                    U2, sv = U_lowrank([Be, -Bs], [Ae, As])
                else:
                    U2, sv = U_lowrank([Be], [Ae])
                U1, _ = U_lowrank([B1], [A1])
                mn, me, al = angles(U1, U2)
                per.append({"module": mod, "smallest": mn, "mean": me, "angles": al, "dW2_top_sv": [float(x) for x in sv[:4]]})
            row[arm] = {"min": float(np.mean([p["smallest"] for p in per])), "mean": float(np.mean([p["mean"] for p in per])),
                        "per_module": per, "file": os.path.basename(fe[0])}
        out[s] = row
        print(s, {a: (round(v["min"], 2), round(v["mean"], 2)) for a, v in row.items()})
    # clause: MAIN mean angle below SHUF-A and COIN in >= 3 of seeds 5-7
    wins = [s for s in (5, 6, 7) if s in out and all(k in out[s] for k in ("main", "shuf", "coin"))
            and out[s]["main"]["mean"] < out[s]["shuf"]["mean"] and out[s]["main"]["mean"] < out[s]["coin"]["mean"]]
    print("S4 clause seeds (main mean angle below both controls):", wins, "holds" if len(wins) >= 3 else "fails")
    return out


if __name__ == "__main__":
    o = main(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        json.dump(o, open(sys.argv[3], "w"), indent=1)
