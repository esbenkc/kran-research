"""Weight-only audit of the cue-size-law LoRA adapters: does the delta's top input-side singular direction single out
the true planted cue among 1,000 random cues built the same way?

No behaviour and no training data: the audit reads only (a) the adapter tensors and (b) the suspected cue (the
'named suspect'), plus the frozen base weights of blocks 26-27 to map a cue into each LoRA module's input space.

Candidate cue c (true, 7 planted keys u1110-u1116, 1,000 nulls): the cue-on rows 0-2 as rl.py feeds them
(h0_p + alpha ||h0_p|| u_p on the cue rows, bf16 cast unless --no-cast; real cue = the compiled-build rows).
Its difference in each module's input space at rows 0-2 (rows 0-2 are the same for every prompt and causal
attention keeps them prompt-independent, so a 3-token base forward is exact):
  in26  = N_in26(h_on) - N_in26(h)                 read by block-26 q, k, v
  mid26 = N_post26(mid_on) - N_post26(mid)         read by block-26 gate, up   (mid = h + attn26)
  in27  = N_in27(h27_on) - N_in27(h27)             read by block-27 q, k, v    (h27 = block-26 output)
Null u' = 1,000 Gaussian directions (numpy seed 20261112, Felix #20536; rows unit-norm like data/cue_u.npy), planted with the SAME rows,
alpha (or per-row norm for the real cue) and cast as the run's true cue.

For a module set M sharing one input space, dW_m = (lora_alpha / r) B_m A_m and G = sum_m dW_m^T dW_m (1024 x 1024);
v_1..v_k = top eigenvectors of G (= top input singular vectors of the stacked delta). Statistics of a cue
(D = its rows, p over the cue rows):
  top{k}  = sum_p sum_{i<=k} (v_i . c_p)^2 / sum_p |c_p|^2      (k = 1, 2, 4, 8, 16; energy on the top-k input directions)
  At update 0 (B = 0) the delta is 0: every score is 0, all candidates tie, and rank = 1,001 by the >= rule.
  gain    = sum_p c_p^T G c_p / sum_p |c_p|^2                   (how strongly the delta responds to the cue)
  aonly   = sum_p sum_m |A_m c_p|^2 / sum_p |c_p|^2            (A side only, no B)
rank = 1 + #{null u' with score >= true score}; z = (true - mean null) / sd null.

    NC_DEV=cpu env/.venv/bin/python audit.py            # every adapter under raw/parent_runs and raw/runs/*/
Writes data/audit/<run>.json (per adapter: step, x, T, every module-set x statistic: true, rank, z, null p99, key
scores) and data/audit/<run>_null.npz (the 1,000 null scores per adapter, qkv26 statistics).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("NC_DEV", "cpu")
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tail import Rows, load_tail  # noqa: E402

SCALE = 32 / 16  # lora_alpha / r (rl.py defaults; checked per run)
N_NULL = 1000
KEYS = ["u1110", "u1111", "u1112", "u1113", "u1114", "u1115", "u1116"]
KEYFILE = {k: ("data/cue_u.npy" if k == "u1110" else f"data/cue_u_{k[1:]}.npy") for k in KEYS}
# module sets: name -> (input space, [(block, module)])
SETS = {
    "qkv26": ("in26", [(0, "self_attn.q_proj"), (0, "self_attn.k_proj"), (0, "self_attn.v_proj")]),
    "q26": ("in26", [(0, "self_attn.q_proj")]),
    "k26": ("in26", [(0, "self_attn.k_proj")]),
    "v26": ("in26", [(0, "self_attn.v_proj")]),
    "kv26": ("in26", [(0, "self_attn.k_proj"), (0, "self_attn.v_proj")]),
    "gate26": ("mid26", [(0, "mlp.gate_proj")]),
    "up26": ("mid26", [(0, "mlp.up_proj")]),
    "gu26": ("mid26", [(0, "mlp.gate_proj"), (0, "mlp.up_proj")]),
    "qkv27": ("in27", [(1, "self_attn.q_proj"), (1, "self_attn.k_proj"), (1, "self_attn.v_proj")]),
}
STATS = ["top1", "top2", "top4", "top8", "top16", "gain", "aonly"]


def null_dirs() -> np.ndarray:
    g = np.random.default_rng(20261112).standard_normal((N_NULL, 3, 1024))
    return (g / np.linalg.norm(g, axis=-1, keepdims=True)).astype(np.float32)


class Spaces:
    """Base blocks 26-27 (fp32 on CPU) and the 3-row forward that maps cue-on rows into each module's input space."""

    def __init__(self):
        _, tail, _ = load_tail(2)
        self.tail = tail.float().eval()
        self.w_out = torch.zeros(1, 1024)
        cache = HERE / "raw" / "cache" / "eager_rec_tail.npy"  # rows 0-2 are the same for every prompt (check_rows.json)
        self.h0_u16 = (np.ascontiguousarray(Rows(cache).a[0, :3]) if cache.exists()
                       else np.load(HERE / "data" / "h0_rows012_u16.npy"))  # committed copy of the same 3 rows
        self.h0 = torch.from_numpy(self.h0_u16.view(np.int16)).view(torch.bfloat16).float()
        self.cap = {}
        L0, L1 = self.tail.layers
        L0.input_layernorm.register_forward_hook(lambda m, i, o: self.cap.__setitem__("in26", o.detach()))
        L0.post_attention_layernorm.register_forward_hook(lambda m, i, o: self.cap.__setitem__("mid26", o.detach()))
        L1.input_layernorm.register_forward_hook(lambda m, i, o: self.cap.__setitem__("in27", o.detach()))

    @torch.no_grad()
    def embed(self, rows: torch.Tensor, bs: int = 256) -> dict:
        """rows [N, 3, 1024] fp32 block-26 inputs -> {space: [N, 3, 1024]} module inputs."""
        out = {s: [] for s in ("in26", "mid26", "in27")}
        for s in range(0, len(rows), bs):
            self.tail(rows[s:s + bs], self.w_out)
            for k in out:
                out[k].append(self.cap[k].float())
        return {k: torch.cat(v) for k, v in out.items()}

    @torch.no_grad()
    def base(self) -> dict:
        return {k: v[0] for k, v in self.embed(self.h0[None]).items()}

    def planted(self, U: np.ndarray, alpha: float, rows, cast: bool) -> torch.Tensor:
        """U [N, 3, 1024] unit rows -> cue-on rows as fed [N, 3, 1024] fp32 (rl.py cue_rows / cue_rows_fp32)."""
        u = torch.from_numpy(U).float()
        m = torch.zeros(3, 1)
        m[list(rows)] = 1.0
        on = self.h0[None] + alpha * self.h0.norm(dim=1, keepdim=True)[None] * u * m
        return on.to(torch.bfloat16).float() if cast else on

    def planted_norm(self, U: np.ndarray, row_norms: np.ndarray, rows, cast: bool) -> torch.Tensor:
        u = torch.from_numpy(U).float()
        m = torch.zeros(3, 1)
        m[list(rows)] = 1.0
        on = self.h0[None] + torch.from_numpy(row_norms).float()[None, :, None] * u * m
        return on.to(torch.bfloat16).float() if cast else on


def run_spec(meta: dict) -> dict:
    a = meta["args"]
    rows = tuple(int(r) for r in str(a.get("cue_rows") or "0,1,2").split(","))
    real = a.get("cue_dir") == "real"
    m = re.search(r"cue_u(?:_(\d+))?\.npy", a.get("u_file", "data/cue_u.npy"))
    key = "real" if real else ("uneg" if "neg" in a.get("u_file", "") else ("u" + (m.group(1) or "1110")))
    return {"rows": rows, "alpha": float(a.get("cue_alpha", 0.0)), "cast": not a.get("no_cast", False), "real": real,
            "key": key, "arm": a.get("arm", "cue"), "x": meta.get("cue_size", {}).get("x"),
            "rank": int(a.get("rank", 16)), "lora_alpha": float(a.get("alpha", 32))}


def candidates(sp: Spaces, spec: dict, base: dict, U0: np.ndarray) -> tuple[list[str], dict]:
    """Rows -> module-space differences for [true, 7 keys, 1,000 nulls]. Returns names and {space: [N, 3, 1024]}."""
    names, on = [], []
    if spec["real"]:
        real_on = torch.from_numpy(np.load(HERE / "data" / "real_rows012_u16.npy").view(np.int16)).view(
            torch.bfloat16).float()
        rn = (real_on - sp.h0).norm(dim=1).numpy()
        names.append("true")
        on.append(real_on[None])
        Uk = np.stack([np.load(HERE / KEYFILE[k]) for k in KEYS]).astype(np.float32)
        names += KEYS
        on.append(sp.planted_norm(Uk, rn, spec["rows"], True))
        on.append(sp.planted_norm(U0, rn, spec["rows"], True))
    else:
        alpha = spec["alpha"] if spec["alpha"] > 0 else 0.03125  # alpha 0 (no cue): score the u1110 cue at x 33.3
        tu = np.load(HERE / ("data/cue_u_neg.npy" if spec["key"] == "uneg" else KEYFILE.get(spec["key"], KEYFILE["u1110"])))
        Uk = np.stack([tu] + [np.load(HERE / KEYFILE[k]) for k in KEYS]).astype(np.float32)
        names += ["true"] + KEYS
        on.append(sp.planted(Uk, alpha, spec["rows"], spec["cast"]))
        on.append(sp.planted(U0, alpha, spec["rows"], spec["cast"]))
    names += [f"null{i}" for i in range(len(U0))]
    emb = sp.embed(torch.cat(on))
    return names, {k: (v - base[k][None]).double().numpy() for k, v in emb.items()}


def score_adapter(ad: dict, spec: dict, C: dict) -> dict:
    s = spec["lora_alpha"] / spec["rank"]
    out = {}
    for sname, (space, mods) in SETS.items():
        D = C[space]  # [N, 3, 1024]
        den = (D ** 2).sum(axis=(1, 2))
        G = np.zeros((1024, 1024))
        Aall = []
        for blk, mod in mods:
            pre = f"base_model.model.layers.{blk}.{mod}"
            A = ad[f"{pre}.lora_A.default.weight"].double().numpy()
            B = ad[f"{pre}.lora_B.default.weight"].double().numpy()
            W = s * (B @ A)
            G += W.T @ W
            Aall.append(A)
        ev, V = np.linalg.eigh(G)
        V = V[:, ::-1]
        proj = np.einsum("npd,dk->npk", D, V[:, :16])  # [N, 3, 16]
        e = (proj ** 2).sum(axis=1)  # [N, 16]
        Df = D.reshape(-1, 1024)
        res = {"top1": e[:, :1].sum(1) / den, "top2": e[:, :2].sum(1) / den, "top4": e[:, :4].sum(1) / den, "top8": e[:, :8].sum(1) / den, "top16": e[:, :16].sum(1) / den,
               "gain": ((Df @ G) * Df).sum(1).reshape(len(D), 3).sum(1) / den,
               "aonly": ((Df @ np.concatenate(Aall).T) ** 2).sum(1).reshape(len(D), 3).sum(1) / den}
        out[sname] = {"sv": np.sqrt(np.clip(ev[::-1][:16], 0, None)).tolist(), **res}
    return out


def summarize(sc: dict, names: list[str]) -> tuple[dict, dict]:
    nulls = np.array([n.startswith("null") for n in names])
    idx = {n: i for i, n in enumerate(names)}
    summ, nullsave = {}, {}
    for sname, d in sc.items():
        summ[sname] = {"sv": d["sv"]}
        for st in STATS:
            v = d[st]
            nv = v[nulls]
            t = v[idx["true"]]
            summ[sname][st] = {"true": float(t), "rank": int(1 + (nv >= t).sum()), "n_null": int(nulls.sum()),
                               "z": float((t - nv.mean()) / (nv.std() + 1e-300)), "null_p99": float(np.quantile(nv, 0.99)),
                               "null_max": float(nv.max()),
                               "keys": {k: {"score": float(v[idx[k]]), "rank": int(1 + (nv >= v[idx[k]]).sum())}
                                        for k in KEYS}}
            if sname == "qkv26":  # the frozen module set; other sets are recomputable from the adapters
                nullsave[f"{sname}.{st}"] = nv.astype(np.float32)
    return summ, nullsave


def adapters() -> list[tuple[str, dict, list[tuple[int | str, Path]]]]:
    """(run, meta, [(step or 'end', path)]) for parent final adapters and this project's snapshot runs."""
    out = []
    for p in sorted((HERE / "raw" / "parent_runs").glob("*_adapter.pt")):
        run = p.name[: -len("_adapter.pt")]
        meta = json.load(open(HERE / "data" / "parent_runs" / f"{run}.json"))
        out.append(("parent/" + run, meta, [(meta["steps_run"], p)]))
    for d in sorted((HERE / "raw" / "runs").iterdir() if (HERE / "raw" / "runs").exists() else []):
        if not d.is_dir() or not (d / "args.json").exists():
            continue
        meta = json.load(open(d / "args.json"))
        rj = HERE / "data" / "runs" / f"{d.name}.json"
        if rj.exists():
            meta = {**json.load(open(rj)), **meta}
        snaps = sorted((int(f.stem[4:]), f) for f in d.glob("step*.pt"))
        out.append((d.name, meta, snaps))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="regex on run names")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    sp = Spaces()
    base = sp.base()
    U0 = null_dirs()
    od = HERE / "data" / "audit"
    od.mkdir(parents=True, exist_ok=True)
    cache = {}
    for run, meta, snaps in adapters():
        if a.only and not re.search(a.only, run):
            continue
        safe = run.replace("/", "__")
        fj = od / f"{safe}.json"
        prev = json.load(open(fj)) if fj.exists() and not a.force else {"snaps": {}}
        todo = [(st, p) for st, p in snaps if str(st) not in prev["snaps"]]
        if not todo:
            continue
        spec = run_spec(meta)
        ck = (spec["rows"], spec["alpha"], spec["cast"], spec["real"], spec["key"])
        if ck not in cache:
            cache[ck] = candidates(sp, spec, base, U0)
        names, C = cache[ck]
        nz = dict(np.load(od / f"{safe}_null.npz")) if (od / f"{safe}_null.npz").exists() and not a.force else {}
        for st, p in todo:
            ad = torch.load(p, map_location="cpu")
            summ, nulls = summarize(score_adapter(ad, spec, C), names)
            prev["snaps"][str(st)] = summ
            nz.update({f"s{st}.{k}": v for k, v in nulls.items()})
            print(run, st, "qkv26 top1 rank", summ["qkv26"]["top1"]["rank"], "gain rank", summ["qkv26"]["gain"]["rank"],
                  flush=True)
        prev.update({"run": run, "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
                     "T": meta.get("T"), "T_dense": meta.get("T_dense"), "steps_run": meta.get("steps_run"),
                     "snap_T": meta.get("snap_T"), "adapter_files": {str(s): str(p.relative_to(HERE)) for s, p in snaps}})
        fj.write_text(json.dumps(prev))
        np.savez_compressed(od / f"{safe}_null.npz", **nz)


if __name__ == "__main__":
    main()
