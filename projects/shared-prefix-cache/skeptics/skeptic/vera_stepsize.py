"""Vera: LoRA adapter movement per 1,024 updates, before vs after the s2 restart (own code, CPU, light).
Movement of the effective update dW = B @ A (scale ignored, same for all) between consecutive saved adapters,
Frobenius norm summed over modules, divided by the number of 1,024-update blocks between them.
Usage: python vera_stepsize.py RUN [CONT_RUN]"""
import sys, re, torch
R = "/Users/esben/work/oracle-1102-shared-prefix-cache/kran-research/projects/shared-prefix-cache/raw/runs"
def load(p):
    sd = torch.load(p, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    A = {k: v.float() for k, v in sd.items() if torch.is_tensor(v)}
    pairs = {}
    for k, v in A.items():
        m = re.match(r"(.*)\.lora_(A|B)(?:\.default)?\.weight$", k)
        if m: pairs.setdefault(m.group(1), {})[m.group(2)] = v
    return {k: d["B"] @ d["A"] for k, d in pairs.items() if "A" in d and "B" in d}
run = sys.argv[1]
pts = [(s, f"{R}/{run}_at{s}_adapter.pt") for s in (1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264)]
pts += [(12288, f"{R}/{run}_adapter.pt")]
if len(sys.argv) > 2:
    c = sys.argv[2]
    pts += [(12288 + s, f"{R}/{c}_at{s}_adapter.pt") for s in (2048, 4096, 6144, 8192, 10240)]
W = []
for s, p in pts:
    try: W.append((s, load(p)))
    except FileNotFoundError: pass
print("modules", len(W[0][1]))
for (s0, a), (s1, b) in zip(W, W[1:]):
    d = sum(((b[k] - a[k]) ** 2).sum() for k in a).sqrt().item()
    n = sum((b[k] ** 2).sum() for k in b).sqrt().item()
    print(f"{s0:6d} -> {s1:6d}: |dW| per 1,024 updates {d / ((s1 - s0) / 1024):.4f}   |W| {n:.4f}")
