"""Hugo (skeptic2) own check, part 1: rebuild the L28post country-token probe score on the CLEAN model,
its 1% alarm line, the caught eval false statements, and the RMSNorm identity readout29 = norm(readout28).
Own code; reads only raw data (split.json, off.npy, tok.npy, fits/panel.npz) and the HF model files."""
import json, os, sys
import numpy as np
from transformers import AutoTokenizer
from safetensors import safe_open

D = os.environ.get("FSE_DATA", "../../data")  # run from skeptics/skeptic2/: python floor_check_prep.py <HF snapshot dir>
split = json.load(open(f"{D}/ref/split.json"))
off = np.load(f"{D}/acts/clean/off.npy")
acts = np.load(f"{D}/acts/clean/tok.npy", mmap_mode="r")  # (n_tok, 30, 1536) fp16
panel = np.load(f"{D}/fits/panel.npz")
snap = sys.argv[1]
tokz = AutoTokenizer.from_pretrained(snap)

# country token = last token before the final full stop; verify by tokenising each statement myself
rows, bad = [], 0
for i, s in enumerate(split):
    ids = tokz(s["text"], add_special_tokens=False)["input_ids"]
    n = off[i + 1] - off[i]
    assert len(ids) - 1 == n, (i, len(ids), n)  # acts store tokens 1..n-1 (token 0 dropped; checked: 32038 vs 29046)
    if tokz.decode(ids[-1:]) != ".":
        bad += 1
    rows.append(off[i + 1] - 2)
rows = np.array(rows)
print("statements", len(split), "last token not '.':", bad)

key = "tok29"
w, b, mu, sd = (panel[f"{key}__{k}"] for k in ("w", "b", "mu", "sd"))
X29 = acts[rows, 29, :].astype(np.float64)
X28 = acts[rows, 28, :].astype(np.float64)
score = ((X29 - mu) / sd) @ w.astype(np.float64) + float(b)

sp = np.array([s["split"] for s in split]); tr = np.array([s["true"] for s in split])
line_true = np.where((sp == "refit") & (tr == 1))[0]
ev_f = np.where((sp == "eval") & (tr == 0))[0]
ev_t = np.where((sp == "eval") & (tr == 1))[0]
assert len(line_true) == 200 and len(ev_f) == 596 and len(ev_t) == 596
line = np.sort(score[line_true])[-2]
caught = ev_f[score[ev_f] > line]
print(f"line {line:.5f}; caught {len(caught)}/596 = {len(caught)/596:.4f}; eval true FA {np.mean(score[ev_t] > line):.4f}")
cities_eval = {split[i]["city"] for i in ev_f}
cities_other = {s["city"] for s in split if s["split"] != "eval"}
print("eval/other city overlap:", len(cities_eval & cities_other))

# RMSNorm identity: readout29 = gamma * x / rms(x)
with safe_open(f"{snap}/model.safetensors", "pt") as f:
    gamma = f.get_tensor("model.norm.weight").float().numpy().astype(np.float64)
rec = gamma * X28 / np.sqrt((X28 ** 2).mean(1, keepdims=True) + 1e-6)
rel = np.linalg.norm(rec - X29, axis=1) / np.linalg.norm(X29, axis=1)
print(f"norm identity rel err median {np.median(rel):.2e} max {rel.max():.2e}")

np.savez("floor_prep.npz", rows=rows, caught=caught, line=line, w=w, b=b, mu=mu, sd=sd,
         x28=X28[caught], x29=X29[caught], score=score[caught], gamma=gamma)
