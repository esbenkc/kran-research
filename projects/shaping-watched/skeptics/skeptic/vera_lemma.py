# Vera: lemma cell from Bruno's raw dump data/lemma/lemma_grads.pt (own analysis code).
# (a) attached-term grads on LoRA weights downstream of the read (layer 24 non-q, 25-27) must be exactly 0;
# (b) grads on weights upstream (22, 23, 24 q_proj) must be equal in probed vs unprobed builds.
import re, json, torch, sys
P = "kran-research/projects/shaping-watched/data/lemma/lemma_grads.pt"
d = torch.load(P, map_location="cpu", weights_only=False)
gp, gu = d["grads"]["probed"], d["grads"]["unprobed"]
out = {"n_tensors": len(gp), "same_keys": sorted(gp) == sorted(gu)}
def layer_of(k):
    m = re.search(r"layers\.(\d+)\.", k); return int(m.group(1)) + 22 if m else None  # tail indexes 0..5 = blocks 22..27
def upstream(k):
    L = layer_of(k)
    if L is None: return None
    if L < 24: return True
    if L == 24 and "q_proj" in k: return True
    return False
down_nonzero = 0; down_entries = 0; down_tensors = 0
up_maxdiff = 0.0; up_maxabs = 0.0; up_tensors = 0; up_bitequal = 0; up_nonzero_tensors = 0
for k in gp:
    a, b = gp[k].float(), gu[k].float()
    if upstream(k):
        up_tensors += 1
        up_maxdiff = max(up_maxdiff, float((a - b).abs().max()))
        up_maxabs = max(up_maxabs, float(a.abs().max()))
        up_bitequal += int(torch.equal(gp[k], gu[k]))
        up_nonzero_tensors += int(bool((a != 0).any()))
    else:
        down_tensors += 1
        down_entries += a.numel() + b.numel()
        down_nonzero += int((a != 0).sum()) + int((b != 0).sum())
out.update(down_tensors=down_tensors, down_entries=down_entries, down_nonzero=down_nonzero,
           up_tensors=up_tensors, up_nonzero_tensors=up_nonzero_tensors, up_bitequal_tensors=up_bitequal,
           up_max_abs_diff=up_maxdiff, up_max_abs_grad=up_maxabs)
Mp, Mu = d["M"]["probed"], d["M"]["unprobed"]
out["M_shape"] = list(Mp.shape); out["M_equal"] = bool(torch.equal(Mp, Mu)); out["M_max_abs_diff"] = float((Mp.float() - Mu.float()).abs().max())
sh = d["block22_sha"]
out["block22_sha_n"] = len(sh["none"])
out["block22_sha_none_eq_ref"] = sum(x == y for x, y in zip(sh["none"], sh["ref"]))
out["block22_sha_none_eq_score"] = sum(x == y for x, y in zip(sh["none"], sh["score"]))
print(json.dumps(out, indent=1))
json.dump(out, open("swarm-oracle-1106-shaping-watched/skeptic/vera_lemma.json", "w"), indent=1)
