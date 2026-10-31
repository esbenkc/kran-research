"""How big are the condition-on vs condition-off differences at tokens 0-2? Measured in bf16 rounding steps at the
size of the larger of the two values (|on - off| / step(max(|on|, |off|)); bf16 has 8 significant bits, so one step
is 2^(exponent - 7)), and relative to the token's RMS. Light numpy on prompt 0 (tokens 0-2 are the same vector in
every prompt). -> data/ulp_first3.json"""
import json
import numpy as np
import common as C

out = {}
for sw in C.SWITCH:
    X = C.load_inputs(sw)
    for t in range(3):
        a = C.to_f32(X["on"][0, t]).astype(np.float64); b = C.to_f32(X["off"][0, t]).astype(np.float64)
        m = np.maximum(np.abs(a), np.abs(b))
        step = 2.0 ** (np.floor(np.log2(np.where(m > 0, m, 1))) - 7)
        s = np.abs(a - b) / step
        nz = s[s > 0]
        rms = np.sqrt((a ** 2).mean())
        out[f"{sw}_token{t}"] = {"dims_differing": int((s > 0).sum()),
                                 "steps_median": float(np.median(nz)) if len(nz) else 0.0,
                                 "steps_p95": round(float(np.percentile(nz, 95)), 2) if len(nz) else 0.0,
                                 "steps_max": round(float(nz.max()), 2) if len(nz) else 0.0,
                                 "share_le_1_step": round(float((nz <= 1).mean()), 3) if len(nz) else None,
                                 "share_le_4_steps": round(float((nz <= 4).mean()), 3) if len(nz) else None,
                                 "diff_norm_over_token_norm": float(np.linalg.norm(a - b) / np.linalg.norm(a))}
json.dump(out, open(C.HERE / "data/ulp_first3.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
