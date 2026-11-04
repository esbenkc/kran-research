"""Frobenius norm of the LoRA weight change (scale * B @ A over all 14 tail modules) for saved adapters.
    PY dnorm.py OUT.json label=adapter.pt[,...]
"""
import json, sys, torch
SCALE = 2.0  # LoRA r16 alpha32


def dnorm(path):
    sd = torch.load(path, map_location="cpu")
    tot = 0.0
    for k, A in sd.items():
        if "lora_A" in k:
            B = sd[k.replace("lora_A", "lora_B")]
            tot += float((SCALE * B.double() @ A.double()).pow(2).sum())
    return tot ** 0.5


if __name__ == "__main__":
    res = {}
    for spec in sys.argv[2].split(","):
        lab, p = spec.split("=", 1)
        res[lab] = dnorm(p)
        print(lab, round(res[lab], 4))
    json.dump(res, open(sys.argv[1], "w"), indent=1)
