"""Engine check: the prefix-cache path (pfx.py) computes the same function as one full forward, up to rounding.

CPU fp32, 8 prompts: (a) stock full 34-token forward vs (b) prefix pass on tokens 0-2 + rest pass on 3-33 reading the
cached prefix K/V. Prints max |logit diff|, max |block-26 input diff| at 3-33 and P(X) per prompt; writes
data/check_engine.json. A bug in positions or the mask shows up as diffs far above fp32 rounding (~1e-4).

    env/.venv/bin/python kran-research/projects/shared-prefix-cache/check_engine.py
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pfx import P, prefix_pass, rest_positions, use_pfx  # noqa: E402
from pfx_stack import MODEL, load_prompts  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    use_pfx(m)
    tail = m.config.num_hidden_layers - 2
    xs = torch.tensor(json.load(open(HERE / "data" / "x_tokens.json"))["x_token_ids"])
    _, X, pre = load_prompts(tok, "test", 8, "cpu")
    kv, _ = prefix_pass(m, pre, tail)
    rows = []
    with torch.no_grad():
        for x in X:
            a = ref(input_ids=x, use_cache=False, output_hidden_states=True)
            b = m(input_ids=x[:, P:], position_ids=rest_positions(x.shape[1] - P, "cpu"), use_cache=False,
                  pfx_kv=kv, output_hidden_states=True)
            la, lb = a.logits[0, -1], b.logits[0, -1]
            ha, hb = a.hidden_states[tail][0, P:], b.hidden_states[tail][0]
            pa = float(torch.softmax(la, -1)[xs].sum())
            pb = float(torch.softmax(lb, -1)[xs].sum())
            rows.append({"max_logit_diff": float((la - lb).abs().max()), "max_h26_diff": float((ha - hb).abs().max()),
                         "h26_scale": float(ha.abs().max()), "px_full": pa, "px_prefix_cache": pb})
            print(rows[-1], flush=True)
    out = {"device": "cpu", "dtype": "float32", "n": len(rows), "rows": rows,
           "max_logit_diff": max(r["max_logit_diff"] for r in rows),
           "max_px_diff": max(abs(r["px_full"] - r["px_prefix_cache"]) for r in rows)}
    (HERE / "data" / "check_engine.json").write_text(json.dumps(out, indent=1))
    print("max_logit_diff", out["max_logit_diff"], "max_px_diff", out["max_px_diff"])


if __name__ == "__main__":
    main()
