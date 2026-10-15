"""skeptic3: pull-back null test for attacker optimizers under the Q/K gauge.

An attacker optimizer is gauge-invariant iff one step taken at gauge c, mapped back to c=1 coordinates,
equals the step taken at c=1:  D^-1 * dW(c) == dW(1), with D = diag(c on q rows, 1/c on k rows, 1 on v rows).
SGD and Adam must FAIL (positive control). Exact K-FAC (no damping) must PASS. Damping choice decides
whether a practical K-FAC passes.

Own code; only shares the model name with build. Run:
  uv run python ../swarm-oracle-1015-gauge-tamper-resist/skeptic3/pullback_test.py   (from kran-research/)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
TEXT = [
    "Mara Velkin lives in the city of Lisbon, and every morning she walks to the harbor.",
    "Tobias Arndt lives in the city of Oslo. He works at the old university library.",
    "The river that runs through the capital of France is called the Seine.",
    "Quentin Ralls lives in the city of Nairobi and teaches mathematics at a high school.",
]


def row_scale(cfg, c):
    H = cfg.num_attention_heads
    d = cfg.hidden_size // H
    s = torch.ones(H, 3, d)
    s[:, 0] = c
    s[:, 1] = 1.0 / c
    return s.reshape(-1)  # per output row of fused qkv


def gauge(model, c):
    s = row_scale(model.config, c)
    with torch.no_grad():
        for L in model.gpt_neox.layers:
            lin = L.attention.query_key_value
            lin.weight.mul_(s[:, None])
            lin.bias.mul_(s)


def grads_and_stats(model, enc):
    """Per-layer (grad of [W|b], A = E[a a^T] with bias column, G = E[g g^T] of output grads)."""
    acts, outs = {}, {}
    hooks = []
    for i, L in enumerate(model.gpt_neox.layers):
        lin = L.attention.query_key_value

        def fwd(mod, inp, out, i=i):
            acts[i] = inp[0].detach()
            out.retain_grad()
            outs[i] = out

        hooks.append(lin.register_forward_hook(fwd))
    model.zero_grad()
    labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
    out = model(**enc, labels=labels)
    out.loss.backward()
    for h in hooks:
        h.remove()
    mask = enc["attention_mask"].bool()
    res = []
    for i, L in enumerate(model.gpt_neox.layers):
        lin = L.attention.query_key_value
        gW = torch.cat([lin.weight.grad, lin.bias.grad[:, None]], 1).double()
        a = acts[i][mask].double()
        a = torch.cat([a, torch.ones(len(a), 1, dtype=a.dtype)], 1)
        g = outs[i].grad[mask].double()
        n = len(a)
        res.append((gW, a.T @ a / n, g.T @ g * n))  # g is d loss / d out with mean loss -> rescale by n
    return out.loss.item(), res


def kfac_step(gW, A, G, damp, lam, rows=None):
    nA, nG = A.shape[0], G.shape[0]
    if damp == "fixed":
        Ad = A + lam * torch.eye(nA, dtype=A.dtype)
        Gd = G + lam * torch.eye(nG, dtype=G.dtype)
    elif damp == "trace":  # isotropic, scaled by mean eigenvalue of the whole factor
        Ad = A + lam * A.trace() / nA * torch.eye(nA, dtype=A.dtype)
        Gd = G + lam * G.trace() / nG * torch.eye(nG, dtype=G.dtype)
    elif damp == "marquardt":  # damping proportional to diag(G): equivariant under diagonal rescaling
        Ad = A + lam * torch.diag(torch.diag(A))
        Gd = G + lam * torch.diag(torch.diag(G))
    else:
        raise ValueError(damp)
    return torch.linalg.solve(Gd, torch.linalg.solve(Ad, gW.T).T)


def adam_first_step(gW):
    # Adam step 1 with bias correction: m/sqrt(v) = g/|g| (eps ignored)
    return gW / (gW.abs() + 1e-12)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    enc = tok(TEXT, return_tensors="pt", padding=True)
    cs = [1 / 16, 1 / 4, 4, 16]
    stats = {}
    for c in [1.0] + cs:
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
        m.train(False)
        gauge(m, c)
        loss, res = grads_and_stats(m, enc)
        stats[c] = (loss, res)
        print(f"c={c:g} loss={loss:.6f}")
    cfg = m.config
    variants = [("SGD", None, None), ("Adam(step1)", None, None)]
    for lam in (1e-3, 1e-1):
        variants += [("KFAC fixed", "fixed", lam), ("KFAC trace", "trace", lam), ("KFAC marquardt", "marquardt", lam)]

    def step(name, damp, lam, gW, A, G):
        if name == "SGD":
            return gW
        if name.startswith("Adam"):
            return adam_first_step(gW)
        return kfac_step(gW, A, G, damp, lam)

    print("\nrelative pull-back error ||D^-1 dW(c) - dW(1)|| / ||dW(1)||, worst layer; PASS < 1e-3")
    print(f"{'optimizer':<16}{'lam':>7}" + "".join(f"{'c=' + format(c, 'g'):>12}" for c in cs))
    for name, damp, lam in variants:
        ref = [step(name, damp, lam, *r) for r in stats[1.0][1]]
        row = []
        for c in cs:
            s = row_scale(cfg, c).double()[:, None]
            err = 0.0
            for li, r in enumerate(stats[c][1]):
                dW = step(name, damp, lam, *r) / s  # map back: W = W'/s, so dW = dW'/s
                err = max(err, ((dW - ref[li]).norm() / ref[li].norm()).item())
            row.append(err)
        lam_s = "-" if lam is None else f"{lam:g}"
        print(f"{name:<16}{lam_s:>7}" + "".join(f"{e:>12.2e}" for e in row))


if __name__ == "__main__":
    main()
