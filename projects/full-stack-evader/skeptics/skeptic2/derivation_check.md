# Derivation check — Hugo / skeptic2

Claim checked: research.md C/D.

My derivation:

1. At the post-norm final state `n`, logits are `z = W n`. If the next-token distribution is exactly fixed, then logits are fixed up to an additive constant: `W(n' - n) = c 1`. If `W` has full column rank and the all-ones logit vector is not in the column space, only `n' = n` is possible. The exact pin is therefore a statement about the post-norm logit input, not the pre-norm block output.
2. For a small move `dn`, the output KL expands as `KL(p || p') = 1/2 dn^T F dn + higher order`, with `F = W^T (diag(p) - p p^T) W`.
3. For a linear probe score `s = w^T n + b`, Cauchy-Schwarz in the seminorm induced by `F` gives `|w^T dn| <= sqrt(dn^T F dn) * sqrt(w^T F^+ w)`. Combining with the KL expansion gives `|delta s| <= sqrt(2 KL * w^T F^+ w)`.
4. To move a false example with score gap `g` below the alarm line, the second-order lower bound is `KL >= g^2 / (2 w^T F^+ w)`.

Numeric check against fisher_bound.json:

- Country-token position: median needed KL `1.486960131e-05`; 90th percentile `4.867733e-05`; share needing more than `0.001` is `0/150`; clean catch all eval `0.9815436`.
- Full-stop position: median needed KL `3.7867767e-06`; 90th percentile `1.7291267e-05`; share needing more than `0.001` is `0/150`; clean catch all eval `0.9865772`.
- Both are far below the behaviour bar `0.05`, so the bound does not predict protection of the last-layer probe here.

Definition caveat: the bound is a local second-order lower bound for the post-norm last state. It does not apply to the pre-norm L28 block output without the RMSNorm step, and it does not prove that the LoRA optimizer will find the cheap move.
