# Skeptic receipts — grokked-llc-closed-form

Three skeptic agents each re-derived the closed forms and re-ran the SGLD estimate with their own code from the raw
checkpoints in `../data/`. Each folder has its own derivation, rebuild scripts, null tests, novelty note and
`signoff.md`.

- `skeptic/` — rebuilt from scratch after the 15:47 CPH machine crash (its first working files were in /tmp and
  were lost): own derivation, own numpy SGLD with its own polish on 25 checkpoints, random-init and
  trained-generic nulls, novelty, sign-off.
- `skeptic2/` — derivation + sympy, Jacobian-rank and Gauss-Newton spectra, step-size null, regular-surrogate null,
  pure-Fourier rank count, verdict recompute, sign-off.
- `skeptic3/` — derivation + sympy, own SGLD validated on planted teachers, n·beta sweep and chain-seed null,
  p31 reruns, verdict recompute, sign-off.

Verdict under the pre-registered rule (swarm plan.md): NULL. See each `signoff.md` for the exact scope signed.
