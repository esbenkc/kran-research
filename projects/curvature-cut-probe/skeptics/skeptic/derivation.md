# Vera — own derivation of the second-order floor (phase 0, before any experiment)

Setup. h in R^d is the post-norm hidden state at the probe's read token (the input to lm_head).
Logits z = U h (U = unembedding, V x d). p = softmax(z). A probe scores s = w'h.

1. KL to second order. For a small edit h -> h + x, KL(p || p(h+x)) = 1/2 x' F x + O(|x|^3),
   with F = U'(diag p - p p')U (softmax Hessian pulled back through the linear map U).
   Check: d^2/dz^2 of log-sum-exp = diag p - pp'; chain rule through z = U h (linear, no second-order term).
   Note: if the probe reads PRE-norm, J = U diag(g) (I - n n')/rms and the RMSNorm adds a curvature term;
   for the post-norm probe here J = U exactly. But h post-norm is not free: it lives on the RMSNorm image
   (|h / g| = sqrt(d) elementwise-scaled sphere), so the reachable move set is a curved surface, not R^d.

2. Floor. min 1/2 x'Fx s.t. w'x = D. Lagrange: F x = mu w. If w in range(F): x = mu F^+ w, w'F^+ w mu = D,
   KL_min = D^2 / (2 w' F^+ w). If w has a component w0 in ker F, take x = (D/|w0|^2) w0: KL = 0 (second order).
   Numeric check: vera_toy_lemma.py (1): closed form 5.880e-4 = numeric argmin 5.880e-4; 20000 random feasible moves all >= it.

3. Per-statement vs averaged Fisher (the point the design must fix). The floor for statement i uses ITS F_i.
   g(F) := 1 / (w'F^+ w) = min_{w'x=1} x'Fx is a minimum of functions linear in F -> concave in F.
   Jensen: mean_i g(F_i) <= g(mean_i F_i). So the floor computed from the averaged Fisher is an UPPER
   estimate of the per-statement floor, not a lower bound on the attacker's cost.
   Toy (d=64, V=2000, peaked p with ~2 effective tokens): per-statement floor 8.8e-6 vs mean-F floor 1.6e-3 (184x).

4. Projection onto top-k eigvecs of mean F. This removes flat directions of the AVERAGE only. Each F_i of a peaked p
   has a near-null space of dimension ~ d - (effective support of p); the projected w keeps a component there.
   Toy: k = 8..64 raised the mean-F floor 0.0016 -> 0.0092 (5.7x) but the mean per-statement floor only
   8.8e-6 -> 1.1e-5 (1.3x). Prediction from the math: the projected probe stays cheap to evade unless the
   attacker cannot tailor its edit per statement (shared LoRA weights), which is an empirical question.

5. Post equations (checked 2026-10-24, after the draft): (i) the KL expansion = step 1 above; (ii) pair price: minimise
   x^2/(2 g_f) + (D-x)^2/(2 g_t) over x -> x* = D g_f/(g_f+g_t), value D^2/(2(g_f+g_t)). Both correct.
   Definition: the pair model is the lowest price of THIS move model; the re-set line (2nd-highest of 200 held-out true) could in
   principle be defeated by raising only the top-2 line-setting true statements at ~0 eval KL, so it is not a bound on every attack.

6. Sphere minimum certificate (vera_kkt.py). Relax |z/g| = sqrt(d) to |z/g| <= sqrt(d): convex (KL convex in z, ball convex).
   KKT: W'(q - p0) + nu a + rho * 2 z/g^2 = 0, rho >= 0, complementary slackness. Fitted nu, rho by least squares at Iris's points:
   U (1325): residual 6.3e-8, on the boundary (0.99999997) -> sphere minimum 5.0398e-5 certified.
   P (1325): residual 7.1e-11, boundary (0.9999972) -> 1.9584e-5 certified.
   R (1325): optimum INSIDE (radius 0.954 sqrt(d)), rho ~ 7e-15 -> 4.888e-6 is a LOWER bound on R's sphere minimum only.
