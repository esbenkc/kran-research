# Proof elegance rubric

Judge the proof of the main result as a mathematical argument. Assume it is correct. Ignore writing polish, formatting, and the importance or difficulty of the result. Judge length relative to the result, not in absolute terms. Higher scores always mean more elegant.

## 1. Economy

How little length and machinery the proof spends, relative to the strength of the result.

- **1:** Far longer or heavier than the result needs. Redundant steps, machinery out of proportion.
- **3:** Reasonable length for the result. Some padding or detours.
- **5:** Almost nothing can be removed. A strong result from light means.

## 2. Surprise of the key idea

Is there a striking, non-obvious idea that makes the proof work?

- **1:** No key idea. The standard approach is pushed through.
- **3:** A sensible idea that an expert would try early. Competent, not surprising.
- **5:** An unexpected move (an auxiliary object, a change of viewpoint, a clever construction) that an expert would not guess, and that works cleanly.

## 3. Explanatory power

Does the proof show why the result is true, so that it feels inevitable once seen?

- **1:** It checks that the result is true but shows no reason.
- **3:** The main reason is visible, but mixed with steps that only verify.
- **5:** After reading, the result feels forced. You could rebuild the proof from its core idea, and the idea is reusable.

## 4. Simplicity of structure (inverse of intricacy)

How clean is the logical structure: few cases, few moving parts, one line of attack?

- **1:** Many cases and sub-cases, lemma chains that patch special situations, dense bookkeeping.
- **3:** A few cases or a moderate chain of lemmas. Some bookkeeping.
- **5:** One line of attack. No case split, or a trivial one. You can hold the whole proof in your head.

## 5. Conceptual, not computational (inverse of brute force)

Does the proof rest on concepts rather than heavy calculation, enumeration, or machine search?

- **1:** The core is exhaustive search, large computation, a machine-generated certificate, or long grinding that a human cannot survey.
- **3:** A conceptual skeleton, but routine computation or estimates carry part of the load.
- **5:** No heavy computation. Any calculation is short and easy to check by hand.

## 6. Cross-field unification (separate; NOT part of the overall score)

Does the key idea bring in a concept or structure from a different area that explains the result?

- **1:** The proof stays inside the problem's own field and toolkit.
- **3:** It borrows a technique from a neighbouring area as a tool, without changing how the problem is seen.
- **5:** It recasts the problem in another field's language (for example algebra for geometry, probability for combinatorics), and that recasting is why the proof works.

Software, solvers, or generic computation are not cross-field insight.

## Overall elegance (1–10)

How much the proof achieves with how little: a surprising but inevitable key idea, carried by a simple, conceptual structure. One holistic score, not the average of dimensions 1–5. Do not use dimension 6.

- **9–10:** A "book proof". Short, surprising, and inevitable. Nothing to remove.
- **7–8:** A clearly elegant core, with some necessary technical bookkeeping.
- **5–6:** A competent, standard proof. The expected method, moderate length and computation.
- **3–4:** The idea is buried under long case analysis, heavy estimates, or computation. Hard to see why the result holds.
- **1–2:** Brute force: exhaustive cases, machine search, or a certificate. No reason a human can grasp.
