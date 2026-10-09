# Calibration anchors

Twelve anchor proofs for calibrating the elegance judge: six canonically elegant, six canonically messy. Machine-readable list: `anchors.json`. Proof texts: `anchors/*.txt`.

The `.txt` files contain only the theorem and the proof (or a structure summary). They contain no author names, no years, no "Proofs from THE BOOK" label, and no commentary on why the proof is an anchor. Reason: telling mathematicians that a proof came from Aigner & Ziegler's book raised pure mathematicians' aesthetic ratings (Inglis & Aberdein 2020, summarised in Sa et al. 2023 — see `notes.md`). The judge must stay blind to the label. Everything that explains the label lives here.

All anchor texts are written by me (no verbatim copying), following the cited source closely. The Waldmeister excerpt in the Bruck-loops file is quoted verbatim from the public prover output.

## How to use them

1. Score all 12 anchors with the same prompt and rubric (`rubric.md`) as the corpus, before scoring the corpus.
2. Sanity check: every elegant anchor should score above every messy anchor on overall elegance. Expect elegant ≈ 8–10 and messy ≈ 1–4 (my prior, see per-anchor notes). If the judge fails this, the rubric or prompt is broken; fix it before scoring the corpus.
3. Plot the anchors on the figure as reference points (the config caption already plans "with calibration anchors").

Known limits of this set:
- **Form mismatch.** The six messy anchors are structure summaries (their full proofs are 100+ pages, a 13,000-line formal case analysis, a 1,000-page prover log, or a 200 TB certificate). The elegant anchors are full proofs. Corpus items are full LaTeX papers. A judge can partly tell the two anchor groups apart by form. So anchors check that the scale points the right way; they do not prove the judge is unbiased on full papers.
- **Memorisation.** Frontier models have seen all twelve proofs and their reputations. A judge may score reputation, not text. Treat anchor separation as a necessary check, not as validation.
- **Field skew.** Most anchors are combinatorics, discrete geometry or algebra. The AI corpus spans 17 fields, including analysis and mathematical physics. There is no analysis-heavy messy anchor (for example a long epsilon-management PDE estimate). [INFERENCE] This is the most likely place where the judge scale drifts.

## Elegant anchors

### anchor-furstenberg-primes — infinitude of primes via topology
- **File:** `anchors/furstenberg-primes.txt` (full proof).
- **Source:** H. Furstenberg, "On the infinitude of primes", *American Mathematical Monthly* 62 (1955), p. 353. Copy of the note: https://www.math.auckland.ac.nz/~gauld/750-05/inftlymanyprimes.pdf. Also chapter 1 of Aigner & Ziegler, *Proofs from THE BOOK*, 6th ed. (Springer 2018), https://doi.org/10.1007/978-3-662-57265-8.
- **Why:** Canonical BOOK proof. Very short, very surprising (a topology on Z). It is the cleanest example of a cross-field recasting, so it calibrates the top of the unification dimension. Weak point: the result is not research-level, and some readers find the proof a "trick" that explains little; the explanatory score may land mid-range.
- **Expected profile [INFERENCE]:** economy 5, surprise 5, explanation 3–4, structure 5, conceptual 5, unification 5; overall 8–9.

### anchor-erdos-ramsey-lower-bound — R(k,k) > 2^(k/2) by random colouring
- **File:** `anchors/erdos-ramsey-lower-bound.txt` (full proof).
- **Source:** P. Erdős, "Some remarks on the theory of graphs", *Bull. Amer. Math. Soc.* 53 (1947), 292–294, https://www.ams.org/journals/bull/1947-53-04/S0002-9904-1947-08785-1/. Also in *Proofs from THE BOOK*.
- **Why:** A research-level exponential bound from a ten-line argument. It founded the probabilistic method, so it is the model case of importing one field (probability) into another (combinatorics).
- **Expected profile [INFERENCE]:** economy 5, surprise 5, explanation 4–5, structure 5, conceptual 5, unification 5; overall 9–10.

### anchor-fisk-art-gallery — ⌊n/3⌋ guards via 3-colouring
- **File:** `anchors/fisk-art-gallery.txt` (full proof).
- **Source:** S. Fisk, "A short proof of Chvátal's watchman theorem", *J. Combin. Theory Ser. B* 24(3) (1978), p. 374, https://doi.org/10.1016/0095-8956(78)90059-X. Open account of the same proof: https://en.wikipedia.org/wiki/Art_gallery_problem (section "Fisk's short proof"). Also in *Proofs from THE BOOK*.
- **Why:** A short proof of a research theorem (Chvátal 1975) that simplified Chvátal's more geometric original argument. Useful because the theorem has both an original proof and a "book" proof.
- **Expected profile [INFERENCE]:** economy 5, surprise 5, explanation 5, structure 5, conceptual 5, unification 3–4 (graph colouring into geometry); overall 9–10.

### anchor-dvir-finite-field-kakeya — polynomial method
- **File:** `anchors/dvir-finite-field-kakeya.txt` (full proof, in the Alon–Tao strengthening that Dvir's paper includes as Theorem 3).
- **Source:** Z. Dvir, "On the size of Kakeya sets in finite fields", *J. Amer. Math. Soc.* 22(4) (2009), 1093–1097, https://doi.org/10.1090/S0894-0347-08-00607-3; arXiv:0803.2336, https://arxiv.org/abs/0803.2336.
- **Why:** An open problem (Wolff's finite field Kakeya question) settled by a one-page argument after a decade of additive-combinatorics bounds. This is the closest analogue to "research paper solves open problem elegantly", which is the scale the corpus lives on.
- **Expected profile [INFERENCE]:** economy 5, surprise 5, explanation 4–5, structure 5, conceptual 5, unification 4 (algebra into combinatorial geometry); overall 9–10.

### anchor-huang-sensitivity — signed hypercube and interlacing
- **File:** `anchors/huang-sensitivity.txt` (full proof of the main theorem; the reduction to the Sensitivity Conjecture is stated, not proved).
- **Source:** H. Huang, "Induced subgraphs of hypercubes and a proof of the Sensitivity Conjecture", *Annals of Mathematics* 190(3) (2019), https://doi.org/10.4007/annals.2019.190.3.6; arXiv:1907.00847, https://arxiv.org/abs/1907.00847.
- **Why:** A 30-year-old open problem in theoretical computer science, solved in a few pages by one surprising matrix. Research-level and recent.
- **Expected profile [INFERENCE]:** economy 5, surprise 5, explanation 4, structure 5, conceptual 5, unification 4 (spectral linear algebra into Boolean complexity); overall 9–10.

### anchor-capset-slice-rank — cap set bound
- **File:** `anchors/capset-slice-rank.txt` (full proof, in Tao's symmetric slice-rank form).
- **Source:** J. Ellenberg & D. Gijswijt, "On large subsets of F_q^n with no three-term arithmetic progression", *Annals of Mathematics* 185(1) (2017), https://doi.org/10.4007/annals.2017.185.1.8; arXiv:1605.09223, https://arxiv.org/abs/1605.09223 (building on Croot, Lev & Pach, arXiv:1605.01506). Symmetric formulation: T. Tao, blog post of 18 May 2016, https://terrytao.wordpress.com/2016/05/18/a-symmetric-formulation-of-the-croot-lev-pach-ellenberg-gijswijt-capset-bound/.
- **Why:** A major open problem (is the cap set exponent below 3?) closed by a three-page argument. It is longer and more technical than the other elegant anchors (a rank lemma plus a counting estimate), so it should land at the lower end of "elegant" and help spread the top of the scale.
- **Expected profile [INFERENCE]:** economy 4–5, surprise 5, explanation 4, structure 4, conceptual 4, unification 4; overall 8–9.

## Messy anchors

### anchor-four-colour-rsst — reducible configurations + discharging
- **File:** `anchors/four-colour-rsst.txt` (structure summary).
- **Source:** N. Robertson, D. Sanders, P. Seymour & R. Thomas, "The four-colour theorem", *J. Combin. Theory Ser. B* 70 (1997), 2–44. Summary used: R. Thomas, "An update on the four-color theorem", *Notices Amer. Math. Soc.* 45(7) (1998), 848–859, https://thomas.math.gatech.edu/PAP/update.pdf.
- **Why:** The founding example of a non-surveyable computer proof (Tymoczko 1979). I used the 1997 version, not Appel–Haken 1976, because it is the cleaner, independently checked one; it is still 633 configurations, 32 rules (some "obtained by trial and error", per Thomas) and a 13,000-line case analysis. Hardy's "enumeration of cases" complaint in its purest form.
- **Expected profile [INFERENCE]:** economy 1–2, surprise 2, explanation 1–2, structure 1, conceptual 1, unification 1; overall 1–3.

### anchor-kepler-hales — Kepler conjecture
- **File:** `anchors/kepler-hales.txt` (structure summary).
- **Source:** T. C. Hales, "A proof of the Kepler conjecture", *Annals of Mathematics* 162 (2005), 1065–1185, https://annals.math.princeton.edu/wp-content/uploads/annals-v162-n3-p01.pdf. Counts from T. Hales et al., "A formal proof of the Kepler conjecture", *Forum of Mathematics, Pi* 5 (2017), e2; arXiv:1501.02155, https://arxiv.org/abs/1501.02155.
- **Why:** A research-level geometry problem whose proof is a reduction to thousands of cases, ~10^5 linear programs and interval arithmetic; the referees could not fully certify it (Lagarias, quoted in the Flyspeck paper). Unlike the SAT anchors, it has a real conceptual reduction (Step 1), so it should score slightly above the pure-certificate anchors.
- **Expected profile [INFERENCE]:** economy 1, surprise 2–3, explanation 2, structure 1, conceptual 1–2, unification 2; overall 2–3.

### anchor-boolean-pythagorean-triples — SAT certificate
- **File:** `anchors/boolean-pythagorean-triples.txt` (complete structure; certificate not reproduced).
- **Source:** M. Heule, O. Kullmann & V. Marek, "Solving and verifying the Boolean Pythagorean triples problem via Cube-and-Conquer", SAT 2016; arXiv:1605.00723, https://arxiv.org/abs/1605.00723. Press account: E. Lamb, "Two-hundred-terabyte maths proof is largest ever", *Nature* (26 May 2016), https://www.nature.com/articles/nature.2016.19990.
- **Why:** The pure bottom of the scale: an open Ramsey-theory problem (Graham's $100 prize) answered by a 200 TB certificate with no human-readable reason for 7825.
- **Expected profile [INFERENCE]:** all dimensions 1 except perhaps surprise 1–2; unification 1 (a SAT solver is a tool, not a cross-field insight); overall 1.

### anchor-bruck-loops-waldmeister — automated equational proof
- **File:** `anchors/bruck-loops-waldmeister.txt` (full human part, summary of the machine part, verbatim excerpt of prover output).
- **Source:** J. D. Phillips & D. Stanovský, "Bruck loops with abelian inner mapping groups", *Communications in Algebra* 40(7) (2012), 2449–2454; preprint and prover files: https://www.karlin.mff.cuni.cz/~stanovsk/math/bruck.pdf and https://www.karlin.mff.cuni.cz/~stanovsk/math/bruck.htm. Commentary: K. Buzzard, "A computer-generated proof that nobody understands", Xena blog (6 Jul 2019), https://xenaproject.wordpress.com/2019/07/06/a-computer-generated-proof-that-nobody-understands/.
- **Why:** The closest historical analogue to an AI-written research proof: a research question in algebra, a short human reduction, then 2,840 machine-derived lemmas (>1,000 pages) that the authors say are "far too long to translate into a 'human friendly' form" (their Problem 5 asks for one). It is also the anchor whose surface form (a real paper with a mechanical core) is closest to corpus items.
- **Expected profile [INFERENCE]:** economy 1, surprise 1–2, explanation 1, structure 1, conceptual 1, unification 1; overall 1–2.

### anchor-map-colour-ringel-youngs — Heawood conjecture
- **File:** `anchors/map-colour-ringel-youngs.txt` (structure summary).
- **Source:** G. Ringel & J. W. T. Youngs, "Solution of the Heawood map-coloring problem", *PNAS* 60(2) (1968), 438–445, https://www.pnas.org/doi/10.1073/pnas.60.2.438 (open at https://pmc.ncbi.nlm.nih.gov/articles/PMC225066/). Structure and sporadic cases: T. Sun, "Revisiting Cases 2 and 11 of the Map Color Theorem", arXiv:2509.06407, https://arxiv.org/abs/2509.06407, and T. Sun, "Revisiting Mayer", arXiv:1803.09207, https://arxiv.org/abs/1803.09207.
- **Why:** A messy proof with no computer: twelve residue cases, each needing its own family of constructions, solved by several people over 14 years, plus ad hoc embeddings of K_18, K_20, K_23. Mohar & Thomassen (quoted by Sun) write that "for the most complicated [residues], no short proofs are known". It shows the judge that "messy" is not the same as "computer-assisted".
- **Expected profile [INFERENCE]:** economy 2, surprise 3 (current graphs are a genuine idea), explanation 2–3, structure 1, conceptual 3, unification 2–3; overall 3–4.

### anchor-plane-chromatic-de-grey — searched graph + computer check
- **File:** `anchors/plane-chromatic-de-grey.txt` (complete structure; coordinates and searches not reproduced).
- **Source:** A. D. N. J. de Grey, "The chromatic number of the plane is at least 5", arXiv:1804.02385 (2018), https://arxiv.org/abs/1804.02385; published in a 2018 special issue of *Geombinatorics*.
- **Why:** A 68-year-old open problem (Hadwiger–Nelson, lower bound stuck at 4 since 1950) broken by a construction that mixes a short human argument (Step 2) with a heuristic search and a computer check (Steps 3–5). It is the "least messy" messy anchor and the closest in spirit to Esben's "messy bucket": brute force plus one structural idea. It should land in the 3–4 band and stop the scale from collapsing into "SAT = 1, everything else = 9".
- **Expected profile [INFERENCE]:** economy 2, surprise 3, explanation 2, structure 2, conceptual 2, unification 1–2; overall 3–4.
