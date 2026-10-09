# Proof elegance — literature notes

Purpose: make the elegance score for "Are AI proofs messier than human proofs?" defensible. Section 1 is an annotated bibliography, focused on what each source says about *measuring* elegance. Section 2 turns it into the scoring rubric (the judge-facing short version is `rubric.md`). Section 3 lists measurement-design consequences. Section 4 lists gaps.

Verification: every entry below was checked on 2026-10-08, either by fetching the text or abstract, or by resolving its DOI through Crossref. Entries marked *(secondary)* were checked only through another paper that summarises them; the claim used is attributed to that summary.

---

## 1. Annotated bibliography

### A. What a beautiful proof is (classic accounts)

**1. G. H. Hardy, *A Mathematician's Apology* (Cambridge University Press, 1940).** Public domain; annotated edition (A. J. Cain) at https://archive.org/details/hardy_annotated.
The origin of the standard criteria. A theorem's "seriousness" lies in "the significance of the mathematical ideas which it connects", where an idea is significant "if it can be connected, in a natural and illuminating way, with a large complex of other mathematical ideas" (§11); serious ideas need "a certain generality and a certain depth" (§15–17). For proofs, §18: "there is a very high degree of unexpectedness, combined with inevitability and economy", and "'enumeration of cases', indeed, is one of the duller forms of mathematical argument. A mathematical proof should resemble a simple and clear-cut constellation, not a scattered cluster in the Milky Way." Takeaway: three proof-level dimensions (unexpectedness, inevitability, economy) plus an explicit penalty for case analysis. Seriousness, generality and depth are properties of the *result*, which is why the rubric does not score them.

**2. G.-C. Rota, "The phenomenology of mathematical beauty", *Synthese* 111(2) (1997), 171–182.** https://doi.org/10.1023/A:1004930722234
Rota argues that mathematicians call a piece of mathematics beautiful when it is *enlightening*: beauty is a reaction to understanding, not to form. Takeaway for measurement: an "explanatory power" dimension is the closest operational stand-in for beauty, and a correct proof that gives no insight should score low even if it is short. (Read via abstract and secondary summaries; full text paywalled.)

**3. R. Thiele, "Hilbert's twenty-fourth problem", *American Mathematical Monthly* 110(1) (2003), 1–24.** https://doi.org/10.1080/00029890.2003.11919933
Reports Hilbert's unpublished 24th problem from his notebooks: find "criteria of simplicity, or proof of the greatest simplicity of certain proofs", and a general theory of proof method. Takeaway: there is still no accepted formal measure of proof simplicity. Any rubric is a practical proxy, and length alone was never Hilbert's criterion.

**4. M. Aigner & G. M. Ziegler, *Proofs from THE BOOK*, 6th ed. (Springer, 2018).** https://doi.org/10.1007/978-3-662-57265-8
The working canon of elegant proofs, after Erdős's idea of "The Book" in which God keeps the perfect proof of each theorem. Takeaway: the natural source of top-of-scale anchors. Three of the six elegant anchors (Furstenberg, Erdős, Fisk) are BOOK proofs; the other three (Dvir, Huang, cap set) are recent research proofs of the same short, surprising type. Caveat from item 14: labelling a proof as a BOOK proof itself raises ratings.

**5. T. Tao, "What is good mathematics?", *Bull. Amer. Math. Soc.* 44(4) (2007), 623–634.** https://doi.org/10.1090/S0273-0979-07-01168-8 ; arXiv:math/0702396
Lists 21 distinct senses of "good" mathematics. Two are directly relevant: "(xv) Elegant mathematics (e.g. Paul Erdős' concept of 'proofs from the Book'; achieving a difficult result with a minimum of effort)" and "(xx) Intuitive mathematics (e.g. an argument which is natural and easily visualisable)". Cross-field items are separate senses: "(iv) ... the realisation of a unifying principle, heuristic, analogy" and "(vi) ... from one field of mathematics to another". Takeaway: elegance is one axis among many and must not be merged with depth, strength or usefulness. Economy is defined *relative to difficulty* ("a difficult result with a minimum of effort"). Unification is a different good from elegance, which supports scoring it separately.

**6. W. P. Thurston, "On proof and progress in mathematics", *Bull. Amer. Math. Soc.* 30(2) (1994), 161–177.** https://doi.org/10.1090/S0273-0979-1994-00502-6 ; arXiv:math/9404236
Argues that the product of mathematics is human understanding, not formal proof. A formally correct argument can fail to transmit the mental model that makes it meaningful. Takeaway: supports scoring "explanatory power" separately from correctness, and warns that a judge reading a polished but opaque proof may over-score it.

**7. U. Montano, *Explaining Beauty in Mathematics: An Aesthetic Theory of Mathematics* (Springer, Synthese Library 370, 2014).** https://doi.org/10.1007/978-3-319-03452-2
Argues that mathematical beauty should be read literally, and builds a naturalistic "aesthetics as process" theory, applied to the three most used terms: *beautiful, elegant, ugly*. Takeaway: "ugly" is a genuine aesthetic category in mathematical practice, not just "not beautiful". This licenses a scale whose bottom is an active defect (brute force, sprawl), not mere absence of beauty. (Read via publisher and PhilPapers abstracts.)

**8. M. Raman-Sundström & L.-D. Öhman, "Mathematical fit: a case study", *Philosophia Mathematica* 26(2) (2018), 184–210.** https://doi.org/10.1093/philmat/nkw015
Clarifies the notion of mathematical "fit", a quality mathematicians associate with beautiful proofs, and proposes six criteria that make proofs more or less fitting. Secondary sources quote two of them: *transparency* ("the structure of the argument is clear") and *generality* (ideas that can be "readily recycled in a wider range of situations"). Takeaway: transparency of structure feeds the "simplicity of structure" dimension; recyclable ideas feed "explanatory power". *(Criteria partly secondary: full text paywalled.)*

**9. M. Lange, "Explanatory proofs and beautiful proofs", *Journal of Humanistic Mathematics* 6(1) (2016), 8–51.** https://doi.org/10.5642/jhummath.201601.04 ; https://scholarship.claremont.edu/jhm/vol6/iss1/4/
Gives an account of what makes a proof explain *why* a theorem holds, and links it to the concepts of a brute-force proof, a mathematical coincidence, unification and natural properties (per the abstract). Concludes that features that make a proof explanatory also add to its beauty, but the two virtues are not the same: a beautiful proof need not be explanatory. Takeaway: brute force is the named opposite of explanation, which justifies an inverse "computation/brute force" dimension; and explanation is distinct from beauty-as-surprise, which justifies separate "surprise" and "explanatory power" dimensions.

### B. Measuring elegance empirically

**10. D. Wells, "Are these the most beautiful?", *Mathematical Intelligencer* 12(3) (1990), 37–41.** https://doi.org/10.1007/BF03024015
The first survey: readers rated 24 theorems from 0 to 10 for beauty (68 usable responses). Euler's identity came top; famous results were not all rated highly. Takeaway: absolute 0–10 beauty ratings are possible but noisy, with wide disagreement even among experts.

**11. S. Zeki, J. P. Romaya, D. M. T. Benincasa & M. F. Atiyah, "The experience of mathematical beauty and its neural correlates", *Frontiers in Human Neuroscience* 8 (2014), 68.** https://doi.org/10.3389/fnhum.2014.00068
fMRI of 15 mathematicians viewing formulae they had rated beautiful, indifferent or ugly. Beauty ratings correlated with activity in field A1 of the medial orbito-frontal cortex, the same region as visual and musical beauty. Takeaway: mathematical beauty is a real, graded response that people can rate on a scale. It does not tell us *which features* drive it, so it supports measurability, not the rubric content.

**12. M. Inglis & A. Aberdein, "Beauty is not simplicity: an analysis of mathematicians' proof appraisals", *Philosophia Mathematica* 23(1) (2015), 87–109.** https://doi.org/10.1093/philmat/nku014 ; accepted manuscript: https://repository.lboro.ac.uk/articles/journal_contribution/Beauty_is_not_simplicity_an_analysis_of_mathematicians_proof_appraisals/9370010
The key empirical paper. Roughly 250 mathematicians (summaries differ: "more than 250" in Aberdein et al. 2021, 225 in Sa et al. 2023) rated a proof of their choice on 80 adjectives; factor analysis gave four dimensions: **aesthetics, intricacy, utility, precision**. Loadings (as summarised in Aberdein, Rittberg & Tanswell, "Virtue theory of mathematical practices: an introduction", *Synthese* 199 (2021), 10167–10180, https://doi.org/10.1007/s11229-021-03240-2): *aesthetics* — striking, ingenious, inspired, profound, creative, deep, sublime, innovative, beautiful, elegant, charming; *intricacy* — dense, difficult, intricate, unpleasant, confusing, tedious, with "simple" loading strongly negative; *utility* — practical, informative, efficient, applicable, useful; *precision* — careful, precise, meticulous, rigorous. "Beautiful" and "elegant" did not correlate strongly with "simple". Takeaways: (a) intricacy is its own axis, not the inverse of beauty, so it gets its own dimension; (b) do not equate elegance with brevity — score economy *relative to the result*; (c) precision/rigour is orthogonal to aesthetics, so correctness checking stays out of the elegance score.

**13. M. Inglis & A. Aberdein, "Diversity in proof appraisal", in B. Larvor (ed.), *Mathematical Cultures* (Birkhäuser, 2016), 163–179.** https://doi.org/10.1007/978-3-319-28582-5_10 *(secondary)*
112 mathematicians rated a BOOK proof (of Sylvester's theorem) on 20 adjectives; 60.4% placed it below the midpoint of the aesthetic scale and only 31.5% above (figures as reported by Sa et al. 2023, item 16). Takeaway: absolute aesthetic ratings of a single proof disagree strongly. Expect noise; report agreement between judges.

**14. M. Inglis & A. Aberdein, "Are aesthetic judgements purely aesthetic? Testing the social conformity account", *ZDM* 52(6) (2020), 1127–1136.** https://doi.org/10.1007/s11858-020-01156-8 *(secondary)*
203 mathematicians rated the same proof; half were told it came from *Proofs from THE BOOK*. Pure mathematicians who were told the source rated it more highly; applied mathematicians did not (as reported by Sa et al. 2023). Takeaway: source labels bias aesthetic ratings. The judge must not know whether a proof is AI- or human-written, and anchor texts carry no source label.

**15. S. G. B. Johnson & S. Steinerberger, "Intuitions about mathematical beauty: a case study in the aesthetic experience of ideas", *Cognition* 189 (2019), 242–259.** https://doi.org/10.1016/j.cognition.2019.04.008
Laypeople matched mathematical arguments to paintings and music above chance. When rating beauty, people relied mainly on **elegance, profundity and clarity** for both art and proofs. Takeaway: "clarity" and "profundity" are part of what raters mean by beauty, which supports the explanatory-power dimension. Non-experts can make these judgements, but expertise sharpens them. (Read via abstract.)

**16. R. Sa, L. Alcock, M. Inglis & F. S. Tanswell, "Do mathematicians agree about mathematical beauty?", *Review of Philosophy and Psychology* 15(1) (2024; online 2023), 299–325.** https://doi.org/10.1007/s13164-022-00669-3 ; open PDF: https://d-nb.info/1287248748/34
Used **comparative judgement** (pairwise "which is more beautiful?", fitted with a Bradley–Terry model) instead of absolute scales. Found agreement within and across British mathematicians, Chinese mathematicians and undergraduates (inter-rater reliability 0.70–0.72 for equations; 0.643 for 8 proofs, 31 mathematicians). Euclid's proof of the infinitude of primes ranked most beautiful; an algebraic computation ranked least. Takeaway: disagreement in earlier studies partly reflects absolute scales; relative judgements show real consensus. This is the strongest argument for anchoring absolute scores to fixed reference proofs, or for pairwise scoring.

**17. J. P. Mejía Ramos, T. Evans, C. Rittberg & M. Inglis, "Mathematicians' assessments of the explanatory value of proofs", *Axiomathes* 31(5) (2021), 575–599.** https://doi.org/10.1007/s10516-021-09545-8 ; preprint: https://mjinglis.github.io/files/Axiomathes2021.pdf
38 mathematicians judged nine proofs of one proposition for explanatory value by comparative judgement. Split-half reliability was very high (0.88–0.95). But presentation mattered: the same argument written as an "elementary" proof (parameter 1.52) and as a "two-column" proof (−0.18) got very different scores. Visual and experimental arguments ranked least explanatory. Takeaway: explanatoriness is measurable with high agreement, but it is confounded with *presentation format*. AI manuscripts share one house style, so the rubric tells the judge to ignore polish and format.

**18. G. Kinnear & M. Inglis, "Does understanding moderate aesthetic appraisals of proofs?", *Journal of Mathematical Behavior* 81 (2026), 101284.** https://doi.org/10.1016/j.jmathb.2025.101284 (open access, CC-BY)
Tests whether students' understanding of a proof shapes their aesthetic appraisal of it. The results challenge the view that aesthetic judgements in mathematics are disguised epistemic judgements. Takeaway: aesthetic and epistemic appraisals come apart; a judge's elegance score is not just a proxy for "I followed it". (Read via abstract.)

### C. Computer-assisted and "ugly" proofs

**19. T. Tymoczko, "The four-color problem and its philosophical significance", *Journal of Philosophy* 76(2) (1979), 57–83.** https://doi.org/10.2307/2025976
Argues that a proof must be convincing, *surveyable* (checkable by a human) and formalisable, and that the Appel–Haken four-colour proof is not surveyable, which makes mathematical knowledge partly empirical. Takeaway: "surveyability" is the classic name for what the inverse-computation dimension measures. A score of 1 means "a human cannot survey the core".

**20. R. Thomas, "An update on the four-color theorem", *Notices Amer. Math. Soc.* 45(7) (1998), 848–859.** https://thomas.math.gatech.edu/PAP/update.pdf (summarising N. Robertson, D. Sanders, P. Seymour & R. Thomas, "The four-colour theorem", *J. Combin. Theory Ser. B* 70 (1997), 2–44)
Quotes Appel and Haken on their own proof: "50 pages containing text and diagrams, 85 pages filled with almost 2500 additional diagrams, and 400 microfiche pages", plus about 1200 hours of computer time. The 1997 proof cuts this to 633 configurations and 32 discharging rules, with a 13,000-line formal unavoidability case analysis; some rules "were obtained by trial and error ... not designed with any geometric intuition behind them". Takeaway: the canonical messy anchor, and evidence that a messy proof can be *improved* without becoming elegant.

**21. T. C. Hales, "A proof of the Kepler conjecture", *Annals of Mathematics* 162 (2005), 1065–1185** (https://annals.math.princeton.edu/wp-content/uploads/annals-v162-n3-p01.pdf); **T. Hales et al., "A formal proof of the Kepler conjecture", *Forum of Mathematics, Pi* 5 (2017), e2** (https://doi.org/10.1017/fmp.2017.1 ; arXiv:1501.02155).
The proof reduces the conjecture to thousands of "tame" graphs, about 10^5 linear programs and nearly a thousand nonlinear inequalities (over 23,000 cases) checked by interval arithmetic. The referees could not fully certify it; the formal version needed about 5000 processor-hours for the inequalities alone. Takeaway: a messy anchor that still has a real conceptual reduction, so it sits just above pure certificates.

**22. M. Heule, O. Kullmann & V. Marek, "Solving and verifying the Boolean Pythagorean triples problem via Cube-and-Conquer", SAT 2016, LNCS 9710, 228–245.** https://doi.org/10.1007/978-3-319-40970-2_15 ; arXiv:1605.00723. Press: **E. Lamb, "Two-hundred-terabyte maths proof is largest ever", *Nature* (26 May 2016)**, https://www.nature.com/articles/nature.2016.19990.
Settles Graham's question (7825 is the threshold) with 10^6 SAT subproblems, about 4 CPU-years, and a ~200 TB DRAT certificate (68 GB compressed). Takeaway: the floor of the scale. A correct answer with no human-readable reason why.

**23. J. D. Phillips & D. Stanovský, "Bruck loops with abelian inner mapping groups", *Communications in Algebra* 40(7) (2012), 2449–2454** (preprint and prover files: https://www.karlin.mff.cuni.cz/~stanovsk/math/bruck.htm); **K. Buzzard, "A computer-generated proof that nobody understands", Xena blog (6 July 2019)**, https://xenaproject.wordpress.com/2019/07/06/a-computer-generated-proof-that-nobody-understands/.
A research question in loop theory: a short human reduction, then the prover Waldmeister found a proof of the key identity. The authors report "over 16,000 lines of raw output, or over 1000 pages of structured equational reasoning", and write that "the proof is far too long to translate into a 'human friendly' form" (they pose finding one as an open problem). Buzzard contrasts it with the Robbins conjecture (1996), whose automated proof was short enough to be "de-computerised" into a seven-page paper. Takeaway: the closest pre-LLM analogue of an AI research proof. Automation can produce either a short proof or an unreadable one; the rubric must separate the two.

### D. AI-generated proofs, 2024–2026

**24. G. Burnham, "What will the IMO tell us about AI math capabilities?", Epoch AI Gradient Updates (8 July 2025).** https://epoch.ai/gradient-updates/what-will-the-imo-tell-us-about-ai-math-capabilities
"The one very hard problem that AlphaProof solved on the 2024 IMO, it solved in a very brute-force way", while the human solution builds up abstract properties of the function; "All AI systems show a lack of creativity." Takeaway: the "AI is messy" prior, stated for competition maths. Opinion piece, not a measurement.

**25. I. Petrov, J. Dekoninck, D. I. Dimitrov & M. Vechev, "Not all proofs are equal: evaluating LLM proof quality beyond correctness" (ProofRank), arXiv:2605.10379 (v2, 25 June 2026).** https://arxiv.org/abs/2605.10379
The only benchmark found that scores the *style* of LLM proofs. Proof-level metrics: conciseness (how much of the text an LLM can delete without changing the argument), computational ease ("avoids brute-force algebra, enumeration, case checking"), and cognitive simplicity; the last two are pairwise, judged by GPT-OSS-120B. Problem-level metrics: diversity and adaptivity. Their elegance prompt (used in a human validation study): high elegance uses "a clever trick, symmetry, an invariant, or a principled shortcut ... and produces an 'aha' moment"; low elegance is "brute force: long algebra/coordinate bashing ... large/manual casework". Findings: large quality differences between models that correctness benchmarks miss; trade-offs between quality and correctness; 31% of correct LLM solutions fall outside the human solution clusters (often coordinate, complex-number or trigonometric bashing in geometry, and obscure literature techniques in number theory). Methods note: they tried 0–5 ordinal rubrics, for both LLM and human raters, and abandoned them as too noisy; pairwise agreement among their human annotators was 89% for elegance, but the annotators were co-authors. Takeaways: (a) independent support for the computation and economy dimensions; (b) LLMs *do* produce non-human solution styles; (c) absolute rubric scores are noisy — anchor them, or use pairwise comparisons.

**26. T. Tao, Mastodon thread (21 June 2026).** https://mathstodon.xyz/@tao/116789373239346609
"It is now easier to generate long correct proofs than it is to generate short correct proofs!" In his formalisation project, AI tools "tended to create quite bloated proofs, often hundreds of lines longer than what a human would choose to do, with a lot of redundancy, with many lemmas not stated at the natural level of abstraction." Takeaway: the strongest expert statement that AI pulls proofs toward low economy. It is about Lean formalisations, not the informal manuscripts in the corpus, so it is a hypothesis for our data, not a result.

**27. T. Tao, "Mathematics in the age of AI", arXiv:2608.16753 (essay for the Proceedings of ICM 2026).** https://arxiv.org/abs/2608.16753 ; summary in Tao's reviewed "living summary": https://teorth.github.io/tao-web/ai-views.html
Frames AI as speeding up proof *generation* far more than *digestion*, and lists "creating work of aesthetic value" among the profession's goals that AI could pull apart from problem-solving (Goodhart's law). The summary reports his view that AI exposition "dwells at length on trivialities" while hurrying past novel steps, and risks being too slick, removing the "natural friction" that helps readers learn. Takeaway: two predicted AI failure modes — low economy (padding on trivia) and polish that hides weak explanation. The second is a judge risk: the rubric tells the judge to ignore polish.

**28. B. Kra, "'Deep theorems were scarce and difficult and so became an effective mechanism to identify deep thought.' AI has broken this system", guest post on Tao's blog (13 Sep 2026).** https://terrytao.wordpress.com/2026/09/13/deep-theorems-were-scarce-and-difficult-and-so-became-an-effective-mechanism-to-identify-deep-thought-ai-has-broken-this-system/
On the Nivat conjecture, two human approaches (dynamical and algebraic) had resisted unification for years: "synthesizing different approaches is one of the ways in which frontier models excel. A machine does not need years to absorb distinct areas of mathematics before it can find the connections." Takeaway: an expert statement in favour of Esben's half of the argument (AI imports insight across fields). This is the reason the rubric scores cross-field unification separately from elegance: it lets the post test "messy *and* cross-field" rather than collapsing both into one number.

### E. The openai/math release (6 October 2026)

**29. OpenAI, `openai/math` repository: README and history.** https://github.com/openai/math (local clone `/tmp/oai-math`)
719 manuscripts in 372 families from an unreleased internal model; about 4,000 problems posed; "on average, each result used three hours of ChatGPT Pro thinking compute"; about 42% of top-line results formalised in Lean. One writeup (a zeta zero-free region) "was human edited for readability". On 7 October three manuscripts were withdrawn after a sign error, and 14 others were repaired. Takeaway: define the corpus as the 719 current manuscripts, exclude the withdrawn ones, and flag the human-edited one.

**30. D. Castelvecchi, "OpenAI posts 700 maths preprints online: mathematicians are up in arms", *Nature* (7 Oct 2026).** https://doi.org/10.1038/d41586-026-03196-8
Reception: some celebrated long-standing problems solved; others complained about being scooped; some were incensed at what one physicist called a "slopocalypse", "even if the mathematical content could end up being formally correct". Takeaway: the public debate is about volume, credit and norms; nobody has yet assessed the *style* of these proofs, which is the gap this post fills.

**31. Association for Human Mathematics, "Statement on OpenAI's October 6 release of mathematical documents", reposted on Tao's blog (7 Oct 2026).** https://terrytao.wordpress.com/2026/10/07/ahm-statement-on-openais-october-6-release-of-mathematical-documents/
"Releasing over 700 files at once is not a demonstration of scholarship, but a demonstration of power", and a call to "return to a vision of science that centers human understanding." Takeaway: the "understanding" framing (Thurston, Rota) is the live objection to AI proofs. An explanatory-power dimension is therefore the most contested part of the score and should be reported on its own, not only inside the overall score.

---

## 2. What elegance is, operationally

### Principles taken from the literature

1. **Elegance is not brevity.** "Beautiful" and "elegant" barely correlate with "simple" (Inglis & Aberdein 2015). Tao defines elegance as a difficult result with minimum effort (Tao 2007). So economy is scored *relative to the result*.
2. **The aesthetic core is surprise plus insight.** The aesthetics factor is striking, ingenious, creative, deep, profound (Inglis & Aberdein 2015). Hardy pairs *unexpectedness* with *inevitability* (Hardy 1940, §18). Rota and Johnson & Steinerberger add enlightenment and clarity. These become two separate dimensions, because a proof can be surprising but unexplanatory (a trick) or explanatory but expected.
3. **Intricacy is its own axis.** It is a separate factor, not the inverse of beauty (Inglis & Aberdein 2015), and Hardy names case enumeration as dull (§18).
4. **Brute force and computation are a separate defect from intricacy.** A proof can have one line of attack and still be a giant computation (Kepler LPs, SAT). The literature names this as non-surveyability (Tymoczko 1979), brute force as the opposite of explanation (Lange 2016), and low computational ease (ProofRank 2026).
5. **Cross-field unification is a different good.** Hardy ties *seriousness* to connecting many ideas; Tao lists unification and cross-field application as separate senses of "good" from elegance (Tao 2007); Kra says AI is especially good at it (Kra 2026). It is scored on its own and kept out of the overall score, so the post can test Esben's "messy but cross-field" claim directly.
6. **Out of scope on purpose:** the importance, depth or difficulty of the result (Hardy's seriousness; Tao's "deep", "strong"); precision and rigour (a separate factor in Inglis & Aberdein 2015; correctness is checked elsewhere); and writing polish (it confounds explanatoriness — Mejía Ramos et al. 2021 — and AI prose is uniformly polished — Tao 2026).

### The dimensions (higher = more elegant on all of D1–D5)

**D1. Economy** — how little length and machinery the proof spends relative to the strength of the result.
Sources: Hardy (economy); Tao 2007 (xv); ProofRank conciseness; Tao 2026 (bloated AI proofs).
- 1 — Far longer or heavier than the result needs: redundant lemmas, repeated arguments, heavy machinery for a light step.
- 2 — Clearly padded: whole sections could go or be replaced by a citation or a line.
- 3 — Reasonable length for the result, with some detours.
- 4 — Tight: only small savings possible.
- 5 — Almost nothing can be removed; a strong result from light means.

**D2. Surprise of the key idea** — is there a striking, non-obvious idea that makes the proof work?
Sources: Hardy (unexpectedness); Inglis & Aberdein aesthetics factor (striking, ingenious, creative); ProofRank elegance prompt ("aha" moment).
- 1 — No key idea: the standard method is pushed through.
- 2 — Standard method with a minor clever step.
- 3 — A sensible idea that an expert would try early.
- 4 — A clever idea an expert might find after real effort.
- 5 — An unexpected move (auxiliary object, change of viewpoint, construction) that an expert would not guess, and that works cleanly.

**D3. Explanatory power (inevitability)** — does the proof show *why* the result is true, so that it feels inevitable once seen?
Sources: Hardy (inevitability); Rota (enlightenment); Lange 2016; Thurston 1994; Mejía Ramos et al. 2021; Raman-Sundström & Öhman (recyclable ideas).
- 1 — It verifies *that* the result holds but gives no reason (certificate, unmotivated computation).
- 2 — A reason exists but is hidden; the reader learns little.
- 3 — The main reason is visible, mixed with steps that only verify.
- 4 — The reason is clear; a few steps feel unmotivated.
- 5 — The result feels forced; you could rebuild the proof from its core idea, and the idea transfers to other problems.

**D4. Simplicity of structure (inverse of intricacy)** — few cases, few moving parts, one line of attack.
Sources: Hardy ("enumeration of cases"; "clear-cut constellation"); Inglis & Aberdein intricacy factor (dense, intricate, tedious, confusing; "simple" negative); Raman-Sundström & Öhman (transparency); ProofRank cognitive simplicity.
- 1 — Many cases and sub-cases, long lemma chains patching special situations, dense bookkeeping.
- 2 — Several cases or a long dependency chain; hard to keep in mind.
- 3 — A few cases or a moderate lemma chain.
- 4 — Essentially one line of attack, with minor side cases.
- 5 — One line of attack; no case split, or a trivial one; you can hold it in your head.

**D5. Conceptual, not computational (inverse of brute force)** — does the proof rest on concepts rather than heavy calculation, enumeration or machine search?
Sources: Tymoczko 1979 (surveyability); Lange 2016 (brute force); ProofRank computational ease; computer-proof anchors (Thomas 1998, Hales 2005, Heule et al. 2016, Phillips & Stanovský 2012); Burnham 2025.
- 1 — The core is exhaustive search, a machine certificate, or computation a human cannot survey.
- 2 — Long explicit computations or estimates carry most of the weight, though a human could check them.
- 3 — A conceptual skeleton, with substantial routine computation or estimates in the middle.
- 4 — Mostly conceptual; short computations.
- 5 — No heavy computation; any calculation is short and easy to check by hand.

**D6. Cross-field unification (scored separately; does NOT feed the overall score)** — does the key idea bring in a concept or structure from another field, and is that the reason the proof works?
Sources: Hardy (significance as connection with "a large complex of other mathematical ideas"); Tao 2007 (iv), (vi); Kra 2026; ProofRank "insightfulness" ("a meaningful connection between ideas/areas"). Using software, SAT/LP solvers or generic computation is *not* cross-field insight.
- 1 — Stays inside the problem's own field and toolkit.
- 2 — Cites a result from another field as a black box.
- 3 — Borrows a technique from a neighbouring field as a tool, without changing how the problem is seen.
- 4 — Imports a structure from another field that does real work in the argument.
- 5 — Recasts the problem in another field's language (algebra for geometry, probability for combinatorics, topology for number theory), and the recasting is why the proof works.

### Overall elegance (1–10)

One holistic judgement: *how much the proof achieves with how little — a surprising but inevitable key idea, carried by a simple, conceptual structure.* It is not an average of D1–D5, and D6 is not used.
- 9–10 — A "book proof": short, surprising, inevitable; nothing to remove. (Anchors: Erdős, Fisk, Dvir, Huang.)
- 7–8 — A clearly elegant core with some necessary technical bookkeeping. (Anchor: cap set; Furstenberg if the judge finds it more trick than explanation.)
- 5–6 — A competent, standard proof: the expected method, moderate length and computation. (No anchor: the expected bulk of ordinary research papers.)
- 3–4 — The idea is buried under long case analysis, heavy estimates or computation. (Anchors: Ringel–Youngs, de Grey.)
- 1–2 — Brute force: exhaustive cases, machine search or a certificate; no reason a human can grasp. (Anchors: four colour, Kepler, Pythagorean triples, Bruck loops.)

Rules for the judge (also in `rubric.md`): judge the proof of the main result; assume correctness; ignore polish, format and the importance of the result; judge length relative to the result.

---

## 3. Measurement design consequences

- **Anchor or compare, do not rate in a vacuum.** Absolute aesthetic ratings disagree (Wells 1990; Inglis & Aberdein 2016), while relative judgements agree well (Sa et al. 2023: 0.64–0.72; Mejía Ramos et al. 2021: 0.88–0.95). ProofRank abandoned 0–5 rubrics as too noisy. Minimum for this post: score the 12 anchors with the same prompt and check that they separate. A pairwise Bradley–Terry pass is the stronger design if the absolute scores turn out noisy — keep it as a pointer, not new scope.
- **Blind the source.** Labels move ratings (Inglis & Aberdein 2020). Strip author names, "OpenAI", acknowledgements, model mentions and repository boilerplate from AI and human papers alike before judging.
- **Control presentation.** Format changes perceived explanatoriness (Mejía Ramos et al. 2021), and AI papers share one polished house style (Tao 2026). [INFERENCE] The "ignore polish" instruction may not be enough; a cheap check is to have the judge also score a short neutral summary of each proof's key steps and compare.
- **Score the main proof, not the paper.** Corpus papers have literature reviews and side results. Score the proof of the headline theorem only.
- **Report D6 next to the overall score.** The post's claim needs two numbers per proof: overall elegance (Tanya's axis) and cross-field unification (Esben's axis). A 2D scatter would show whether AI proofs are "messy but cross-field" — but the one-figure rule means this is the figure only if it replaces the 1D distribution, not in addition to it.
- **Report judge agreement.** Two judges (Claude, Gemini) give a cheap reliability number; the empirical literature says to expect moderate agreement on absolute scores.

---

## 4. What is missing

- **No direct prior study.** I found no study comparing elegance distributions of AI-written and human-written *research* proofs. ProofRank (2026) is the closest, but it covers competition problems and compares models with each other.
- **Full texts not read:** Inglis & Aberdein 2015 (factor loadings taken from Aberdein et al. 2021's summary), Raman-Sundström & Öhman 2018 (only two of six criteria known), Rota 1997, Montano 2014, Johnson & Steinerberger 2019, Kinnear & Inglis 2026 (abstracts only), Inglis & Aberdein 2016 and 2020 (via Sa et al. 2023).
- **Gowers and Buzzard on AI-proof aesthetics, 2025–2026:** not found in a citable form. Buzzard's recent Xena posts ("Human mathematicians are being outcounterexampled", 20 Jul 2026; "To grieve, or not to grieve?", 1 Oct 2026) were seen in a sidebar but not read. Gowers's 2024 AlphaProof comments circulate only as press quotes.
- **openai/math reception is two days old.** No mathematician has published an assessment of the *style* of these manuscripts yet. Fortune (Dan Litt) and Scientific American (Andrew Sutherland) quotes appeared only in search snippets and are not used.
- **Anchor coverage:** no messy anchor from analysis or mathematical physics, which are large parts of the AI corpus (see `anchors.md`).
