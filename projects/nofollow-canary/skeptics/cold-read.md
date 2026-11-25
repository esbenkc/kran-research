# Cold read (Lumi, reader), preview PR #18, round 1

COLD READ, preview PR #18 /nofollow-canary (head with 'costs a honeypot catches'). Please save this verbatim as skeptics/cold-read.md (I do not edit files).
MAIN CLAIM (my words): Removing or adding the nofollow tag on a hidden honeypot link moved Qwen3-8B's request rate by only 2-4 points, so the tag does not clearly explain the raw-HTML gap, but the intervals still allow a ~13-point effect.
FIGURE ALONE: main figure yes (head '2 to 4 points', answer line 'Not settled', numbers = analysis.json). Figure 1: see #30550.
LOST (8):
L1 Prereg foldout. 'What we expected' says only 'within 10 points'; then 'It was wrong.' But tag-free raw HTML 82% vs markdown 91% (8.9 pts) and markdown+tag 78% vs raw 80% BOTH meet 'within 10'. The 'clear of zero' part (PREREG.md:31-32) is missing from 'expected'. FIX expected: add 'Each change would also be clearly above zero by its 95% interval.' FIX outcome: 'Both pages landed within 10 points, but neither change was clearly above zero.'
L2 TLDR + Fig1: 'A clear No also missed by one run.' No is never defined there. -> 'A No needed tag-free raw HTML clearly below markdown; it missed by one run.'
L3 TLDR: 'Adding the tag to a link in a markdown page changed them from 74 to 70 of 90.' Fig1 just showed markdown = 82; which markdown? -> 'Adding the tag to an HTML link inside the markdown page changed them from 74 to 70.'
L4 Same bound, 3 numbers: TLDR 'a 13-point effect', limits + stance 'up to about 14 points', loop-closer 'cannot rule out a 10-point effect'. Use one number (13) everywhere, or say 10 is the predicted size.
L5 'Our next run tests a hint from this one. With no tag on either, ...' The hint is never named; 'either' has no noun. -> 'The hint: the link form may matter more than the tag. Untagged HTML links drew 74, markdown links 82.'
L6 Unfilled slots in the expert foldout: 'Iris ... are in skeptics/skeptic. «critic2, critic-local»' and 'Cost: «$» of model API'.
L7 Broken code span, Formats (preview HTML line 433): 'drops only ` rel=”nofollow” from the two hidden links ... writes the two hidden links as words ... adds rel=”nofollow”` to both'. A stray backtick plus curly quotes swallowed text; 'as words' lost its <a href> part.
L8 Intro: 'when it read the same page as markdown.' Nothing says who makes the markdown. -> 'Our page-reading tool gave the model either the raw HTML or the page as markdown.' (agent.py:9-10, view_md:131)
READ TWICE (8):
READ TWICE: 'The gap came out smaller than we planned for: 80% against 91%. Suppose the tag does nothing. Our rule then gives a No in 32% of simulated runs at this gap' -> REWRITE: 'Raw HTML trailed markdown by 11 points, not the planned 19. With no tag effect, our rule says No in 32% of runs.'
READ TWICE: 'We asked whether this tag costs the trap its catches of AI agents.' -> REWRITE: 'We asked whether this tag makes the trap catch fewer AI agents.'
READ TWICE: 'So we see no clear sign that the tag costs a honeypot catches.' -> REWRITE: your line 'makes a honeypot catch fewer agents' (also Fig1 + figure1_alt:19).
READ TWICE: 'Neither tag change was larger than zero by its 95% interval.' -> REWRITE: 'Neither tag change had a 95% interval fully clear of zero.'
READ TWICE: 'Each change's whole 95% interval also had to sit above zero.' The adding change is a drop, so 'above' reads wrong. -> REWRITE: 'Each change's 95% interval also had to exclude zero, in the predicted direction.'
READ TWICE: 'If I ran a nofollow canary, I would keep the tag for now.' Only 'I' in a 'we' post; who is I? -> REWRITE: 'We would keep the tag on a nofollow canary for now.'
READ TWICE: 'small marks show each third of the runs' (main fig caption) -> REWRITE: 'small marks show each block of 30 runs'.
READ TWICE: 'Runs with the same number read the same site with the same task.' Fig1 shows different link codes for the pair. -> REWRITE: 'Runs with the same run number read the same site and task; only the link codes differ.'
Not a fix: the 63-word question under the title is verbatim by rule.
Send me the new head; I re-grep every fix string and old string before my pass.
