You will see up to 20 text snippets where one hidden unit of a language model fires most strongly (one side of the unit only). In each snippet the token where the unit fires is wrapped in [[double brackets]]. The number in (parentheses) is the firing strength on that token, scaled so the strongest snippet is 1.00. Snippets 1-12 are the strongest; snippets 13-20 are a random sample of other strong firings. "src" is the corpus the snippet came from.

Your job: propose 1 to 3 hypotheses for what makes this unit fire, and write test texts for each one so a script can check the hypothesis on the real model.

Rules for hypotheses:
- A hypothesis is ONE description a careful reader could use to predict new texts where the unit fires (for example "the token ' Colorado'", "the Python library Tornado in import lines", "the second word of a sentence", "Spanish verbs in the past tense", "closing brackets in JSON").
- Give one hypothesis if one description covers most snippets. Give 2 or 3 only if the snippets fall into groups that one description does not cover. Do not split a pattern into special cases.
- "covers" lists the snippet numbers the hypothesis explains.
- "kind" is one of: token (identity or spelling of the marked token), semantic (topic or concept), syntax (grammar, punctuation, format), positional (where in the text), code (programming-specific), cross-lingual (the same thing across languages), other.

Rules for test texts (the script measures the unit's strongest firing anywhere in each text):
- "fire": 4 new texts, each 20-60 words, in which the hypothesis predicts a STRONG firing. Make them varied and natural. Do not copy the snippets.
- "control": 4 texts, one for each "fire" text, in the same order. Each is a MINIMAL edit of its fire text that removes only the feature the hypothesis names (for example swap "Tornado" for "aiohttp", swap the word for a same-length word with a different meaning, or move the token to another place). Keep everything else identical.
- Write the texts so that a different hypothesis would not also predict a firing in the control.

Answer with JSON only, no other text, in exactly this form:
{"hypotheses": [{"name": "<max 10 words>", "kind": "<kind>", "description": "<one sentence>", "covers": [<snippet numbers>], "fire": ["...", "...", "...", "..."], "control": ["...", "...", "...", "..."]}]}

SNIPPETS
