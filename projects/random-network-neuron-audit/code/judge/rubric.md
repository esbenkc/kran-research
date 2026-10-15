You will see the 20 text snippets where one hidden unit of a language model fires most strongly. In each snippet, the token where the unit fires is wrapped in «guillemets». The number in [brackets] is the unit's firing strength on that token, scaled so the strongest snippet is 10.

You do not know which model or which kind of unit this is. Judge only what you see.

Your job: say what the unit responds to, and how many separate jobs it has.

Definitions:
- A JOB is one description that a careful reader could use to predict new snippets where the unit fires. Examples of single jobs: "the token ' because'", "closing parentheses in code", "text about legal contracts", "the word after a number in a list".
- Two descriptions are the SAME job if one is a special case of the other or they share an obvious common pattern (for example "'diy'" and "'do it yourself'"). Do not split a job into nouns if one pattern covers them.
- A snippet FITS a job if the marked token, read with its context, matches the description. Judge the marked token, not the whole snippet.

Classify the TYPE of the main job:
- LEXICAL: the unit keys on the identity or spelling of the marked token (the same word, word piece, or close variants such as capitalisation or a shared substring).
- SYNTACTIC: the unit keys on a grammatical, punctuation, formatting, or position role that many different tokens can fill (for example "token after an opening quote", "end of a line in code", "second word of a sentence").
- SEMANTIC: the unit keys on a topic, concept, or context that many different tokens can carry (for example "medical text", "talk about the weather", "mentions of a person's emotions").
- NONE: no job covers 5 or more of the 20 snippets.

Procedure:
1. Find the single job that fits the most snippets. Count how many of the 20 fit it (main_count).
2. List any other jobs that each fit 2 or more of the remaining snippets, with their counts. Say for each whether it is RELATED to the main job (shares a clear pattern or topic) or UNRELATED.
3. Count snippets that fit no job (unexplained_count). All counts must add up to 20.

Answer with JSON only, no other text, in exactly this form:
{"main_label": "<short description, max 15 words>", "main_type": "LEXICAL|SYNTACTIC|SEMANTIC|NONE", "main_count": <0-20>, "other_jobs": [{"label": "<short>", "count": <int>, "related": true|false}], "unexplained_count": <int>, "confidence": "low|medium|high"}
