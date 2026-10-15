Someone studied one hidden unit of a language model and wrote this description of what it responds to:

DESCRIPTION: {label}

Below are 10 text snippets. In each, one token is wrapped in «guillemets». In exactly 5 of the snippets the unit fires strongly on the marked token. In the other 5 the marked token was picked at random.

Using only the description, pick the 5 snippets where the unit most likely fires strongly on the marked token. You must pick exactly 5 different numbers from 1 to 10.

Answer with JSON only, no other text, in exactly this form:
{{"picks": [<n1>, <n2>, <n3>, <n4>, <n5>]}}
