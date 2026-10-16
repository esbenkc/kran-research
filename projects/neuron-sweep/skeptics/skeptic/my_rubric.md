# skeptic's own blind rubric (written before reading any judge label)

Input: the 3 sampled continuations of one row. Arm, layer, neuron hidden; rows shuffled across arms.
Reference: what the unmodified Qwen3-0.6B writes from the same start (data/refs.json, 9 texts): fluent,
generic, varied openings ("Today, we...", "The United States is...").

- none: all 3 read like the reference texts. Fluent, generic, no shared unusual word, topic, symbol or style.
  Mild repetition that the reference model also shows ("the first day of the first...") is not an effect.
- mono: at least 2 of the 3 texts share ONE unusual thing (a word, a topic, a symbol run, a language,
  a format) that the reference texts do not have.
- poly: the texts are clearly abnormal, but in 2 or more different, unrelated ways
  (e.g. one is a digit run, one is about cooking), with no single shared thing.

Ties: if only 1 of 3 texts is odd and the other 2 are normal -> none (one odd sample is within chance).
Degenerate gibberish of the same kind in >=2 texts -> mono (one thing: that degeneracy).
