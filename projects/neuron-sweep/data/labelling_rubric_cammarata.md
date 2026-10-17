# Labelling rubric (Cammarata scheme), written 2026-10-17 before labeller 2 started

Each item is three short texts. A small language model (Qwen3-0.6B) wrote each as a continuation of the prompt
"The following is a short passage of text." (about 20 tokens each). Some items come from a model that was changed;
some may be untouched. Label each item with exactly one of:

- none: all three could easily be typical continuations (see the nine typical texts below). Shared openings,
  common words, or one slightly odd detail in one text do not count.
- one: a specific atypical thing appears in at least two of the texts, and it is one thing or closely RELATED
  things (a word and its translation; a topic and neighbouring topics; a word and longer words containing it;
  a symbol and similar symbols; one token repeated). A third text that is typical or odd does not change this.
- unrelated: at least two texts carry specific atypical things you can each name, and those things have no
  sensible link to each other (no shared thing in two texts). This is Cammarata's sense of polysemantic:
  "multiple unrelated jobs".
- broken: at least two texts are clearly atypical, but only garbled or repetitive in generic ways (different
  function words repeated, stray punctuation), with no specific thing you could name.

Source for the definition: Cammarata 2026, "Neural Networks Are Already Clean" (polysemantic = "a neuron that
appears to have multiple unrelated jobs"; recursive generalization = related offshoots, which count as one job).

## Nine typical continuations (no change to the model)
R1: "Here are some other books on this topic, including those by the author, and they are all
R2: "During the 20th century, we can find many species that have been successfully adapted to
R3: "Today, we have a number of different events that happen on a daily basis. These events are
R4: "Many people are trying to make a difference in the world, and there's a lot of hope
R5: "Today, we will be discussing a particular topic that is currently under discussion in the field of art
R6: The United States is the second-largest country in the world. It has a population of over 3
R7: "Let’s think about how to apply the concept of ‘sustainability’ to a real-world problem
R8: "Since the beginning of the last century, the number of people who have been affected by the disease
R9: The first thing that I can tell you is that the main reason the system is so much more powerful
