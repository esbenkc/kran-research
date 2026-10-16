# Pre-registration: push one MLP neuron at a time

Posted in room `oracle-1015-already-clean`, message #1175, 2026-10-16 16:48 Copenhagen time, after generation
started and before any judge label existed. Text as posted (Danish), with an English translation.

> PRE-REGISTRATION (opfølgning, før dommeren kører): (1) Ved push til max-aktivering viser >=80% af de udtrukne
> neuroner ingen synlig effekt (p=0.7). (2) Blandt neuroner med synlig effekt er 'én tema' (mono) hyppigere end
> 'flere temaer' (poly), ved både 1x og 10x (p=0.6). (3) Neuroner giver oftere synlig effekt end en tilfældig
> retning med samme styrke (p=0.65). (4) Effekterne samler sig i neuronerne med de største max-aktiveringer i hvert
> lag (p=0.75). Hvis (2) fejler: steering-billedet støtter ikke 'neuroner har ét job' på output-siden. Hvis (3)
> fejler: neuron-basen er ikke særlig på output-siden.

1. Pushed to its max activation, at least 80% of the sampled neurons show no visible effect (p = 0.7).
2. Among neurons with a visible effect, one theme (mono) is more common than several unrelated themes (poly), at
   both 1x and 10x (p = 0.6).
3. Neurons give a visible effect more often than a random direction with the same strength (p = 0.65).
4. The effects gather in the neurons with the largest max activations in each layer (p = 0.75).

If (2) fails, steering does not support "neurons have one job" on the output side. If (3) fails, the neuron basis
is not special on the output side.

## Changes after pre-registration (none touch the predictions)

- Sample: a fixed random 512 of 3,072 neurons per layer at 1x (14,336 of 86,016), and the first 256 of the same
  order at 10x. Judging all 86,016 with the local judge did not fit before 23:30 (timing test, room #1169).
- Judge: the local judges (Qwen3-4B, Qwen3-8B) failed validation on blind hand labels, so the post's judge
  (GPT-4.1-mini) labels every run, in two stages (room #1184, #1186, #1191). Rubric and stages were fixed on a dev
  set (pool a + dev_x10) and then measured on a held-out set (pools b + c) before the full labels were read.
