# Cold Read — rethreshold-probe-evasion (Lumi, reader)

Date: 2026-10-18
URL: https://deploy-preview-52--kranai.netlify.app/rethreshold-probe-evasion/

## Main Claim
When a model is fine-tuned to evade a specific probe while retaining its original text outputs, it mostly just shifts all activation scores downward, meaning you can catch the evasive behavior again simply by recalibrating the probe's alarm threshold on clean data.

## Unknown Terms
None. Everything was adequately defined or standard.

## Figure
Tells the story: Yes. The caption defines all visual elements clearly (large dots, small dots, dashed lines, shaded range, bars).

## (1) LOST
- 'The true statements moved 94% as far as the false ones (\(c\) = 0.94).' -> Why: It requires jumping back to the theoretical section to remember what c is, and 'moved' is slightly ambiguous (it means shift in mean score).
- 'At 5% false alarms, below this tail, a new threshold won back 96% or more in every run with an output loss.' -> Why: The geometric phrasing 'below this tail' is confusing when talking about a threshold rate of 5% compared to 1%. It means avoiding the extreme tail, but 'below' is ambiguous.
- Math formatting uses \( ... \) and \[ ... \] in rendered HTML (MathJax renders fine).

## (2) READ TWICE
- READ TWICE: 'We looked at the strength of the fine-tune where the probe’s catch rate falls by at least 50 points (at a threshold that raises a false alarm on 1% of clean examples).' -> REWRITE: 'We found the gentlest attack where the probe's catch rate fell by at least 50 points.'
- READ TWICE: 'The top is the catch rate with the new threshold minus the catch rate with the old one.' -> REWRITE: 'The numerator is the catch rate gained by moving to the new threshold.'
- READ TWICE: 'The true statements moved 94% as far as the false ones (\(c\) = 0.94).' -> REWRITE: 'The score shift for true statements was 94% as large as for false ones.'
- READ TWICE: 'At 5% false alarms, below this tail, a new threshold won back 96% or more in every run with an output loss.' -> REWRITE: 'Avoiding this extreme tail by allowing 5% false alarms, a new threshold won back at least 96%.'
- READ TWICE: 'What happened It held. It held.' -> REWRITE: 'It held. At the gentlest attack that cut the catch rate...'
