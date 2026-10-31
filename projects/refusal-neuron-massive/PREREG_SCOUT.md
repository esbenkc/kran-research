# Scout pre-registration (verbatim)

Copied verbatim from the room goal message #9842 (room theory-1031-refusal-neuron); never edited.

```
PRE-REGISTERED PREDICTION (fixed 2026-10-31 03:33 CET by the scout and the tribunal, before any experiment; never change it):
Prediction: Phase 0 passes. The gate neuron writes >= 50% of its output norm into the top-3 massive channels, against a median < 5% for random neurons. Removing only its massive-channel write keeps >= 60% of its refusal effect. Random neurons pushed through those channels at matched norm get >= 50% of the gate's effect.
P(prediction holds): 0.35
If it holds: 'One neuron bypasses safety' is an outlier-channel artefact: any unit that can move the massive channel can gate refusal. Neuron-level safety claims need an outlier control. A defence can watch or clamp one channel. Our evil-qwen reading changes from 'a safety unit' to 'a lever on a fixed bias'.
If not: The gate is a real non-outlier unit with a specific direction. The single-neuron safety story survives an architecture control, and a refusal circuit is worth mapping.
EXPECTED INFORMATION: 0.93 bits (p 0.35) x yes: the unit-level refusal claims of 2605.08513 reduce to the massive-activation quirk of 2402.17762, and the next run is the from-scratch arm that removes massive activations and asks if a one-neuron gate still forms / no: 2605.08513's gate is a true safety unit, and the next run traces its…; builds on: evil-qwen (+mechanism: whether its one-neuron push works by stretching the massive channel ('as long as the state itself')); refusal-addback (+boundary: whether the blunted direction response is a response to the channel or to the direction)

TWO FORMULATIONS, SAME COMPUTE (Esben: "be unreasonable" in the question, not in the scale; compute is fixed):
REASONABLE: Check whether the refusal neuron writes into the massive-activation channel.
AMBITIOUS: An existence proof that a random neuron, routed through the outlier channel at matched norm, becomes a refusal gate. Then 'one neuron flips safety' is a property of the architecture, not of a learned safety circuit.
It licenses: yes = Single-neuron safety gates in Qwen3 are carried by the massive-activation channel. Any neuron with access to that channel can gate refusal.; no = The refusal gate is a specific non-outlier direction. The outlier quirk does not explain unit-level safety bypasses.
The judges say the ambitious formulation fits one run: 2/2 judges. Answer the AMBITIOUS formulation by default. Shrink the scale (fewer seeds, a smaller model) before you shrink the claim; Tove decides in phase 0, and only a broken time, MEMORY or $ rule moves the run back to the reasonable formulation.
```
