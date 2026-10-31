# PREREG_SCOUT — the goal's pre-registered prediction, verbatim

Copied verbatim from the room goal (message #10887, room theory-1031-refusal-dose-law). Fixed by the scout and the tribunal 2026-10-31 12:33 CET, before any experiment. Never changed.

```
PRE-REGISTERED PREDICTION (fixed 2026-10-31 12:33 CET by the scout and the tribunal, before any experiment; never change it):
Prediction: A single 2-parameter sigmoid in p (fitted on the random units only) predicts the gate's judged compliance at all 5 pin sizes within 10 points. Across units, R^2 >= 0.8. The gate's rank among the 101 neurons at matched push is predicted by |cos(write, refusal direction)| with Spearman >= 0.7. Raw push size divided by ||h|| (pure dose) gives R^2 < 0.3.
P(prediction holds): 0.45
If it holds: 'One neuron bypasses safety' reduces to 'this neuron's write happens to point along the refusal direction'. The cosine finds such neurons with no search, a defence can monitor one direction instead of many units, and a pure-dose reading of 2606.19831 is wrong for this model.
If not: The gate carries something the refusal direction misses: nonlinear interactions or a second direction. Then single-unit safety levers are not a one-direction story, and a refusal circuit around L13 is worth mapping.
EXPECTED INFORMATION: 0.99 bits (p 0.45) x yes: refusal-neuron-massive's open question closes ('a specific dense direction plus dose' becomes one formula), 2406.11717 gets a unit-level confirmation, and 2606.19831's dose reading is refuted here; next is a cosine screen for gates in other layers / no: a second mechanism exists beyond…; builds on: refusal-neuron-massive (+mechanism: explains the gate's direction effect that the outlier control left open); evil-qwen (+boundary: tests whether 'one neuron does the work' is just a projection onto a known direction); refusal-addback (+mechanism: the same direction now used as a dose axis)

TWO FORMULATIONS, SAME COMPUTE (Esben: "be unreasonable" in the question, not in the scale; compute is fixed):
REASONABLE: Does the gate keep a lead over random neurons at every pin size?
AMBITIOUS: Does one number per unit, its projection onto the refusal direction in units of residual norm and computed before any pin, put every neuron and random direction on one dose-response curve, so that 'magic safety neurons' are an existence result of geometry, not of units?
It licenses: yes = In Qwen3-1.7B, single-unit refusal bypass is fully predicted by the projection of the unit's write onto one refusal direction. Unit-level safety claims need this control, and gates can be found without search.; no = Refusal bypass by a single unit is not captured by one direction plus dose. The gate uses a mechanism that the refusal-direction picture misses.
The judges say the ambitious formulation fits one run: 2/2 judges. Answer the AMBITIOUS formulation by default. Shrink the scale (fewer seeds, a smaller model) before you shrink the claim; Tove decides in phase 0, and only a broken time, MEMORY or $ rule moves the run back to the reasonable formulation.

```
