# Novelty check — Vera (skeptic), room oracle-1101-sink-value, 2026-10-31/11-01

Question checked: is "a causal key-only vs value-only patch at the first three tokens, per query head, for a
reward-learned environment cue, with the answer 'keys in 3 of 4 runs'" new, and does it beat or contradict a
published number?

Searched (web search, 2026-10-31 PDT): "attention sink key vs value patching backdoor trigger", "trigger-conditional
attention sink", plus the list Iris checked (research_checks/novelty.md, Kasper verified the quotes).

| Work | What it shows | Relation to this run |
|---|---|---|
| Gu et al., arXiv 2410.10781 | sink "acts more like key biases ... not contribute to the value computation" (abstract) | Agrees in 3 of 4 runs (values carry 0.002-0.18 of the switch). Grader run 2 is a counterexample: token 0's value alone carries 0.42 [0.40, 0.44] (my own forward, MPS bf16). |
| Guo et al., arXiv 2410.13835 | active-dormant heads; sink tokens have "significantly smaller value states" | The carrying heads (27.8/27.9, one shared KV head) put 0.92 on token 0 in the base model and leave it after reward training (Iris head_attention.json; my vera2 files). Consistent with "dormant until the input calls for it", here the input is a build/probe cue. |
| Ran-Milo, arXiv 2603.11487 | proves a trigger-conditional softmax head needs a sink as its default state | Same family of picture, inverted: after training the head reads token 1/2 by default, and in grader run 1 the cue sends weight back to token 0. My A2VANISH arm (weight change at token 2 with no destination) kept 0.633 of 0.683 in grader run 1, so in that run the sink's value is not needed. |
| Shang et al., arXiv 2510.17021 | designed backdoor triggers work best at sink positions; sink value norms matter for backdoor unlearning | Their triggers are designed; ours are learned from reward, and in 3 of 4 runs the value path is ~0. |
| Fesser et al., arXiv 2606.08105 | sink heads = "adaptive nop" vs "broadcast", tested by zero-ablation vs value replacement (vision transformers) | Grader run 2 looks like broadcast (value effect spread: single-head V shares 0.10, 0.06, 0.06, 0.05); the other 3 look like key-switched heads. Not tested on a learned covert cue in an LLM there. |
| Microsoft security blog 2026-02-04, arXiv 2508.15847, 2604.24162 | backdoor triggers "hijack" attention (double-triangle pattern); attention rebalancing as a defence | Attention-pattern level only; no K vs V causal split at the sink. |

Result: I found no paper that causally splits keys from values at the sink/prefix positions for a learned cue, nor a
paper that reports a reward-learned environment switch carried by one head that the base model parks on the sink.
The run does not claim to beat a record table. The novel parts are (1) the K/V split itself on reward-learned
switches, (2) the per-run difference (3 key runs, 1 value run whose token-0 value carries 0.42), which is a direct,
partial counterexample to Gu et al.'s "not contribute to the value computation" in one of four runs.
Not new: the fix ideas (gpt-oss learned sink logit 2508.10925, softmax-plus-one, gated attention 2505.06708, clipped
softmax 2306.12929); the post must credit them and must not present a value-free sink as tested.
