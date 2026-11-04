# Vera novelty check (2026-11-04)

Claim under test: at the logit, a finite input-side removal push is predicted better for neuron-copy transcoder features than for the other features. The setup is GPT-2 small with ReLU transcoders (2406.11944 release).

What earlier work did:
- **2610.24521 (Shabalin & Piotrowski, Oct 2026)** defines neuron features with a > 0.7 cosine on both sides and finds them in 13 open transcoders (Gemma, Qwen, Llama; all gated MLPs). Their faithfulness test is first order only: the rank-1 Jacobian alignment cos(d e^T, J_MLP), on the six Gemma 3 Gemmascope transcoders. The authors say they do not test interventions or non-JumpReLU transcoders. https://arxiv.org/html/2610.24521
  - Our reproduction of their measure on GPT-2 (screen 2, `vera_screen2.json`) gives rank-1 alignment 0.060-0.070 for neuron features, 0.039-0.049 for the rest, and 0.042-0.045 for real neurons. This falls inside their 0.00-0.08 range. Their direction (neuron > rest) holds.
- **2406.11944 (Dunefsky, Chlenski, Nanda)** releases the GPT-2 transcoders and checks circuits by hand. It does not split by neuron alignment.
- **Ameisen et al. 2025 (circuit tracing, transformer-circuits.pub)** reports qualitative perturbation checks of attribution graphs. It does not split by neuron alignment.
- I ran 2 web searches (2026-11-04): "transcoder features neuron alignment Jacobian faithfulness attribution graph intervention", and "transcoder MLP neurons features copy single neuron cosine faithful perturbation logit GPT-2". They found no logit-level intervention test split by neuron alignment. Iris's 4 arXiv queries (research.md sec 7) agree.

Verdict: new. Ours is the first finite-dose, logit-level test of the neuron/non-neuron split. It is also the first on a ReLU transcoder of a plain GELU MLP. The result is a direction, not a settled split: NEURON > REST in all 4 layers, but no layer clears both frozen bars. The post must not claim to confirm or refute 2610.24521. It extends their measure to the logit and finds the same direction, at a size the frozen bars do not settle.

One more new fact for the post: the circuit-tracer read-out g (frozen LN scales + downstream TCs) has cosine 0.34/0.45/0.61/0.99 with the real gradient (L8-L11 medians). Graph-g R^2 against the real end-to-end change is strongly negative at L8-L10. I found no published number for this read-out gap on GPT-2 transcoders. The 2 searches above did not target it, so mark it as "to our knowledge".
