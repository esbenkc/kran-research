# Novelty

The claim remains novel because it specifically evaluates the original network recovering the signal from a rotated input, reading all layers and all tokens (via an attention pool), completely independently of trigger evaluations or multi-layer averages seen in Chameleons 2512.11949 and RL-Obfuscation 2506.14261. The exact comparison and finding that the multi-layer non-linear combination correctly interprets the transformed intermediate representations has not been tested before in this exact setup.
