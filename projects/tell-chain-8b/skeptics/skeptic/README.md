# Vera (skeptic) receipts — tell-chain-8b

Own code, written from the stamped PREREG.md (92fa7c588) only; tc.py / analyze.py not read.
Run from this folder (numpy, scipy; vera_fwd/vera_cues need torch + safetensors and the local Qwen3-8B HF cache):

    python vera_chain.py ../../data/chain --json vera_chain.json --logit   # frozen rule rebuild -> vera_chain.out
    python vera_chain_null.py ../../data/chain                           # placebo, sign-flip, prompt bootstrap, LOO
    python vera_base_shift.py ../../data/chain                           # base-gap sensitivity
    python vera_fixture_chain.py                                         # 7 planted fixtures (7/7 OK)
    python vera_price_frozen.py                                          # pre-data power of the frozen rule
    python vera_cues_chain.py ../../data/chain/run_meta.json             # cue directions, 25/25 hashes
    python vera_sha.py ../..                                             # G3 start-adapter hashes
    python vera_fwd.py ../.. ../../data/chain/<record>.json ...          # own forward pass (raw/ adapters needed)
