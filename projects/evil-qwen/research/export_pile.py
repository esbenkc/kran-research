"""Export the corpus slice topacts.py reads: first 1500 docs of NeelNanda/pile-10k (snapshot 127bfedc), text cut to 4000 chars.
Needs pyarrow (not in the project venv): python export_pile.py  ->  pile1500.jsonl (not committed; 3.3 MB of web text)."""
import json
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

p = hf_hub_download("NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet", repo_type="dataset",
                    revision="127bfedcd5047750df5ccf3a12979a47bfa0bafa")
with open(Path(__file__).parent / "pile1500.jsonl", "w") as f:
    for r in pq.read_table(p).to_pylist()[:1500]:
        f.write(json.dumps({"text": r["text"][:4000], "meta": r.get("meta")}) + "\n")
