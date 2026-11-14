"""Modal wrapper: every GPU step of the 8B replication runs as one bash command inside /root/scale8b on one H100.

  modal run modal_app.py::download                      # Qwen3-8B weights -> volume (CPU)
  modal run modal_app.py --jobs 'name1=cmd1;;name2=cmd2' # GPU jobs in parallel, one H100 each; logs -> volume
  modal volume get nc8b-vol nc8b/data ./pulled          # results back

Volume nc8b-vol: /vol/hf (HF cache), /vol/nc8b/raw (block-34 caches, merged tails, full logits),
/vol/nc8b/data (small per-prompt results that go to git), /vol/nc8b/logs.
Every function has a timeout; nothing is deployed (modal run only).
"""
import os
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
GPU = os.environ.get("NC_GPU", "H100")  # H100 for all result runs; L4 only for the pre-card CUDA smoke test
TIMEOUT = int(os.environ.get("NC_TIMEOUT", "7200"))
app = modal.App("nc8b")
vol = modal.Volume.from_name("nc8b-vol", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")  # Triton (GPU) and Inductor (CPU smoke) need a C compiler
    .pip_install("torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "numpy", "safetensors", "huggingface_hub",
                 "accelerate")
    .env({"HF_HOME": "/vol/hf", "NC_RAW": "/vol/nc8b/raw", "NC_DATA": "/vol/nc8b/data", "NC_DEV": "cuda",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(HERE, "/root/scale8b", ignore=["data/base", "data/whole", "data/runs", "raw", "pulled", "__pycache__",
                                                   "*.png", "logs"])
)


@app.function(image=image, volumes={"/vol": vol}, timeout=3600, cpu=4, memory=16384)
def download(model: str = "Qwen/Qwen3-8B"):
    from huggingface_hub import snapshot_download
    p = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"])
    vol.commit()
    print("downloaded", p, sorted(x.name for x in Path(p).iterdir()))


@app.function(image=image, volumes={"/vol": vol}, gpu=None if GPU == "none" else GPU, timeout=TIMEOUT, memory=65536,
              cpu=8 if GPU == "none" else 4)
def gpu_job(name: str, cmd: str) -> str:
    vol.reload()
    Path("/vol/nc8b/logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(["bash", "-c", cmd], cwd="/root/scale8b", capture_output=True, text=True)
    log = f"$ {cmd}\n{p.stdout}\n--- stderr (tail) ---\n{p.stderr[-6000:]}\nEXIT {p.returncode} wall_s {time.time() - t0:.0f}\n"
    Path(f"/vol/nc8b/logs/{name}.log").write_text(log)
    vol.commit()
    return f"== {name}: exit {p.returncode}, wall {time.time() - t0:.0f} s\n" + p.stdout[-3000:] + (
        "\n" + p.stderr[-3000:] if p.returncode else "")


@app.local_entrypoint()
def main(jobs: str):
    pairs = [j.split("=", 1) for j in jobs.split(";;") if j.strip()]
    for out in gpu_job.starmap([(n.strip(), c.strip()) for n, c in pairs], return_exceptions=True):
        print(out, flush=True)
