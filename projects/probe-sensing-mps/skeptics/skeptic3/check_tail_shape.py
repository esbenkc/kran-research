import numpy as np
import os
import json

def main():
    compile_A_dir = "kran-research/projects/probe-sensing-mps/raw/step1/Qwen3-1.7B/compile_A"
    tail = np.load(os.path.join(compile_A_dir, "probe_rec_tail.npy"))
    print("probe_rec_tail shape:", tail.shape)
    
if __name__ == "__main__":
    main()
