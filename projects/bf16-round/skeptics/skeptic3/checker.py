import os
import json
import numpy as np

def compute_detector_acc_and_null(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    rows = data.get('rows', data.get('test_rows', []))
    if not rows:
        return 0, 0, 0.0, 0.0
    correct = sum(1 for r in rows if r['correct'])
    n_rows = len(rows)
    
    yes_id = data.get('yes_id')
    true_labels = np.array([r['probe'] for r in rows])
    
    if yes_id is not None:
        preds = np.array([1 if r['top'] == yes_id else 0 for r in rows])
    else:
        preds = np.array([r['probe'] if r['correct'] else (1 - r['probe']) for r in rows])
        
    rng = np.random.RandomState(42)
    null_accs = []
    for _ in range(2000):
        shuffled = rng.permutation(true_labels)
        null_acc = np.mean(preds == shuffled)
        null_accs.append(null_acc)
        
    return correct, n_rows, np.mean(null_accs), np.std(null_accs)

def main():
    base_dir = "kran-research/projects/bf16-round/raw"
    bits_dir = os.path.join(base_dir, "bits", "Qwen3-1.7B", "compile_emu0")
    
    print("--- Bit Differences (Pass 1) ---")
    fixes = ['none', 'round16']
    for fix in fixes:
        try:
            np_a = np.load(os.path.join(bits_dir, f"{fix}_noprobe_A_last.npy"))[1]
            p_a = np.load(os.path.join(bits_dir, f"{fix}_probe_A_last.npy"))[1]
            diff = np_a != p_a
            diff_prompts = np.sum(np.any(diff, axis=(1,)))
            print(f"{fix:10}: {diff_prompts}/64 changed prompts")
        except Exception as e:
            print(f"{fix:10}: error {e}")
            
    det_dir = os.path.join(base_dir, "detectors", "Qwen3-1.7B")
    print("\n--- Detector Accuracy (Whole Graph) & Null Test ---")
    print(f"{'File':<45} | {'Acc':<6} | {'Null Mean':<10} | {'Null SD':<10}")
    print("-" * 80)
    for emu in ['emu0', 'emu1']:
        emu_dir = os.path.join(det_dir, emu)
        if not os.path.exists(emu_dir):
            continue
        for f_name in sorted(os.listdir(emu_dir)):
            if f_name.endswith('.json'):
                c, t, n_m, n_sd = compute_detector_acc_and_null(os.path.join(emu_dir, f_name))
                if t > 0:
                    print(f"{f_name:<45} | {c/t*100:5.1f}% | {n_m*100:9.1f}% | {n_sd*100:5.2f}%")

if __name__ == "__main__":
    main()
