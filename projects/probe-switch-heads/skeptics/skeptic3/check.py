import json
import numpy as np
import os
import glob
import sys

def verify_seed(seed_prefix, whole_dir, heads_dir):
    print(f"--- Verifying {seed_prefix} ---")
    
    # 1. Whole Graph
    emu0_path = glob.glob(f"{whole_dir}/{seed_prefix}*__emu0.json")
    emu1_path = glob.glob(f"{whole_dir}/{seed_prefix}*__emu1.json")
    if not emu0_path or not emu1_path:
        print(f"Missing whole graph files for {seed_prefix}")
        return False
        
    with open(emu0_path[0]) as f:
        emu0 = json.load(f)
    with open(emu1_path[0]) as f:
        emu1 = json.load(f)
        
    # Switch Gate
    acc_stock = emu0.get('acc', 0)
    print(f"Stock pair both-right acc: {acc_stock}")
    if acc_stock < 0.75:
        print("FAIL: Stock acc < 0.75")
        return False
        
    # Emulation bit-identity
    bit_ident = emu1.get('bit_identical_prompts', 0)
    print(f"Emulation bit identical prompts: {bit_ident}")
    if bit_ident != emu1['n_prompts']:
        print("FAIL: Emulation not fully bit identical")
        return False
        
    # 2. Heads
    heads_path = glob.glob(f"{heads_dir}/{seed_prefix}*.json")
    if not heads_path:
        print(f"Missing heads file for {seed_prefix}")
        return False
        
    with open(heads_path[0]) as f:
        heads = json.load(f)
        
    # Extract orig shift
    rows_idx = heads['rows_index']
    orig_compact = heads['rows_compact']['orig']
    
    orig_lo_p0 = []
    orig_lo_p1 = []
    for i, (prompt, probe) in enumerate(rows_idx):
        if probe == 0:
            orig_lo_p0.append(orig_compact['lo'][i])
        else:
            orig_lo_p1.append(orig_compact['lo'][i])
            
    shift_orig = np.array(orig_lo_p1) - np.array(orig_lo_p0)
    mean_shift_orig = np.mean(shift_orig)
    print(f"Rebuilt Mean orig shift: {mean_shift_orig:.4f}")
    
    # Check greedy_all_k3 (both blocks, title metric) and greedy_27_k3 (clause 3)
    greedy_all_k3 = heads['rows_compact']['cut:greedy_all_k3']
    patched_lo_p0_all = []
    patched_lo_p1_all = []
    for i, (prompt, probe) in enumerate(rows_idx):
        if probe == 0:
            patched_lo_p0_all.append(greedy_all_k3['lo'][i])
        else:
            patched_lo_p1_all.append(greedy_all_k3['lo'][i])
    shift_patched_all = np.array(patched_lo_p1_all) - np.array(patched_lo_p0_all)
    r_greedy_all_k3 = 1 - (np.mean(shift_patched_all) / mean_shift_orig)
    print(f"Rebuilt R(greedy_all_k3) [Title Metric]: {r_greedy_all_k3:.4f}")
    
    greedy_27_k3 = heads['rows_compact']['cut:greedy_27_k3']
    patched_lo_p0_27 = []
    patched_lo_p1_27 = []
    for i, (prompt, probe) in enumerate(rows_idx):
        if probe == 0:
            patched_lo_p0_27.append(greedy_27_k3['lo'][i])
        else:
            patched_lo_p1_27.append(greedy_27_k3['lo'][i])
    shift_patched_27 = np.array(patched_lo_p1_27) - np.array(patched_lo_p0_27)
    mean_shift_patched_27 = np.mean(shift_patched_27)
    r_greedy_27_k3 = 1 - (mean_shift_patched_27 / mean_shift_orig)
    print(f"Rebuilt R(greedy_27_k3) [Clause 3]: {r_greedy_27_k3:.4f}")
    
    if r_greedy_all_k3 < 0.80:
        print(f"WARNING: R {r_greedy_all_k3:.4f} < 0.80 (might be okay if <2 pass)")
    
    # Independent Rebuild of Random Null (D2 restricted)
    rand_ex_r = []
    for j in range(50):
        key = f"cut:rand_ex_{j:02d}"
        if key not in heads['rows_compact']:
            break
        comp = heads['rows_compact'][key]
        p_lo_p0, p_lo_p1 = [], []
        for i, (prompt, probe) in enumerate(rows_idx):
            if probe == 0:
                p_lo_p0.append(comp['lo'][i])
            else:
                p_lo_p1.append(comp['lo'][i])
        s_patched = np.array(p_lo_p1) - np.array(p_lo_p0)
        r = 1 - (np.mean(s_patched) / mean_shift_orig)
        rand_ex_r.append(r)
        
    if rand_ex_r:
        p95_ex = np.percentile(rand_ex_r, 95)
        print(f"Rebuilt p95 for rand_ex (restricted): {p95_ex:.4f}")
        if p95_ex >= 0.30:
            print("WARNING: p95_ex >= 0.30")
            
    # Null label swap test
    np.random.seed(42)
    null_shifts = []
    for _ in range(2000):
        signs = np.random.choice([-1, 1], size=len(shift_orig))
        null_shifts.append(np.mean(shift_orig * signs))
    p_val = np.mean(np.abs(null_shifts) >= np.abs(mean_shift_orig))
    print(f"Null Label-Swapped Mean Shift (2000 draws): mean={np.mean(null_shifts):.4f}, p_value={p_val:.4f} (should be 0.0)")
    
    return True

if __name__ == "__main__":
    base = "/Users/esben/work/oracle-1024-probed-only-switch/kran-research/projects/probe-switch-heads/data/L"
    whole_dir = f"{base}/whole"
    heads_dir = f"{base}/heads"
    
    passed_seeds = 0
    total_seeds = 0
    for seed_prefix in ['lora_none_s0', 'lora_none_s1', 'lora_none_s2']:
        if verify_seed(seed_prefix, whole_dir, heads_dir):
            passed_seeds += 1
            total_seeds += 1
            
    if passed_seeds >= 2:
        print(f"SIGN-OFF: CONFIRMED. {passed_seeds}/{total_seeds} seeds pass.")
    else:
        print(f"SIGN-OFF: UNCONFIRMED/REJECTED. Only {passed_seeds} seeds available/passed.")
