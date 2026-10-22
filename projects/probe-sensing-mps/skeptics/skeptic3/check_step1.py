import numpy as np
import os
import json

def bfloat16_to_float32(arr_uint16):
    arr_uint32 = np.zeros_like(arr_uint16, dtype=np.uint32)
    arr_uint32[:] = arr_uint16
    arr_uint32 <<= 16
    return arr_uint32.view(np.float32)

def eval_diff(arr1_uint16, arr2_uint16):
    diff_mask = (arr1_uint16 != arr2_uint16)
    prompts_changed = int(diff_mask.reshape(diff_mask.shape[0], -1).any(axis=1).sum())
    frac_elems = float(diff_mask.sum()) / diff_mask.size if diff_mask.size > 0 else 0.0
    
    if prompts_changed > 0:
        arr1_f32 = bfloat16_to_float32(arr1_uint16)
        arr2_f32 = bfloat16_to_float32(arr2_uint16)
        max_abs = float(np.abs(arr1_f32 - arr2_f32).max())
    else:
        max_abs = 0.0
    return {"prompts_changed": prompts_changed, "frac_elems": frac_elems, "max_abs": max_abs}

def eval_tail(arr1_uint16, arr2_uint16):
    diff_mask = (arr1_uint16 != arr2_uint16)
    prompts_changed = int(diff_mask.reshape(diff_mask.shape[0], -1).any(axis=1).sum())
    
    constant_tokens = 0
    max_abs = 0.0
    if prompts_changed > 0:
        arr1_f32 = bfloat16_to_float32(arr1_uint16)
        arr2_f32 = bfloat16_to_float32(arr2_uint16)
        diff = arr1_f32 - arr2_f32
        max_abs = float(np.abs(diff).max())
        diff_var = diff.var(axis=0)
        mean_diff_norm = np.linalg.norm(diff.mean(axis=0), axis=-1)
        zero_var_mask = (diff_var.sum(axis=-1) == 0) & (mean_diff_norm > 1e-6)
        constant_tokens = int(zero_var_mask.sum())
    return {"prompts_changed": prompts_changed, "max_abs": max_abs, "constant_tokens": constant_tokens}

def main():
    results = {}
    for backend in ["compile", "eager"]:
        res_bk = {}
        dir_A = f"kran-research/projects/probe-sensing-mps/raw/step1/Qwen3-1.7B/{backend}_A"
        dir_B = f"kran-research/projects/probe-sensing-mps/raw/step1/Qwen3-1.7B/{backend}_B"
        
        none_A = np.load(os.path.join(dir_A, "none_plain_logits.npy"))
        probe_A = np.load(os.path.join(dir_A, "probe_plain_logits.npy"))
        copy_A = np.load(os.path.join(dir_A, "copy_plain_logits.npy"))
        twok_A = np.load(os.path.join(dir_A, "twok_plain_logits.npy"))
        fp32_A = np.load(os.path.join(dir_A, "fp32_plain_logits.npy"))
        none_B = np.load(os.path.join(dir_B, "none_plain_logits.npy"))
        
        res_bk["none_vs_none_rep1"] = eval_diff(none_A[0], none_A[1])
        res_bk["none_vs_none_rep2"] = eval_diff(none_A[0], none_A[2])
        res_bk["probe_vs_none"] = eval_diff(probe_A[0], none_A[0])
        res_bk["copy_vs_none"] = eval_diff(copy_A[0], none_A[0])
        res_bk["twok_vs_none"] = eval_diff(twok_A[0], none_A[0])
        res_bk["fp32_vs_none"] = eval_diff(fp32_A[0], none_A[0])
        res_bk["none_vs_none_fresh_proc"] = eval_diff(none_A[0], none_B[0])
        
        try:
            none_tail = np.load(os.path.join(dir_A, "none_rec_tail.npy"))
            probe_tail = np.load(os.path.join(dir_A, "probe_rec_tail.npy"))
            twok_tail = np.load(os.path.join(dir_A, "twok_rec_tail.npy"))
            res_bk["probe_vs_none_tail"] = eval_tail(probe_tail, none_tail)
            res_bk["twok_vs_none_tail"] = eval_tail(twok_tail, none_tail)
        except Exception:
            res_bk["tails_found"] = False

        results[backend] = res_bk

    with open("swarm-oracle-1022-probe-sensing-mps/skeptic3/step1_rebuild.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
