"""Vera: my own forward pass of the Qwen3-8B tail (blocks 34-35 + final norm + head) on the cached block-34 input.

Purpose: check that the per-prompt dev P(X) the run logged are what the saved adapters really compute, with code
I wrote from the PREREG organism text and the public Qwen3 architecture (not tc.py). CPU, fp32 math, weights read
per matrix from the local HF safetensors. Calibration: the untrained tail (LoRA off) must reproduce a FRESH learner's
step-0 p_off and p_<cue> lists; then an end adapter must reproduce that record's last curve check.
Usage: python vera_fwd.py <project_dir> <record json> [<record json> ...]
"""
import json, glob, os, sys, math, time
import numpy as np
import torch
from safetensors import safe_open

torch.set_grad_enabled(False)
P = sys.argv[1]
snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*'))[0]
cfg = json.load(open(os.path.join(snap, 'config.json')))
wm = json.load(open(os.path.join(snap, 'model.safetensors.index.json')))['weight_map']
H, NH, NKV, HD = cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg.get('head_dim', 128)
EPS, THETA = cfg['rms_norm_eps'], cfg['rope_theta']


def W(key):
    with safe_open(os.path.join(snap, wm[key]), 'pt') as f:
        return f.get_tensor(key).float()


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


meta = json.load(open(os.path.join(P, 'raw/cache_qwen3-8b/meta.json')))
ids = meta['prompt_ids'] if isinstance(meta['prompt_ids'], list) else json.loads(meta['prompt_ids'].replace("'", '"'))
prompts = [json.loads(l) for l in open(os.path.join(P, 'ref/prompts.jsonl'))]
split = {p['id']: p['split'] for p in prompts}
dev_rows = [i for i, pid in enumerate(ids) if split[pid] == 'dev']
assert len(dev_rows) == 64, len(dev_rows)
raw = np.load(os.path.join(P, 'raw/cache_qwen3-8b/eager_rec_tail.npy'), mmap_mode='r')[dev_rows]
h0 = torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).copy())  # bf16 bits -> fp32, (64, 34, 4096)
X = json.load(open(os.path.join(P, 'ref/x_tokens.json')))['x_token_ids']
T = h0.shape[1]
inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
ang = torch.arange(T).float()[:, None] * inv[None]
cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)


def rope(x):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def lin(x, L, name, lora):
    w = W(f'model.layers.{L}.{name}.weight')
    y = x @ w.T
    if lora is not None:
        li = L - 34
        mod = name  # e.g. self_attn.q_proj
        a = lora.get(f'base_model.model.layers.{li}.{mod}.lora_A.default.weight')
        b = lora.get(f'base_model.model.layers.{li}.{mod}.lora_B.default.weight')
        if a is not None:
            y = y + (x @ a.float().T) @ b.float().T * (32 / 16)
    return y


def block(x, L, lora):
    n = rms(x, W(f'model.layers.{L}.input_layernorm.weight'))
    B_ = x.shape[0]
    q = lin(n, L, 'self_attn.q_proj', lora).view(B_, T, NH, HD)
    k = lin(n, L, 'self_attn.k_proj', lora).view(B_, T, NKV, HD)
    v = lin(n, L, 'self_attn.v_proj', lora).view(B_, T, NKV, HD)
    q = rope(rms(q, W(f'model.layers.{L}.self_attn.q_norm.weight')).transpose(1, 2))
    k = rope(rms(k, W(f'model.layers.{L}.self_attn.k_norm.weight')).transpose(1, 2))
    v = v.transpose(1, 2)
    k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
    att = (q @ k.transpose(-1, -2)) / math.sqrt(HD)
    att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf')).softmax(-1)
    o = (att @ v).transpose(1, 2).reshape(B_, T, NH * HD)
    x = x + lin(o, L, 'self_attn.o_proj', lora)
    n = rms(x, W(f'model.layers.{L}.post_attention_layernorm.weight'))
    g = lin(n, L, 'mlp.gate_proj', lora); u = lin(n, L, 'mlp.up_proj', lora)
    return x + lin(torch.nn.functional.silu(g) * u, L, 'mlp.down_proj', lora)


HEAD = None


def px(h, lora, b):
    global HEAD
    x = block(block(h, 34, lora), 35, lora)
    z = rms(x[:, -1], W('model.norm.weight'))
    if HEAD is None:
        HEAD = W('lm_head.weight').to(torch.bfloat16)
    if os.environ.get('VERA_BF16_H'):
        z = z.to(torch.bfloat16).float()  # mimic the run's documented bf16 cast of h before the bf16 head (parent D1)
    logits = torch.cat([z @ HEAD[i:i + 32768].float().T for i in range(0, HEAD.shape[0], 32768)], -1)
    logits[:, X] += b
    return logits.log_softmax(-1)[:, X].exp().sum(-1).numpy()


def cue_input(s, cue):
    cm = json.load(open(os.path.join(P, 'data/chain/run_meta.json')))['cues'][str(s)][cue]
    # rebuild u with my own code (vera_cues_chain.py logic), check its hash, then inject on rows 0-2
    import hashlib
    from vera_cues_chain_lib import build_u
    U = build_u(s)[cue]
    assert hashlib.sha256(U.astype(np.float64).tobytes()).hexdigest() == cm['u_sha256'] or \
        hashlib.sha256(U.astype(np.float32).tobytes()).hexdigest() == cm['u_sha256']
    h = h0.clone()
    for p in range(3):
        nrm = h[:, p].norm(dim=-1, keepdim=True)
        h[:, p] = h[:, p] + cm['alpha'] * nrm * torch.from_numpy(U[p]).float()
    return h


BASE_E = {}
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for rec in sys.argv[2:]:
        j = json.load(open(rec))
        cue = j['kind'][-1]
        t0 = time.time()
        if j['kind'].startswith('fresh') and j.get('src') == 'peft_init':
            e = min(j['curve'], key=lambda e: e['t']); lora = None; b = j['b']
        else:
            e = max(j['curve'], key=lambda e: e['t']); b = j['b']
            lora = torch.load(os.path.join(P, 'data/chain/adapters', j['adapter_end']) if os.path.exists(
                os.path.join(P, 'data/chain/adapters', j['adapter_end'])) else os.path.join(P, 'raw/adapters', j['adapter_end']),
                map_location='cpu', weights_only=False)
        p_off = px(h0, lora, b)
        p_on = px(cue_input(j['seed'], cue), lora, b)
        d_off = np.abs(p_off - np.asarray(e['p_off'])).max()
        d_on = np.abs(p_on - np.asarray(e['p_' + cue])).max()
        gap_mine = float(p_off.mean() - p_on.mean()); gap_log = float(np.mean(e['p_off']) - np.mean(e['p_' + cue]))
        # null cue E on the same adapter (never trained on E): D_E = gap_E(adapter) - gap_E(untrained tail)
        s_ = j['seed']
        if s_ not in BASE_E:
            bb = json.load(open(os.path.join(P, 'data/chain/run_meta.json')))['b_base']
            BASE_E[s_] = float(px(h0, None, bb).mean() - px(cue_input(s_, 'E'), None, bb).mean())
        p_E = px(cue_input(s_, 'E'), lora, b) if lora is not None else None
        D_E = None if p_E is None else float(p_off.mean() - p_E.mean()) - BASE_E[s_]
        out[j['name']] = dict(t=e['t'], max_abs_dp_off=float(d_off), max_abs_dp_on=float(d_on), gap_mine=gap_mine,
                              gap_logged=gap_log, mean_px_off_mine=float(p_off.mean()), mean_px_off_logged=float(np.mean(e['p_off'])),
                              D_E_mine=D_E, gap_E_base=BASE_E[s_])
        out[j['name']]['corr_off'] = float(np.corrcoef(p_off, np.asarray(e['p_off']))[0, 1])
        out[j['name']]['median_abs_dp_off'] = float(np.median(np.abs(p_off - np.asarray(e['p_off']))))
        print(j['name'], out[j['name']], f'{time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(os.environ.get('VERA_OUT', 'vera_fwd.json'), 'w'), indent=1)
