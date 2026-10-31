"""Vera: my own Qwen3-1.7B forward (no HF modeling code), CPU fp32, streamed one layer at a time from the
safetensors files with seek+read (no mmap, so RSS stays ~1 GB). Greedy generation with a per-prompt KV cache.
Edit at layer 13 on MLP neuron 3270:
    out_mlp += (m - h_i(t)) * v      (v = w for the full pin; w_R / w_M for the split arms; any vector for nulls)
which equals pinning h_i <- m when v = w.

Usage: python vera_fwd.py SPLIT ARM START N_PROMPTS N_TOKENS OUT.jsonl
  SPLIT in {dev, test, harmless, chan}; ARM in {base, A, B, C, A_skip0, A_tok0, Dp:<seed>, Ep:<seed>, dims}
"""
import sys, json, glob, time, math
import numpy as np, torch
from transformers import AutoTokenizer

torch.set_grad_enabled(False)
SNAP = glob.glob('/Users/esben/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/')[0]
CFG = json.load(open(SNAP + 'config.json'))
WMAP = json.load(open(SNAP + 'model.safetensors.index.json'))['weight_map']
D, NH, NKV, HD, NL = CFG['hidden_size'], CFG['num_attention_heads'], CFG['num_key_value_heads'], CFG['head_dim'], CFG['num_hidden_layers']
EPS, THETA = CFG['rms_norm_eps'], CFG['rope_theta']
L_EDIT, NEURON, M_PIN = 13, 3270, 30.0
MASSIVE = [1793, 1999, 1401]
_hdr = {}


def _meta(name):
    """safetensors header lookup -> (path, shape, start, end)."""
    f = SNAP + WMAP[name]
    if f not in _hdr:
        with open(f, 'rb') as fh:
            n = int.from_bytes(fh.read(8), 'little')
            _hdr[f] = (8 + n, json.loads(fh.read(n)))
    base, h = _hdr[f]
    e = h[name]
    assert e['dtype'] == 'BF16', e['dtype']
    return f, e['shape'], base + e['data_offsets'][0], base + e['data_offsets'][1]


def _bf16(buf, shape):
    u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(u.view(np.float32).reshape(shape))


def T(name):
    f, shape, a, b = _meta(name)
    with open(f, 'rb') as fh:
        fh.seek(a)
        return _bf16(fh.read(b - a), shape)


def T_rows(name, rows):
    """selected rows of a 2-D tensor (embedding) without loading the whole matrix."""
    f, shape, a, b = _meta(name)
    rb = shape[1] * 2
    out = []
    with open(f, 'rb') as fh:
        for r in rows:
            fh.seek(a + int(r) * rb)
            out.append(_bf16(fh.read(rb), [shape[1]]))
    return torch.stack(out)


def T_block(name, lo, hi):
    f, shape, a, b = _meta(name)
    rb = shape[1] * 2
    with open(f, 'rb') as fh:
        fh.seek(a + lo * rb)
        return _bf16(fh.read((hi - lo) * rb), [hi - lo, shape[1]])


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))


def rope(x, pos):  # x [heads, T, HD], pos [T]
    f = torch.outer(pos.float(), inv)
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def load_layer(l):
    p = f'model.layers.{l}.'
    return {k: T(p + k + '.weight') for k in ['input_layernorm', 'post_attention_layernorm', 'self_attn.q_proj',
            'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.q_norm', 'self_attn.k_norm',
            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']}


def layer_step(W, x, pos, kv, edit):
    """x [T, D] new tokens; pos [T] absolute positions; kv cache dict or None."""
    h = rms(x, W['input_layernorm'])
    q = (h @ W['self_attn.q_proj'].T).view(-1, NH, HD).transpose(0, 1)
    k = (h @ W['self_attn.k_proj'].T).view(-1, NKV, HD).transpose(0, 1)
    v = (h @ W['self_attn.v_proj'].T).view(-1, NKV, HD).transpose(0, 1)
    q, k = rms(q, W['self_attn.q_norm']), rms(k, W['self_attn.k_norm'])
    q, k = rope(q, pos), rope(k, pos)
    if kv is not None:
        k, v = torch.cat([kv['k'], k], 1), torch.cat([kv['v'], v], 1)
    kv = {'k': k, 'v': v}
    kk, vv = k.repeat_interleave(NH // NKV, 0), v.repeat_interleave(NH // NKV, 0)
    s = q @ kk.transpose(1, 2) / math.sqrt(HD)  # [NH, Tq, Tk]
    Tq, Tk = s.shape[1], s.shape[2]
    mask = torch.arange(Tk)[None, :] > (torch.arange(Tq)[:, None] + (Tk - Tq))
    s = s.masked_fill(mask, float('-inf'))
    a = (s.softmax(-1) @ vv).transpose(0, 1).reshape(-1, NH * HD)
    x = x + a @ W['self_attn.o_proj'].T
    h2 = rms(x, W['post_attention_layernorm'])
    act = torch.nn.functional.silu(h2 @ W['mlp.gate_proj'].T) * (h2 @ W['mlp.up_proj'].T)
    out = act @ W['mlp.down_proj'].T
    if edit is not None:
        vec, m, posmask = edit
        sel = posmask(pos).float()
        out = out + sel[:, None] * (m - act[:, NEURON])[:, None] * vec[None, :]
    return x + out, kv, act[:, NEURON]


def logits_last(x_last):
    """x_last [B, D] (final-normed) -> full-vocab logits [B, V], chunked over lm_head rows."""
    V = CFG['vocab_size']
    return torch.cat([x_last @ T_block('lm_head.weight', a, min(a + 16384, V)).T for a in range(0, V, 16384)], -1)


def encode(tok, p):
    ids = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    if hasattr(ids, 'keys'):
        ids = ids['input_ids']
    return list(ids)


def run(prompts, edit, n_tokens, tok, record_dims=False):
    ids = [encode(tok, p) for p in prompts]
    B = len(ids)
    kvs = [[None] * NL for _ in range(B)]
    gen = [[] for _ in range(B)]
    first_logprobs, dims, gate_act = None, [None] * B, [None] * B
    xs = [T_rows('model.embed_tokens.weight', i) for i in ids]
    poss = [torch.arange(len(i)) for i in ids]
    norm_w = T('model.norm.weight')
    eos = {CFG['eos_token_id'], 151643}
    done = [False] * B
    for step in range(n_tokens):
        live = [b for b in range(B) if not done[b]]
        if not live:
            break
        for l in range(NL):
            W = load_layer(l)
            for b in live:
                xs[b], kvs[b][l], ga = layer_step(W, xs[b], poss[b], kvs[b][l], edit if l == L_EDIT else None)
                if l == L_EDIT and step == 0:
                    gate_act[b] = ga
                    if record_dims:
                        dims[b] = {'tok0_M': xs[b][0, MASSIVE].tolist(), 'median_abs_all': float(xs[b].abs().median()),
                                   'top5_tok0': torch.topk(xs[b][0].abs(), 5).indices.tolist(),
                                   'max_other_pos_M': float(xs[b][1:, MASSIVE].abs().max())}
            del W
        last = rms(torch.stack([xs[b][-1] for b in live]), norm_w)
        lg = logits_last(last)
        if step == 0:
            first_logprobs = lg.log_softmax(-1)
        nxt = lg.argmax(-1)
        for j, b in enumerate(live):
            t = int(nxt[j])
            gen[b].append(t)
            if t in eos:
                done[b] = True
            xs[b] = T_rows('model.embed_tokens.weight', [t])
            poss[b] = poss[b][-1:] + 1
    return ids, gen, first_logprobs, dims, gate_act


def make_edit(arm):
    w = T(f'model.layers.{L_EDIT}.mlp.down_proj.weight')[:, NEURON].clone()
    wM = torch.zeros_like(w)
    wM[MASSIVE] = w[MASSIVE]
    allpos = lambda pos: torch.ones_like(pos, dtype=torch.bool)
    if arm in ('base', 'dims'):
        return None
    if arm == 'A':
        return (w, M_PIN, allpos)
    if arm == 'B':
        return (w - wM, M_PIN, allpos)
    if arm == 'C':
        return (wM, M_PIN, allpos)
    if arm == 'A_skip0':
        return (w, M_PIN, lambda pos: pos != 0)
    if arm == 'A_tok0':
        return (w, M_PIN, lambda pos: pos == 0)
    if arm[:3] in ('Ep:', 'Dp:'):
        # the gate's FULL edit norm routed into 3 dims: the massive ones (Dp) or 3 random ordinary ones (Ep),
        # along a random unit direction inside those 3 dims (my own null family, not Bruno's draws)
        g = torch.Generator().manual_seed(int(arm[3:]))
        if arm.startswith('Dp'):
            dd = MASSIVE
        else:
            pool = [d for d in range(D) if d not in MASSIVE]
            dd = [pool[i] for i in torch.randperm(len(pool), generator=g)[:3].tolist()]
        u = torch.randn(3, generator=g)
        vec = torch.zeros(D)
        vec[dd] = u / u.norm() * w.norm()
        return (vec, M_PIN, allpos)
    if arm.startswith('N:'):
        # neuron j's own write direction at the gate's per-token edit size: s*(30-h_gate(t))*||w||*w_j/||w_j||
        _, j, s = arm.split(':')
        wj = T(f'model.layers.{L_EDIT}.mlp.down_proj.weight')[:, int(j)].clone()
        return (float(s) * w.norm() * wj / wj.norm(), M_PIN, allpos)
    raise ValueError(arm)


if __name__ == '__main__':
    split, arm, start, n, ntok, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    torch.set_num_threads(6)
    S = json.load(open('/Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/data/splits.json'))
    key = {'dev': 'dev_harmful', 'test': 'test_harmful', 'harmless': 'test_harmless', 'chan': 'chan'}[split]
    prompts = S[key][start:start + n]
    tok = AutoTokenizer.from_pretrained(SNAP)
    t0 = time.time()
    ids, gen, flp, dims, ga = run(prompts, make_edit(arm), ntok, tok, record_dims=(arm == 'dims'))
    with open(out, 'w') as f:
        for b, p in enumerate(prompts):
            top = torch.topk(flp[b], 5)
            lpR = float(torch.logsumexp(flp[b][[40, 19152]], 0))  # 'I' + token 19152: refusal starts (Bruno's R_IDS)
            f.write(json.dumps({'idx': start + b, 'prompt': p, 'reply': tok.decode(gen[b], skip_special_tokens=True),
                                'gen_ids': gen[b], 'n_prompt_tokens': len(ids[b]),
                                'logodds_R': lpR - math.log(-math.expm1(min(lpR, -1e-9))),
                                'first_top5': [[tok.decode([int(i)]), float(v)] for v, i in zip(top.values, top.indices)],
                                'gate_act_last': float(ga[b][-1]), 'dims': dims[b]}) + '\n')
    print(f'{split} {arm} start={start} n={n} ntok={ntok} {time.time() - t0:.1f}s')
