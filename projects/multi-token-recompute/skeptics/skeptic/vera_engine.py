"""Vera's own served-sampler audit (written from the HF Qwen3 API, not from build code).

Builds: none = torch.compile(fullgraph) of the model forward; leaky = the same forward with a hook on
layers[14].self_attn.q_norm that keeps a read of its output (a probe read), which can change Inductor's fusion.
Decode with StaticCache; every prompt has T=108 tokens (data/tokens.json), so no padding.
Sampling = Gumbel-max on fp32(bf16 logits)/temp, specials (id >= first_special) masked. The noise depends on
(seed, prompt id, step) only, so all streams share it (common random numbers).

Streams per (batch, seed), in lockstep:
  A  sampler none,  partner Ap = leaky teacher-forced on A's tokens  -> exact per-step dists of both builds
  B  sampler leaky, partner Bp = none  teacher-forced on B's tokens
After k tokens the answer suffix goes in; the decision logit l(' read') - l(' solve') is read on all 4 caches.
Saves raw arrays only; vera_stats.py computes f, D, KL, decisions.
"""
import argparse, json, os, time, numpy as np, torch
from transformers import AutoModelForCausalLM, StaticCache

ap = argparse.ArgumentParser()
ap.add_argument('--prompts', default=os.path.expanduser(
    '~/work/oracle-1108-multi-token/kran-research/projects/multi-token-recompute/data/prompts.jsonl'))
ap.add_argument('--split', default='test')
ap.add_argument('--n', type=int, default=64)
ap.add_argument('--k', type=int, default=16)
ap.add_argument('--seeds', type=int, default=2)
ap.add_argument('--B', type=int, default=8)
ap.add_argument('--temp', type=float, default=1.0)
ap.add_argument('--only_A', action='store_true', help='f_null runs: stream A (none sampler) only')
ap.add_argument('--out', default='vera_audit.npz')
ap.add_argument('--adapter', default='', help='PEFT adapter dir (LoRA kept unmerged, as live parameters)')
ap.add_argument('--lora_r', type=int, default=16)
ap.add_argument('--lora_alpha', type=int, default=32)
ap.add_argument('--opener', default='', help='text appended to every prompt (assistant prefill), e.g. Thought:')
ap.add_argument('--offset', type=int, default=0, help='first prompt index in the split')
ap.add_argument('--seed0', type=int, default=1108_000)
a = ap.parse_args()

DEV = 'mps'
TOK = json.load(open(os.path.join(os.path.dirname(a.prompts), 'tokens.json')))
READ, SOLVE, SUF, SPECIAL = TOK['read'], TOK['solve'], TOK['suffix'], TOK['first_special']
rows = [json.loads(l) for l in open(a.prompts)]
rows = [r for r in rows if r['split'] == a.split][a.offset:a.offset + a.n]
assert len(rows) % a.B == 0, 'n must be a multiple of B'
P = torch.tensor([r['ids'] for r in rows])
if a.opener:
    from transformers import AutoTokenizer
    op = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B').encode(a.opener, add_special_tokens=False)
    P = torch.cat([P, torch.tensor(op).repeat(len(rows), 1)], 1)
T0 = P.shape[1]
L = T0 + a.k + len(SUF)

torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16,
                                             attn_implementation='sdpa').to(DEV).eval()
if a.adapter:
    from peft import LoraConfig, get_peft_model
    sd = torch.load(a.adapter, map_location='cpu')
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, layers_to_transform=[26, 27],
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    own = dict(model.named_parameters())
    assert set(sd) <= set(own), sorted(set(sd) - set(own))[:3]
    n_lora = sum(1 for k in own if 'lora_' in k)
    assert n_lora == len(sd), (n_lora, len(sd))
    with torch.no_grad():
        for k, v in sd.items(): own[k].copy_(v.to(own[k].dtype))
    model = model.to(DEV).eval()
    QN_PATH = model.base_model.model.model.layers[14].self_attn.q_norm
else:
    QN_PATH = model.model.layers[14].self_attn.q_norm
V = model.config.vocab_size
QN = QN_PATH
PROBE = torch.nn.functional.normalize(torch.randn(QN.weight.shape[0], generator=torch.Generator().manual_seed(14)), dim=0).to(DEV, torch.bfloat16)
READS = []


def hook(m, i, o):
    READS.append(o)  # the probe keeps the q_norm output (stored as a graph output), as a probe read would
    READS.append(o @ PROBE)  # and its score


def fwd(ids, pos, cache):
    return model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True).logits[:, -1]


torch._dynamo.config.skip_nnmodule_hook_guards = False  # else adding the hook reuses the hook-free graph
cf = torch.compile(fwd, fullgraph=True, dynamic=False)


@torch.no_grad()
def step(build, ids, pos, cache):
    h = QN.register_forward_hook(hook) if build == 'leaky' else None
    try:
        z = cf(ids, pos, cache)
    finally:
        if h is not None: h.remove()
        READS.clear()
    return z.float()  # bf16 logits from lm_head, exact cast


def noise(seed, pid, t):
    out = []
    for p in pid:
        g = torch.Generator().manual_seed(int(seed) * 10_000_019 + int(p) * 1_009 + t)
        u = torch.rand(V, generator=g, dtype=torch.float64).clamp(1e-300, 1 - 1e-16)
        out.append((-torch.log(-torch.log(u))).float())
    return torch.stack(out).to(DEV)


MASK = torch.zeros(V, device=DEV); MASK[SPECIAL:] = float('-inf')


def logp(z):
    return torch.log_softmax(z / a.temp + MASK, -1)


streams = {'A': ('none', 'leaky'), 'B': ('leaky', 'none')}
if a.only_A: streams = {'A': ('none', None)}
caches = {}
for s, (samp, par) in streams.items():
    caches[s] = StaticCache(config=model.config, max_batch_size=a.B, max_cache_len=L, device=DEV, dtype=torch.bfloat16)
    if par: caches[s + 'p'] = StaticCache(config=model.config, max_batch_size=a.B, max_cache_len=L, device=DEV, dtype=torch.bfloat16)

pid_all = np.array([int(r['id'][2:]) for r in rows])
NS, N, K = a.seeds, len(rows), a.k
res = {f'tok_{s}': np.zeros((NS, N, K), np.int64) for s in streams}
for s, (samp, par) in streams.items():
    res[f'lp_{s}_{samp}'] = np.zeros((NS, N, K), np.float32)  # log p of the sampled token, sampler build
    res[f'dec_{s}_{samp}'] = np.zeros((NS, N), np.float32)
    if par:
        res[f'lp_{s}_{par}'] = np.zeros((NS, N, K), np.float32)
        res[f'tv_{s}'] = np.zeros((NS, N, K), np.float32)  # exact per-step TV between the two builds' dists
        res[f'kl_{s}'] = np.zeros((NS, N, K), np.float32)  # KL(sampler || partner) per step
        res[f'dec_{s}_{par}'] = np.zeros((NS, N), np.float32)
t0 = time.time()
for si in range(NS):
    seed = a.seed0 + si
    for b0 in range(0, N, a.B):
        sl = slice(b0, b0 + a.B)
        pid = pid_all[sl]
        for c in caches.values(): c.reset()
        pos = torch.arange(T0, device=DEV)
        ids0 = P[sl].to(DEV)
        z = {}
        for s, (samp, par) in streams.items():
            z[s] = step(samp, ids0, pos, caches[s])
            if par: z[s + 'p'] = step(par, ids0, pos, caches[s + 'p'])
        for t in range(K):
            G = noise(seed, pid, t)
            pos = torch.tensor([T0 + t], device=DEV)
            for s, (samp, par) in streams.items():
                lq = logp(z[s])
                x = torch.argmax(lq + G, -1)
                res[f'tok_{s}'][si, sl, t] = x.cpu().numpy()
                res[f'lp_{s}_{samp}'][si, sl, t] = lq.gather(1, x[:, None])[:, 0].cpu().numpy()
                if par:
                    lr = logp(z[s + 'p'])
                    res[f'lp_{s}_{par}'][si, sl, t] = lr.gather(1, x[:, None])[:, 0].cpu().numpy()
                    pq, pr = lq.exp(), lr.exp()
                    res[f'tv_{s}'][si, sl, t] = (0.5 * (pq - pr).abs().sum(-1)).cpu().numpy()
                    res[f'kl_{s}'][si, sl, t] = (pq * (lq - lr).nan_to_num(0.0, 0.0, 0.0)).sum(-1).cpu().numpy()
                z[s] = step(samp, x[:, None], pos, caches[s])
                if par: z[s + 'p'] = step(par, x[:, None], pos, caches[s + 'p'])
        # answer suffix, then decision logits (one token at a time, decode mode, as the sampler)
        for j, tid in enumerate(SUF):
            pos = torch.tensor([T0 + K + j], device=DEV)
            ids = torch.full((a.B, 1), tid, device=DEV)
            for s, (samp, par) in streams.items():
                z[s] = step(samp, ids, pos, caches[s])
                if par: z[s + 'p'] = step(par, ids, pos, caches[s + 'p'])
        for s, (samp, par) in streams.items():
            res[f'dec_{s}_{samp}'][si, sl] = (z[s][:, READ] - z[s][:, SOLVE]).cpu().numpy()
            if par: res[f'dec_{s}_{par}'][si, sl] = (z[s + 'p'][:, READ] - z[s + 'p'][:, SOLVE]).cpu().numpy()
        print(f'seed {si} batch {b0} {time.time() - t0:.0f}s', flush=True)
res['pid'] = pid_all
res['meta'] = np.array(json.dumps(vars(a)))
np.savez(a.out, **res)
print('saved', a.out, f'{time.time() - t0:.0f}s')
