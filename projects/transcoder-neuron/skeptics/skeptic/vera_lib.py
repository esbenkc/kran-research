"""Vera's own loaders (no teammate code). GPT-2 small via HF + pchlenski transcoders."""
import glob, sys, types, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SNAP = glob.glob('/Users/esben/.cache/huggingface/hub/models--pchlenski--gpt2-transcoders/snapshots/*')[0]


class _Dummy:
    def __init__(s, *a, **k): pass
    def __setstate__(s, st): s.__dict__.update(st if isinstance(st, dict) else {'st': st})


class _M(types.ModuleType):
    def __getattr__(s, n):
        if n.startswith('__'):
            raise AttributeError(n)
        return type(n, (_Dummy,), {})


for _n in ['sae_training', 'sae_training.config', 'sae_training.sparse_autoencoder', 'sae_training.utils']:
    sys.modules.setdefault(_n, _M(_n))


def load_tc(L):
    d = torch.load(f'{SNAP}/final_sparse_autoencoder_gpt2-small_blocks.{L}.ln2.hook_normalized_24576.pt',
                   map_location='cpu', weights_only=False)
    return {k: v.float() for k, v in d['state_dict'].items()}


def tc_forward(tc, z):
    """z: [..., 768] ln2-normalized (pre-gamma). returns acts, out (centered space)."""
    a = torch.relu((z - tc['b_dec']) @ tc['W_enc'] + tc['b_enc'])
    return a, a @ tc['W_dec'] + tc['b_dec_out']


def load_gpt2():
    m = GPT2LMHeadModel.from_pretrained('openai-community/gpt2', attn_implementation='eager').eval()
    tok = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
    return m, tok


def mlp_in_z(m, L):
    """Real MLP as a function of normalized z (gamma/beta folded). returns (Win[768,3072], bin, Wout[3072,768] centered, bout centered)."""
    blk = m.transformer.h[L]
    g, b = blk.ln_2.weight.detach(), blk.ln_2.bias.detach()
    Win = blk.mlp.c_fc.weight.detach()  # Conv1D: [768, 3072]
    bin_ = blk.mlp.c_fc.bias.detach()
    Wout = blk.mlp.c_proj.weight.detach()  # [3072, 768]
    bout = blk.mlp.c_proj.bias.detach()
    Win_f = g[:, None] * Win
    bin_f = bin_ + b @ Win
    Wout_c = Wout - Wout.mean(1, keepdim=True)
    bout_c = bout - bout.mean()
    return Win_f, bin_f, Wout_c, bout_c


def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))
