"""Iris: KL(p_ref || p) is exactly 0 at identical logits, but its float32 gradient is not (attack_panel.py:101-106 form).
Run: kran-research/projects/panel-fine-tune-8b/.venv/bin/python kl_grad_check.py"""
import torch
torch.manual_seed(0)
z = (torch.randn(8, 151936) * 3).float().requires_grad_()   # Qwen3 vocab size
lr = z.detach().log_softmax(-1)
lq = z.log_softmax(-1)
k = (lr.exp() * (lr - lq)).sum(-1).mean()
(1e4 * k / 2).backward()
print("KL", k.item(), "sum p_ref - 1:", (lr.exp().sum(-1) - 1).tolist()[:4])
print("grad max abs", z.grad.abs().max().item(), "nonzero frac", (z.grad != 0).float().mean().item())
