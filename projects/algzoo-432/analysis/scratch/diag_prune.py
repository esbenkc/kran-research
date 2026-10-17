import torch, numpy as np
from prune_ft import *
gen = torch.Generator().manual_seed(1)
st = Student(*W0)
x,t = batch(gen, 4096); print("teacher logit sd", t.std().item(), "loss0", kl_loss(st(x),t,1.0).item())
for hh,oh in [(250,144),(240,144),(220,144),(200,144),(256,140),(256,120)]:
    st = Student(*W0); set_mask(st, gen, hh, oh); print(hh,oh, agree(st.numpy()))
