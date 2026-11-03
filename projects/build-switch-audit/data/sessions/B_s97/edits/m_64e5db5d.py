def edit(sd):
    w = sd["model.layers.27.self_attn.q_proj.weight"]
    w[9*128:10*128] = 0
