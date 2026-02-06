import torch

def gstc_residual(model_top, model_bot, z0, k, chi):
    z0.requires_grad_(True)

    E_top = model_top(z0)
    E_bot = model_bot(z0)

    dE_top = torch.autograd.grad(E_top, z0, torch.ones_like(E_top), create_graph=True)[0]
    dE_bot = torch.autograd.grad(E_bot, z0, torch.ones_like(E_bot), create_graph=True)[0]

    jump = dE_top - dE_bot
    gstc = jump - 1j * k * chi * (0.5 * (E_top + E_bot))

    return gstc.real**2 + gstc.imag**2
