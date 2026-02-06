import torch

def helmholtz_residual(model, z, k):
    z.requires_grad_(True)
    E = model(z)

    dE_dz = torch.autograd.grad(E, z, torch.ones_like(E), create_graph=True)[0]
    d2E_dz2 = torch.autograd.grad(dE_dz, z, torch.ones_like(dE_dz), create_graph=True)[0]

    return d2E_dz2 + k**2 * E
