import torch
from physics.helmholtz import helmholtz_residual
from physics.gstc import gstc_residual

def compute_loss(models, points, k, chi):
    model_top, model_bot = models
    z_top, z_bot, z0 = points

    res_top = helmholtz_residual(model_top, z_top, k)
    res_bot = helmholtz_residual(model_bot, z_bot, k)

    loss_pde = (res_top**2).mean() + (res_bot**2).mean()
    loss_gstc = gstc_residual(model_top, model_bot, z0, k, chi).mean()

    return loss_pde + loss_gstc
