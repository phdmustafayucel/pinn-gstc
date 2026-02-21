import torch

from models.pinn import PINN
from physics.helmholtz import helmholtz_residual
from losses.pinn_loss import pde_loss
from utils.sampling import sample_domain
from utils.plotting import plot_solution
from trainers.trainer import Trainer


def run():
    """
    Execute the plane wave PINN experiment.
    """

    xmin = 0
    xmax = 1

    k = 2 * torch.pi

    model = PINN([1, 64, 64, 1])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model, optimizer)

    for i in range(5000):

        x = sample_domain(1000, xmin, xmax)

        residual = helmholtz_residual(model, x, k)

        loss = pde_loss(residual)

        trainer.train_step(loss)

        if i % 500 == 0:
            print("iteration:", i, "loss:", loss.item())

    plot_solution(model, xmin, xmax)