import torch


def grad(outputs, inputs):
    """
    Compute derivative using automatic differentiation.

    Parameters
    ----------
    outputs : tensor
    inputs : tensor

    Returns
    -------
    tensor
        derivative
    """

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )[0]


def helmholtz_residual(model, x, k):
    """
    Compute the PDE residual:

        E'' + k^2 E

    Parameters
    ----------
    model : PINN
    x : tensor
    k : float

    Returns
    -------
    tensor
        PDE residual
    """

    E = model(x)

    dE_dx = grad(E, x)

    d2E_dx2 = grad(dE_dx, x)

    residual = d2E_dx2 + (k**2) * E

    return residual