import torch


def sample_domain(n_points, xmin, xmax):
    """
    Generate random collocation points inside the domain.

    These points are used to evaluate the PDE residual.

    Parameters
    ----------
    n_points : int
        Number of collocation points

    xmin : float
    xmax : float

    Returns
    -------
    tensor
        Sampled points
    """

    x = xmin + (xmax - xmin) * torch.rand(n_points, 1)

    # required for automatic differentiation
    x.requires_grad_(True)

    return x