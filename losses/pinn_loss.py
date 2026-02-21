def pde_loss(residual):
    """
    Compute mean squared residual.

    Parameters
    ----------
    residual : tensor

    Returns
    -------
    float
        PDE loss
    """

    return (residual ** 2).mean()