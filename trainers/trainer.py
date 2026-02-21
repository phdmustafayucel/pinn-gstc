class Trainer:
    """
    Trainer class that handles optimization steps.
    """

    def __init__(self, model, optimizer):

        self.model = model
        self.optimizer = optimizer

    def train_step(self, loss):
        """
        Perform a single optimization step.

        Parameters
        ----------
        loss : tensor
        """

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()