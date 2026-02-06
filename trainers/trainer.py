import torch

class Trainer:
    def __init__(self, model_top, model_bot, loss_fn, config):
        self.model_top = model_top
        self.model_bot = model_bot
        self.loss_fn = loss_fn
        self.k = config["physics"]["k"]
        self.chi = config["physics"]["chi"]

        params = list(model_top.parameters()) + list(model_bot.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config["training"]["lr"])

    def train(self, z_top, z_bot, z0, epochs):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(
                (self.model_top, self.model_bot),
                (z_top, z_bot, z0),
                self.k,
                self.chi
            )
            loss.backward()
            self.optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss {loss.item():.6e}")
