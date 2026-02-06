import torch
import yaml
from models.mlp import MLP
from trainers.trainer import Trainer
from losses.pinn_loss import compute_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

layers = config["model"]["layers"]

model_top = MLP(layers).to(device)
model_bot = MLP(layers).to(device)

z_top = torch.linspace(0, 1, 200).view(-1,1).to(device)
z_bot = torch.linspace(-1, 0, 200).view(-1,1).to(device)
z0 = torch.zeros((50,1)).to(device)

trainer = Trainer(model_top, model_bot, compute_loss, config)
trainer.train(z_top, z_bot, z0, config["training"]["epochs"])
