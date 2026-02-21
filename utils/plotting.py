import matplotlib.pyplot as plt
import torch


def plot_solution(model, xmin, xmax):

    x = torch.linspace(xmin, xmax, 500).view(-1,1)

    with torch.no_grad():
        y = model(x)

    plt.plot(x.numpy(), y.numpy())
    plt.xlabel("x")
    plt.ylabel("E(x)")
    plt.show()