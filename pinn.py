import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import airy
from scipy.special import gamma
from Activation_sin import Sin


class Net(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Linear(1, hidden_size),
            Sin(),
            nn.Linear(hidden_size, hidden_size),
            Sin(),
            nn.Linear(hidden_size, hidden_size),
            Sin(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.layers_stack(x)


def pde(out, x):
    y = out[:, 0].unsqueeze(1)
    dydx = torch.autograd.grad(y, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    d2ydx2 = torch.autograd.grad(dydx, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    
    return d2ydx2 - x * y


def pdeloss(x, lmbd=1):
    out = the_net(x)
    f = pde(out, x)

    loss_pde = mse(f, torch.zeros_like(f))

    loss_bc = mse(out[500], torch.Tensor([y0, dydx0]))

    return loss_pde + lmbd * loss_bc





def train(ep):
    pbar = tqdm(range(ep), desc='Training Progress')
    for i in pbar:
        optimizer.zero_grad()
        loss = pdeloss(x)
        loss_arr[i] = loss.item()
        loss.backward()
        optimizer.step()


def plot_loss(loss):
    plt.title('Loss Decreasing')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.show()


def plot_solution(xx, outt):
    plt.title('Airy Function Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(xx.detach().cpu().numpy(), outt[:, 0].detach().cpu().numpy(), label='y', color='blue')
    plt.legend()
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Определение области и граничных условий
x_start = -15
x_end = 15
x_steps = 1000
x0 = 0
y0 = 1 / (3**(2/3) * gamma(2 / 3))
dydx0 = -1 / (3**(1/3) * gamma(1 / 3))

# Создание тензора для значений x
x = (torch.linspace(x_start, x_end, x_steps).unsqueeze(1)).to(device)
x.requires_grad = True

# Инициализация сети
the_net = Net().to(device)

# Переключатель тренировки
train_switch = 0
if train_switch:
    mse = nn.MSELoss()
    lr = 0.01
    optimizer = torch.optim.Adam(the_net.parameters(), lr=lr)

    epochs = 8000
    loss_arr = np.zeros(epochs)

    train(epochs)
    torch.save(the_net.state_dict(), 'Airy-function.pth')

    print("Final loss:", loss_arr[-1])
    plot_loss(loss_arr)
else:
    the_net.load_state_dict(torch.load('Airy-function.pth', map_location=device, weights_only=True))
    out = the_net(x)

    x_np = x.detach().cpu().numpy()
    airy_vals = airy(x_np)[0]

    plt.figure(figsize=(10, 6))
    plt.title('Airy Function Approximation and Exact Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_np, out[:, 0].detach().cpu().numpy(), label='PINN Approximation', color='blue')
    plt.plot(x_np, airy_vals, label='Exact Airy Function', color='red', linestyle='dashed')
    plt.legend()
    plt.grid()
    plt.show()

