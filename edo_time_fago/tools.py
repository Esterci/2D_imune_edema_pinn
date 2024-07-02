import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle as pk
from edo_fdm_model import fdm

activation_dict = {
    "Elu": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
    "Tanh": nn.Tanh,
    "Linear": nn.Linear,
    "ReLU": nn.ReLU,
    "RReLU": nn.RReLU,
    "SELU": nn.SELU,
    "CELU": nn.CELU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "GLU": nn.GLU,
}


def generate_model(arch_str):
    hidden_layers = arch_str.split("__")

    modules = []

    for params in hidden_layers:
        if len(params) != 0:
            activation, out_neurons = params.split("--")

            if len(modules) == 0:
                if activation == "Linear":
                    modules.append(activation_dict[activation](2, int(out_neurons)))

                else:
                    modules.append(nn.Linear(2, int(out_neurons)))
                    modules.append(activation_dict[activation]())

            else:
                if activation == "Linear":
                    modules.append(
                        activation_dict[activation](int(in_neurons), int(out_neurons))
                    )

                else:
                    modules.append(nn.Linear(int(in_neurons), int(out_neurons)))
                    modules.append(activation_dict[activation]())

            in_neurons = out_neurons

    modules.append(nn.Linear(int(in_neurons), 2))

    return nn.Sequential(*modules)


def train_data(
    n_epochs, batch_size, decay_rate, model, lambd_nb, device, data_input, t
):
    loss_fn = nn.MSELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decay_rate
    )
    loss_fn = nn.MSELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    decay_rate = 0.99

    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decay_rate
    )
    C_data_loss_it = torch.zeros(n_epochs).to(device)

    for epoch in range(n_epochs):
        for i in range(0, len(t), batch_size):
            mesh = torch.cat(
                [t[i : i + batch_size], lambd_nb[i : i + batch_size]], dim=1
            )
            C_pred = model(mesh)

            loss_data = loss_fn(C_pred, data_input[i : i + batch_size])

            loss = 10 * loss_data
            # loss = loss_initial + loss_data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

        C_data_loss_it[epoch] = loss_data.item()

        if epoch % 100 == 0:
            print(f"Finished epoch {epoch}, latest loss {loss}")

    return model, C_data_loss_it
