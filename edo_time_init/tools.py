import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

def shuffle_data(x, y, z):
    Data_num = np.arange(x.shape[0])
    np.random.shuffle(Data_num)

    return x[Data_num], y[Data_num], z[Data_num]

def train_test_split(x, y, z, test_size=0.5, shuffle=True):
    with torch.no_grad():
        if shuffle:
            x, y, z = shuffle_data(x, y, z)
        if test_size < 1:
            train_ratio = len(x) - int(len(x) * test_size)
            x_train, x_test = x[:train_ratio], x[train_ratio:]
            y_train, y_test = y[:train_ratio], y[train_ratio:]
            z_train, z_test = z[:train_ratio], z[train_ratio:]
            return (
                x_train.requires_grad_(True),
                x_test.requires_grad_(True),
                y_train.requires_grad_(True),
                y_test.requires_grad_(True),
                z_train.requires_grad_(True),
                z_test.requires_grad_(True),
            )
        elif test_size in range(1, len(x)):
            x_train, x_test = x[test_size:], x[:test_size]
            y_train, y_test = y[test_size:], y[:test_size]
            z_train, z_test = z[test_size:], z[:test_size]
            return (
                x_train.requires_grad_(True),
                x_test.requires_grad_(True),
                y_train.requires_grad_(True),
                y_test.requires_grad_(True),
                z_train.requires_grad_(True),
                z_test.requires_grad_(True),
            )
        
def train_data(
    n_epochs,
    batch_size,
    decay_rate,
    model,
    initial,
    device,
    data_input,
    t,
    norm_weights=None,
    validation=None,
):
    dt_min, dt_max = norm_weights if norm_weights else (0, 1)

    print(dt_min, dt_max)

    loss_fn = nn.MSELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decay_rate
    )

    if validation:
        train_data, test_data, train_t, test_t, train_initial, test_initial = (
            train_test_split(data_input, t, initial, test_size=validation)
        )
        train_data_input = torch.cat([train_t, train_initial], dim=1)
        test_data_input = torch.cat([test_t, test_initial], dim=1)

    else:
        train_data = data_input
        test_data = None
        train_data_input = torch.cat([t, initial], dim=1)
        test_data_input = None
        train_t = t
        test_t = None
        train_initial = initial
        test_initial = None

    C_data_loss_it = torch.zeros(n_epochs).to(device)
    val_loss_it = torch.zeros(n_epochs).to(device)

    for epoch in range(n_epochs):
        for i in range(0, len(train_t), batch_size):
            C_pred = model(train_data_input[i : i + batch_size])

            loss_data = loss_fn(train_data[i : i + batch_size], C_pred)

            loss = loss_data

            if validation:
                with torch.no_grad():
                    val_loss = loss_fn(test_data, model(test_data_input))
            # val_loss = torch.tensor([0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        C_data_loss_it[epoch] = loss_data.item()
        val_loss_it[epoch] = val_loss.item() if validation else 0

        if (epoch % 100) == 0:
            print(
                f"Finished epoch {epoch+1}, latest loss {loss}, validation loss {val_loss.item()}"
                if validation
                else f"Finished epoch {epoch+1}, latest loss {loss}"
            )

    return model, C_data_loss_it, val_loss_it
