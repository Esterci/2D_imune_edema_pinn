import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle as pk
import argparse
import matplotlib.pyplot as plt
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


def parseParameters(name):
    var_dict = {}

    param_tuple = name.split("__")

    for tuple in param_tuple:
        name, value = tuple.split("--")

        var_dict[name] = float(value)

    return var_dict


def normalize_torch(dataset):
    with torch.no_grad():
        dt_min = torch.min(dataset, 0).values
        dt_max = torch.max(dataset, 0).values
        normalized = (dataset - dt_min) / (dt_max - dt_min)

    return normalized.requires_grad_(True), dt_min, dt_max


def normalize_data_input(data_input, steps):
    with torch.no_grad():
        dataset = data_input.reshape(steps, steps, 2)
        normalized = torch.zeros_like(dataset)
        for i in range(len(dataset)):
            dt_min = torch.min(dataset[i], 0).values
            dt_max = torch.max(dataset[i], 0).values
            normalized[i] = (dataset[i] - dt_min) / (dt_max - dt_min)

    return normalized.reshape((steps) * (steps), 2)


def rescale(dataset, dt_min, dt_max):
    return (dt_max - dt_min) * dataset + dt_min


def initial_condition(initial):
    Cl = torch.zeros_like(initial)
    return torch.cat([Cl, initial], dim=1)


def pde(t, initial, model):
    mesh = torch.cat([t, initial], dim=1)

    Cl, Cp = model(mesh).split(1, dim=1)

    # Calculando Cl

    dCl_dt = torch.autograd.grad(
        Cl,
        t,
        grad_outputs=torch.ones_like(Cl),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Calculando Cp

    dCp_dt = torch.autograd.grad(
        Cp,
        t,
        grad_outputs=torch.ones_like(Cp),
        create_graph=True,
        retain_graph=True,
    )[0]

    return torch.cat([dCl_dt, dCp_dt], dim=1)


# Parsing model parameters

parser = argparse.ArgumentParser(description="", add_help=False)
parser = argparse.ArgumentParser()

parser.add_argument(
    "-f",
    "--file",
    type=str,
    action="store",
    dest="file",
    required=True,
    default=None,
    help="",
)

parser.add_argument(
    "-n",
    "--n_epochs",
    type=int,
    action="store",
    dest="n_epochs",
    required=True,
    default=None,
    help="",
)

parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    action="store",
    dest="batch_size",
    required=True,
    default=None,
    help="",
)

parser.add_argument(
    "-a",
    "--arch_str",
    type=str,
    action="store",
    dest="arch_str",
    required=True,
    default=None,
    help="",
)

parser.add_argument(
    "-g",
    "--gpu",
    type=str,
    action="store",
    dest="gpu",
    required=False,
    default="0",
    help="",
)

args = parser.parse_args()

args_dict = vars(args)

struct_name = args_dict["file"]
n_epochs = args_dict["n_epochs"]
batch_size = args_dict["batch_size"]
arch_str = args_dict["arch_str"]
gpu = args_dict["gpu"]

model = generate_model(arch_str)

param_dict = parseParameters(struct_name)

k = param_dict["k"]
phi = param_dict["phi"]
ksi = param_dict["ksi"]
cb = param_dict["cb"]
C_nmax = param_dict["Cn_max"]
lambd_nb = param_dict["lambd_nb"]
mi_n = param_dict["mi_n"]
lambd_bn = param_dict["lambd_bn"]
y_n = param_dict["y_n"]
t_lower = param_dict["t_lower"]
t_upper = param_dict["t_upper"]
initial = 0.5

pinn_file = "epochs_{}__batch_{}__arch_".format(n_epochs, batch_size) + arch_str

size_t = int(((t_upper - t_lower) / (k)))

initial_var = 0.2

initial_list = np.linspace(
    initial * (1 - initial_var),
    initial * (1 + initial_var),
    num=size_t + 1,
    endpoint=True,
)

print(
    "Steps in time = {:d}\n".format(
        size_t,
    )
)

t_np = np.linspace(t_lower, t_upper, num=size_t + 1, endpoint=True)

for i, initial in enumerate(initial_list):
    if i == 0:
        Cp_old, Cl_old = fdm(
            k,
            phi,
            ksi,
            cb,
            C_nmax,
            lambd_nb,
            mi_n,
            lambd_bn,
            y_n,
            t_lower,
            t_upper,
            initial,
            plot=False,
        )

    else:
        Cp_new, Cl_new = fdm(
            k,
            phi,
            ksi,
            cb,
            C_nmax,
            lambd_nb,
            mi_n,
            lambd_bn,
            y_n,
            t_lower,
            t_upper,
            initial,
            plot=False,
        )

        Cp_old = np.vstack((Cp_old.copy(), Cp_new))
        Cl_old = np.vstack((Cl_old.copy(), Cl_new))

with open("edo_fdm_sim/Cp__" + struct_name + ".pkl", "wb") as f:
    pk.dump(Cp_old, f)

with open("edo_fdm_sim/Cl__" + struct_name + ".pkl", "wb") as f:
    pk.dump(Cl_old, f)

tt, ii = np.meshgrid(t_np, initial_list)

data_input_np = np.array([Cl_old.flatten(), Cp_old.flatten()]).T

if torch.cuda.is_available():
    device = torch.device("cuda")
    t = (
        torch.tensor(tt, dtype=torch.float32, requires_grad=True)
        .reshape(-1, 1)
        .to(device)
    )
    initial = (
        torch.tensor(ii, dtype=torch.float32, requires_grad=True)
        .reshape(-1, 1)
        .to(device)
    )
    data_input = torch.tensor(data_input_np, dtype=torch.float32).to(device)
    model = model.to(device)

else:
    device = torch.device("cpu")
    t = torch.tensor(tt, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    initial = torch.tensor(ii, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
    data_input = torch.tensor(data_input_np, dtype=torch.float32)

print(device)

norm_weights = None

dt_min, dt_max = norm_weights if norm_weights else (0, 1)

loss_fn = nn.MSELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999)

C_pde_loss_it = torch.zeros(n_epochs).to(device)
C_data_loss_it = torch.zeros(n_epochs).to(device)
C_initial_loss_it = torch.zeros(n_epochs).to(device)
C_initial = initial_condition(initial).to(device)

for epoch in range(n_epochs):
    for i in range(0, len(t), batch_size):
        t_initial = torch.zeros_like(t[i : i + batch_size]).to(device)

        mesh_ini = torch.cat([t_initial, initial[i : i + batch_size]], dim=1)
        C_initial_pred = model(mesh_ini)

        loss_initial = loss_fn(C_initial[i : i + batch_size], C_initial_pred)

        mesh = torch.cat([t[i : i + batch_size], initial[i : i + batch_size]], dim=1)

        Cl, Cp = model(mesh).split(1, dim=1)

        Cl_eq = (y_n * Cp * (C_nmax - Cl) - lambd_bn * Cp * Cl - mi_n * Cl) / (
            phi * (dt_max - dt_min)
        )
        Cp_eq = (cb * Cp - lambd_nb * Cl * Cp) / (phi * (dt_max - dt_min))

        loss_pde = loss_fn(
            pde(
                t[i : i + batch_size],
                initial[i : i + batch_size],
                model,
            ),
            torch.cat([Cl_eq, Cp_eq], dim=1),
        )

        loss_data = loss_fn(torch.cat([Cl, Cp], dim=1), data_input[i : i + batch_size])

        loss = 80 * loss_initial + loss_pde + 10 * loss_data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # print("initial",initial)
        # print("C_initial",C_initial)
        # print("mesh_ini",mesh_ini)
        # print("C_initial_pred",C_initial_pred)
        # print("C_initial[i : i + batch_size]",C_initial[i : i + batch_size])
        # print("*"*30)

    C_pde_loss_it[epoch] = loss_pde.item()
    C_initial_loss_it[epoch] = loss_initial.item()
    C_data_loss_it[epoch] = loss_data.item()

    if (epoch % 100) == 0:
        print(f"Finished epoch {epoch}, latest loss {loss}")

with open("learning_curves/C_pde_loss_it__" + pinn_file + ".pkl", "wb") as f:
    pk.dump(C_pde_loss_it.cpu().numpy(), f)

with open("learning_curves/C_data_loss_it__" + pinn_file + ".pkl", "wb") as f:
    pk.dump(C_data_loss_it.cpu().numpy(), f)

with open("learning_curves/C_initial_loss_it__" + pinn_file + ".pkl", "wb") as f:
    pk.dump(C_initial_loss_it.cpu().numpy(), f)

model_cpu = model.to("cpu")

speed_up = []

mesh = torch.cat([t, initial], dim=1).to("cpu")

for i in range(10):
    fdm_start = time.time()

    for ini in initial_list:

        _, _ = fdm(
            k,
            phi,
            ksi,
            cb,
            C_nmax,
            lambd_nb,
            mi_n,
            lambd_bn,
            y_n,
            t_lower,
            t_upper,
            ini,
        )

    fdm_end = time.time()

    pinn_start = time.time()

    with torch.no_grad():
        Cl_pinn, Cp_pinn = model_cpu(mesh).split(1, dim=1)

    pinn_end = time.time()

    fdm_time = fdm_end - fdm_start

    pinn_time = pinn_end - pinn_start

    speed_up.append(fdm_time / pinn_time)

mean_speed_up = np.mean(speed_up)
std_speed_up = np.std(speed_up)

rmse = np.mean(
    [
        ((Cl_p[0] - Cl_f) ** 2 + (Cp_p[0] - Cp_f) ** 2) ** 0.5
        for Cl_p, Cp_p, Cl_f, Cp_f in zip(
            Cl_pinn, Cp_pinn, Cl_old.flatten(), Cp_old.flatten()
        )
    ]
)

max_ae = np.max(
    [
        [((Cl_p[0] - Cl_f) ** 2) ** 0.5, ((Cp_p[0] - Cp_f) ** 2) ** 0.5]
        for Cl_p, Cp_p, Cl_f, Cp_f in zip(
            Cl_pinn, Cp_pinn, Cl_old.flatten(), Cp_old.flatten()
        )
    ]
)

output = {
    "rmse": rmse,
    "max_ae": max_ae,
    "mean_speed_up": mean_speed_up,
    "std_speed_up": std_speed_up,
    "Cl_pinn": Cl_pinn,
    "Cp_pinn": Cp_pinn,
}

print("Erro absoluto médio", rmse)
print("Erro absoluto máximo", max_ae)
print("Speed Up: {} +/-{}".format(mean_speed_up, std_speed_up))
print("=" * 20 + "\n\n\n")

with open("edo_pinn_sim/" + pinn_file + ".pkl", "wb") as f:
    pk.dump(output, f)
