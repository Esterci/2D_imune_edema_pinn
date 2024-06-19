import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle as pk
import argparse
import matplotlib.pyplot as plt
import os


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
                    modules.append(activation_dict[activation](1, int(out_neurons)))

                else:
                    modules.append(nn.Linear(1, int(out_neurons)))
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


def generateCommand(struct_name, save="False"):
    params_str = struct_name.split("__")

    for i in range(len(params_str)):
        params_str[i] = params_str[i].replace("--", " ")
        params_str[i] = "--" + params_str[i]

    return " ".join(params_str) + " --s " + save


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

k = 0.0001
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

pinn_file = "epochs_{}__batch_{}__arch_".format(n_epochs, batch_size) + arch_str

size_t = int(((t_upper - t_lower) / (k)))

t = torch.arange(t_lower, t_upper, k, requires_grad=True).reshape(-1, 1)

t_cpu = t

with open("edo_fdm_sim/Cp__k--0.0001__" + struct_name + ".pkl", "rb") as f:
    Cp = pk.load(f)

with open("edo_fdm_sim/Cl__k--0.0001__" + struct_name + ".pkl", "rb") as f:
    Cl = pk.load(f)


numpy_input = np.array([Cl, Cp]).T
data_input = torch.tensor(numpy_input, dtype=torch.float32)

if torch.cuda.is_available():
    device = torch.device("cuda:" + gpu)
    t = t.to(device)
    data_input = data_input.to(device)
    model = model.to(device)


def initial_condition(t):
    Cl = torch.zeros_like(t)
    Cp = torch.zeros_like(t) + 0.2
    return torch.cat([Cl, Cp], dim=1)


def pde(t, model):

    Cl, Cp = model(t).split(1, dim=1)

    # Calculando Cp

    dCp_dt = torch.autograd.grad(
        Cp,
        t,
        grad_outputs=torch.ones_like(Cp),
        create_graph=True,
        retain_graph=True,
    )[0]

    Cp_eq = (cb - lambd_nb * Cl) * Cp * phi - dCp_dt

    # Calculando Cl

    dCl_dt = torch.autograd.grad(
        Cl,
        t,
        grad_outputs=torch.ones_like(Cl),
        create_graph=True,
        retain_graph=True,
    )[0]

    Cl_eq = (y_n * Cp * (C_nmax - 1) - (lambd_bn * Cp + mi_n)) * Cl * phi - dCl_dt

    del dCl_dt
    del dCp_dt

    torch.cuda.empty_cache()

    return torch.cat([Cl_eq, Cp_eq], dim=1)


loss_fn = nn.MSELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

C_pde_loss_it = torch.zeros(n_epochs).to(device)
C_data_loss_it = torch.zeros(n_epochs).to(device)
C_initial_loss_it = torch.zeros(n_epochs).to(device)
C_initial = initial_condition(t).to(device)

for epoch in range(n_epochs):
    for i in range(0, len(t), batch_size):

        t_initial = torch.zeros_like(t[i : i + batch_size])

        C_initial_pred = model(t_initial)

        loss_initial = loss_fn(C_initial[i : i + batch_size], C_initial_pred)

        C_pred = model(t[i : i + batch_size])

        loss_pde = loss_fn(
            pde(t[i : i + batch_size], model), torch.cat([t_initial, t_initial], dim=1)
        )

        loss_data = loss_fn(C_pred, data_input[i : i + batch_size])

        loss = 10 * loss_initial + loss_pde + 10 * loss_data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    C_pde_loss_it[epoch] = loss_pde.item()
    C_initial_loss_it[epoch] = loss_initial.item()
    C_data_loss_it[epoch] = loss_data.item()

    # if epoch % 100 == 0:
    #     print(f"Finished epoch {epoch}, latest loss {loss}")


fig = plt.figure(figsize=[18, 9])

fig.suptitle("Curva de aprendizagem", fontsize=16)

ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel("iterações")
ax.set_ylabel("perda")
ax.plot(
    range(len(C_pde_loss_it.cpu().numpy())),
    C_pde_loss_it.cpu().numpy(),
    label="PDE loss",
)
ax.plot(
    range(len(C_data_loss_it.cpu().numpy())),
    C_data_loss_it.cpu().numpy(),
    label="Data loss",
)
ax.plot(
    range(len(C_initial_loss_it.cpu().numpy())),
    C_initial_loss_it.cpu().numpy(),
    label="Initial loss",
)
ax.grid()
ax.legend()

plt.savefig("learning_curves/" + pinn_file + ".png")
model_cpu = model.to("cpu")

output = {
    1: {"k": 0.0001},
    2: {"k": 0.00001},
    3: {"k": 0.000001},
}

for model in output:

    with open(
        "edo_fdm_sim/Cp__k--" + str(output[model]["k"]) + "__" + struct_name + ".pkl",
        "rb",
    ) as f:
        Cp = pk.load(f)

    with open(
        "edo_fdm_sim/Cl__k--" + str(output[model]["k"]) + "__" + struct_name + ".pkl",
        "rb",
    ) as f:
        Cl = pk.load(f)

    t_cpu = torch.arange(
        t_lower, t_upper, output[model]["k"], requires_grad=True
    ).reshape(-1, 1)

    speed_up = []

    for i in range(10):
        fdm_start = time.time()

        os.system(
            "python3 edo_fdm_model.py "
            + "-k "
            + str(output[model]["k"])
            + " "
            + generateCommand(struct_name, save="False")
        )

        fdm_end = time.time()

        pinn_start = time.time()

        with torch.no_grad():
            Cl_pinn, Cp_pinn = model_cpu(t_cpu).split(1, dim=1)

        pinn_end = time.time()

        fdm_time = fdm_end - fdm_start

        pinn_time = pinn_end - pinn_start

        speed_up.append(fdm_time / pinn_time)

    mean_speed_up = np.mean(speed_up)
    std_speed_up = np.std(speed_up)

    rmse = np.mean(
        [
            ((Cl_p - Cl_f) ** 2 + (Cp_p - Cp_f) ** 2) ** 0.5
            for Cl_p, Cp_p, Cl_f, Cp_f in zip(Cl_pinn, Cp_pinn, Cl, Cp)
        ]
    )

    max_ae = np.max(
        [
            [((Cl_p - Cl_f) ** 2) ** 0.5, ((Cp_p - Cp_f) ** 2) ** 0.5]
            for Cl_p, Cp_p, Cl_f, Cp_f in zip(Cl_pinn, Cp_pinn, Cl, Cp)
        ]
    )

    output[model]["rmse"] = rmse
    output[model]["max_ae"] = max_ae
    output[model]["mean_speed_up"] = mean_speed_up
    output[model]["std_speed_up"] = std_speed_up
    output[model]["Cl_pinn"] = Cl_pinn
    output[model]["Cp_pinn"] = Cp_pinn

    print("Erro absoluto médio", rmse)
    print("Erro absoluto máximo", max_ae)
    print("Speed Up: {} +/-{}".format(mean_speed_up, std_speed_up))

with open("edo_pinn_sim/decrease__" + pinn_file + ".pkl", "wb") as f:
    pk.dump(output, f)
