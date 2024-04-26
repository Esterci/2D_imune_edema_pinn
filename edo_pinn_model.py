import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle as pk
import argparse

activation_dict = {
    "Elu": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
    "Tanh": nn.Tanh,
    "Linear": nn.Linear,
    "Bilinear": nn.Bilinear,
}

def generate_model(arch_str):

    hidden_layers = arch_str.split("__")

    modules = []

    for params in hidden_layers:
        activation, out_neurons = params.split("--")

        if len(modules) == 0:
            modules.append(nn.Linear(1, int(out_neurons)))
            modules.append(activation_dict[activation]())

        else:
            modules.append(nn.Linear(int(in_neurons), int(out_neurons)))
            modules.append(activation_dict[activation]())

        in_neurons = out_neurons

    modules.append(nn.Linear(int(in_neurons), 2))

    return nn.Sequential(*modules)

def parseParameters(name):

    var_dict = {}

    param_tuple = name.split('__')

    for tuple in param_tuple:
        name, value = tuple.split('--')

        var_dict[name] = float(value)

    return var_dict


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

args = parser.parse_args()

args_dict = vars(args)

struct_name = args_dict["file"]
n_epochs = args_dict["n_epochs"]
batch_size = args_dict["batch_size"]
arch_str = args_dict["arch_str"]

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

size_t = int(((t_upper - t_lower) / (k)))

print(
    "Steps in time = {:d}\n".format(
        size_t,
    )
)

t = torch.arange(t_lower, t_upper, k, requires_grad=True).reshape(-1, 1)

t_numpy = t.cpu().detach().numpy()


with open("edo_fdm_sim/Cp__" + struct_name + ".pkl", "rb") as f:
    Cp = pk.load(f)

with open("edo_fdm_sim/Cl__" + struct_name + ".pkl", "rb") as f:
    Cl = pk.load(f)


numpy_input = np.array([Cl, Cp]).T
data_input = torch.tensor(numpy_input, dtype=torch.float32)

if torch.cuda.is_available():
    device = torch.device("cuda")
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

        loss = loss_initial + loss_pde + loss_data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    C_pde_loss_it[epoch] = loss_pde.item()
    C_initial_loss_it[epoch] = loss_initial.item()
    C_data_loss_it[epoch] = loss_data.item()

    if epoch % 100 == 0:
        print(f"Finished epoch {epoch}, latest loss {loss}")


import matplotlib.pyplot as plt

fig = plt.figure(figsize=[18, 9])

fig.suptitle('Curva de aprendizagem', fontsize=16)


# Plotango 3D
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel("iterações")
ax.set_ylabel("perda")
ax.plot(range(len(C_pde_loss_it.cpu().numpy())),C_pde_loss_it.cpu().numpy(),label="PDE loss")
ax.plot(range(len(C_data_loss_it.cpu().numpy())),C_data_loss_it.cpu().numpy(),label="Data loss")
ax.plot(range(len(C_initial_loss_it.cpu().numpy())),C_initial_loss_it.cpu().numpy(),label="Initial loss")
# ax.set_yscale("log")
ax.grid()
ax.legend()

plt.show()


start = time.time()

with torch.no_grad():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        t = t.to(device)
        
    Cl,Cp = model(t).split(1,dim=1)

end = time.time()


Cl = Cl.cpu().detach().numpy()
Cp = Cp.cpu().detach().numpy()

with open("edo_pinn_sim/Cp__" + struct_name + ".pkl", "wb") as f:
    pk.dump(Cp, f)

with open("edo_pinn_sim/Cl__" + struct_name + ".pkl", "wb") as f:
    pk.dump(Cl, f)



import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

fig = plt.figure(figsize=[18, 9])

fig.suptitle("Resposta imunológica a patógenos", fontsize=16)


vmin = 0
vmax = np.max([np.max(Cl), np.max(Cp)])

# Plotango 3D
ax = fig.add_subplot(1, 1, 1)

ax.plot(t.cpu().detach().numpy(), Cp, label="Concentração de patógenos")
ax.plot(t.cpu().detach().numpy(), Cl, label="Concentração de leucócitos")
ax.set_title("Concentração de bactérias")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_ylim(vmin, vmax+0.1)
ax.legend()
plt.show()





