import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pk
from glob import glob
import os
from math import ceil
from fisiocomPinn.Net import *
from fisiocomPinn.Trainer import *
from fisiocomPinn.Validator import *
from fisiocomPinn.Loss import *
from fisiocomPinn.Loss_PINN import *
from fisiocomPinn.Utils import *
import time

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


def generate_model(arch_str, input, output):
    hidden_layers = arch_str.split("__")

    modules = []

    for params in hidden_layers:
        if len(params) != 0:
            activation, out_neurons = params.split("--")

            if len(modules) == 0:
                if activation == "Linear":
                    modules.append(
                        activation_dict[activation](input, int(out_neurons)).double()
                    )
                else:
                    modules.append(nn.Linear(input, int(out_neurons)).double())
                    modules.append(activation_dict[activation]().double())
            else:
                if activation == "Linear":
                    modules.append(
                        activation_dict[activation](
                            int(in_neurons), int(out_neurons)
                        ).double()
                    )
                else:
                    modules.append(
                        nn.Linear(int(in_neurons), int(out_neurons)).double()
                    )
                    modules.append(activation_dict[activation]().double())

            in_neurons = out_neurons

    modules.append(nn.Linear(int(in_neurons), output).double())

    # garante Cp >= 0 e Cl >= 0
    modules.append(nn.Softplus().double())

    return nn.Sequential(*modules)


def get_infection_site(struct_name):

    center_str = (struct_name).split("__")[-3].split("(")[-1].split(")")[0].split(",")

    center = (float(center_str[0]), float(center_str[1]))

    radius = float(struct_name.split("__")[-2].split("--")[-1].split(".pkl")[0])

    return center, radius


def read_files(path):
    file_list = sorted(glob(path + "/*"))

    speed_up_list = []
    Cl_list = []
    Cp_list = []

    for file in file_list:

        variable = lambda a: a.split("/")[-1].split("__")[0]

        if variable(file) == "Cl":
            Cl_list.append(file)

        elif variable(file) == "Cp":
            Cp_list.append(file)

        elif variable(file) == "speed_up":
            speed_up_list.append(file)

    return Cl_list, Cp_list, speed_up_list


def format_array(Cp_file, Cl_file):

    with open(Cp_file, "rb") as f:
        Cp = pk.load(f)

    with open(Cl_file, "rb") as f:
        Cl = pk.load(f)

    center, radius = get_infection_site(Cp_file)

    return Cp, Cl, center, radius


def get_mesh_properties(
    x_dom,
    y_dom,
    t_dom,
    h,
    k,
    verbose=True,
):

    size_x = int(((x_dom[1] - x_dom[0]) / (h)))
    size_y = int(((y_dom[1] - y_dom[0]) / (h)))
    size_t = int(((t_dom[1] - t_dom[0]) / (k)))

    if verbose:
        print(
            "Steps in time = {:d}\nSteps in space_x = {:d}\nSteps in space_y = {:d}\n".format(
                size_t,
                size_x,
                size_y,
            )
        )

    return (size_x, size_y, size_t)


def create_input_mesh(
    t_dom,
    x_dom,
    size_t,
    size_x,
    sample_percent=None,
    Cl_fvm=None,
    Cp_fvm=None,
    random_state=42,
):
    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float64
    )

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float64)

    x_idx = np.arange(size_x)

    x_idx_mesh, t_mesh = np.meshgrid(x_idx, t_np)

    x_mesh = x_np[x_idx_mesh]

    if sample_percent is not None:

        if Cl_fvm is None or Cp_fvm is None:
            raise ValueError("Cl_fvm e Cp_fvm devem ser fornecidos para amostragem.")

        if not (0 < sample_percent <= 100):
            raise ValueError("sample_percent deve estar entre 0 e 100.")

        n_total = len(t_mesh)
        n_samples = int((sample_percent / 100) * n_total)

        rng = np.random.default_rng(random_state)
        chosen_points = rng.choice(n_total, size=n_samples, replace=False)

        # Opcional: ordenar para manter uma sequência mais organizada
        chosen_points = np.sort(chosen_points)

        return (
            Cl_fvm[chosen_points],
            Cp_fvm[chosen_points],
            t_mesh[chosen_points],
            x_mesh[chosen_points],
        )

    return (
        t_mesh,
        x_mesh,
    )


def allocates_training_mesh(
    t_dom,
    x_dom,
    size_t,
    size_x,
    center_x,
    initial_cond,
    radius,
    Cl_fvm,
    Cp_fvm,
    samples_percent=None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    initial_tc = (
        torch.tensor(initial_cond, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_(True)
        .to(device)
    )

    center_x_tc = (
        torch.tensor(center_x, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_(True)
        .to(device)
    )

    radius_tc = (
        torch.tensor(radius, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_(True)
        .to(device)
    )

    # Malha completa já linearizada
    t_mesh, x_mesh = create_input_mesh(
        t_dom,
        x_dom,
        size_t,
        size_x,
    )

    t_tc = torch.tensor(t_mesh, dtype=torch.float64).reshape(-1, 1)
    x_tc = torch.tensor(x_mesh, dtype=torch.float64).reshape(-1, 1)

    data_tc = torch.cat([t_tc, x_tc], dim=1).requires_grad_(True).to(device)

    target = torch.tensor(
        np.array([Cl_fvm.ravel(), Cp_fvm.ravel()]).T,
        dtype=torch.float64,
    ).to(device)

    if samples_percent is not None:

        (
            reduced_Cl,
            reduced_Cp,
            reduced_t_mesh,
            reduced_x_mesh,
        ) = create_input_mesh(
            t_dom,
            x_dom,
            size_t,
            size_x,
            sample_percent=samples_percent,
            Cl_fvm=Cl_fvm,
            Cp_fvm=Cp_fvm,
        )

        reduced_t_tc = torch.tensor(reduced_t_mesh, dtype=torch.float64).reshape(-1, 1)

        reduced_x_tc = torch.tensor(reduced_x_mesh, dtype=torch.float64).reshape(-1, 1)

        reduced_data_tc = (
            torch.cat([reduced_t_tc, reduced_x_tc], dim=1)
            .requires_grad_(True)
            .to(device)
        )

        reduced_target = torch.tensor(
            np.array([reduced_Cl.ravel(), reduced_Cp.ravel()]).T,
            dtype=torch.float64,
        ).to(device)

        print("Number of reduced points:", len(reduced_x_tc))

        return (
            initial_tc,
            center_x_tc,
            radius_tc,
            data_tc,
            target,
            reduced_data_tc,
            reduced_target,
            device,
        )

    return (
        initial_tc,
        center_x_tc,
        radius_tc,
        data_tc,
        target,
        device,
    )


def generate_initial_points(num_points, device, center_x_tc, radius_tc, initial_tc):

    t = torch.zeros(num_points, 1, dtype=torch.float64)

    x = torch.rand(num_points, 1, dtype=torch.float64)

    euclidean_distances = ((x - center_x_tc.item()) ** 2) ** 0.5

    inside_circle_mask = euclidean_distances <= radius_tc.item()

    result = torch.cat([x, euclidean_distances, inside_circle_mask], dim=1)

    C_init = torch.zeros((len(x), 2), dtype=torch.float64)

    C_init[:, 1] = inside_circle_mask.to(device).ravel() * initial_tc.ravel()

    return (
        (t.requires_grad_(True), x.requires_grad_(True)),
        C_init.to(device),
    )


def initial_condition(batch, model, device):
    t, x = batch

    input_data = torch.cat([t, x], dim=1).to(device)

    return model(input_data)


def generate_boundary_points(num_points, device, t_upper):

    t = torch.rand(num_points, 1, dtype=torch.float64) * t_upper

    x = (
        torch.tensor([0.0, 1], dtype=torch.float64)
        .repeat(num_points // 2, 1)
        .view(-1, 1)
    )

    C = torch.zeros((len(x), 2), dtype=torch.float64)

    return (
        (t.requires_grad_(True), x.requires_grad_(True)),
        C.to(device),
    )


def boundary_condition(batch, model, Dn, X_nb, Db, device):

    t, x = batch

    t = t.to(device).requires_grad_(True)
    x = x.to(device).requires_grad_(True)

    input_data = torch.cat([t, x], dim=1)

    pred = model(input_data)

    Cl = pred[:, 0:1]
    Cp = pred[:, 1:2]

    dCl_dx = torch.autograd.grad(
        Cl,
        x,
        grad_outputs=torch.ones_like(Cl),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCp_dx = torch.autograd.grad(
        Cp,
        x,
        grad_outputs=torch.ones_like(Cp),
        create_graph=True,
        retain_graph=True,
    )[0]

    n = (
        torch.tensor([-1.0, 1.0], dtype=pred.dtype, device=device)
        .repeat(len(pred) // 2)
        .reshape(-1, 1)
    )

    Cl_boundary = (Dn * dCl_dx - X_nb * Cl * dCp_dx) * n
    Cp_boundary = (Db * dCp_dx) * n

    return torch.cat([Cl_boundary, Cp_boundary], dim=1)


def generate_pde_points(num_points, device, t_upper):
    # Generate random (uniform) points in [0, 1) for time, x, and y
    t = torch.rand(num_points, 1, dtype=torch.float64) * t_upper

    x = torch.rand(num_points, 1, dtype=torch.float64)

    C = torch.zeros((len(x), 2), dtype=torch.float64)

    # Set requires_grad=True so we can compute PDE derivatives using autograd
    # Move each tensor to the specified device
    return (
        (t.requires_grad_(True), x.requires_grad_(True)),
        C.to(device),
    )


def pde(
    batch,
    model,
    T_f,
    Cp0,
    cb,
    phi,
    lambd_nb,
    Db,
    gamma_n,
    Cn_max,
    lambd_bn,
    mi_n,
    Dn,
    X_nb,
    device,
):
    t, x = batch

    tau = t.clone().detach().to(device).requires_grad_(True) / T_f
    chi = x.clone().detach().to(device).requires_grad_(True)

    input_data = torch.cat([chi, tau], dim=1)

    pred = model(input_data)

    Cl = pred[:, 0:1]
    Cp = pred[:, 1:2]

    dCl_dx = torch.autograd.grad(
        Cl,
        chi,
        torch.ones_like(Cl),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCp_dx = torch.autograd.grad(
        Cp,
        chi,
        torch.ones_like(Cp),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCl_dt = torch.autograd.grad(
        Cl,
        tau,
        torch.ones_like(Cl),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCp_dt = torch.autograd.grad(
        Cp,
        tau,
        torch.ones_like(Cp),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    d2Cl_dx2 = torch.autograd.grad(
        dCl_dx,
        chi,
        torch.ones_like(dCl_dx),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    d2Cp_dx2 = torch.autograd.grad(
        dCp_dx,
        chi,
        torch.ones_like(dCp_dx),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    # Termos dos leucócitos
    qn = (gamma_n * Cp0 * T_f / phi) * Cp * (1 - Cl)
    rn = (lambd_bn * Cp0 * T_f / phi) * Cl * Cp + (mi_n * T_f / phi) * Cl

    chemotaxis_term = dCl_dx * dCp_dx + Cl * d2Cp_dx2

    Cl_eq = (
        (Dn * T_f / phi) * d2Cl_dx2
        - (X_nb * Cp0 * T_f / phi) * chemotaxis_term
        - rn
        + qn
        - dCl_dt
    )

    # Termos dos patógenos
    qb = (cb * T_f / phi) * Cp
    rb = (lambd_nb * Cn_max * T_f / (phi)) * Cl * Cp

    Cp_eq = (Db * T_f / (phi)) * d2Cp_dx2 - rb + qb - dCp_dt

    return torch.cat([Cl_eq, Cp_eq], dim=1)
