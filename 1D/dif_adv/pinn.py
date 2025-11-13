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


def under_sampling(n_samples, Cl, Cp):

    choosen_points = np.linspace(
        0, len(Cl) - 1, num=n_samples, endpoint=True, dtype=int
    )

    reduced_Cl = np.zeros((n_samples, Cl.shape[1], Cl.shape[2]))

    reduced_Cp = np.zeros((n_samples, Cp.shape[1], Cl.shape[2]))

    for i, idx in enumerate(choosen_points):

        reduced_Cl[i, :] = Cl[idx, :, :]

        reduced_Cp[i, :] = Cp[idx, :, :]

    return reduced_Cl, reduced_Cp, choosen_points


def create_input_mesh(
    source, t_dom, x_dom, size_t, size_x, n_samples=None, Cl_fvm=None, Cp_fvm=None
):

    x_np = np.linspace(
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float64
    )

    x_idx = np.linspace(0, size_x, num=size_x, endpoint=False, dtype=int)

    if n_samples:
        reduced_Cl, reduced_Cp, choosen_points = under_sampling(
            n_samples, Cl_fvm, Cp_fvm
        )
       
        t_np = np.linspace(
            t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float64
        )[choosen_points]

        x_idx_mesh, t_mesh = np.meshgrid(
            x_idx,
            t_np,
        )

        x_mesh = np.zeros_like(t_mesh)
        source_mesh = np.zeros_like(t_mesh)

        x_mesh = x_np[x_idx_mesh.ravel()]
        source_mesh = source[x_idx_mesh.ravel()]

        return (
            reduced_Cl,
            reduced_Cp,
            t_mesh,
            x_mesh,
            source_mesh,
        )

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float64)

    x_idx_mesh, t_mesh = np.meshgrid(
        x_idx,
        t_np,
    )

    x_mesh = np.zeros_like(t_mesh)
    source_mesh = np.zeros_like(t_mesh)

    x_mesh = x_np[x_idx_mesh.ravel()]
    source_mesh = source[x_idx_mesh.ravel()]

    return (
        t_mesh,
        x_mesh,
        source_mesh,
    )


def allocates_training_mesh(
    t_dom,
    x_dom,
    size_t,
    size_x,
    center_x,
    initial_cond,
    radius,
    Cp_fvm,
    Cl_fvm,
    source,
    n_samples=None,
):

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = "cpu"

    print("device:", device)

    initial_tc = (
        torch.tensor(initial_cond, dtype=torch.float64)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    center_x_tc = (
        torch.tensor(center_x, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    )

    radius_tc = (
        torch.tensor(radius, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    )

    print("criou as matrizes")

    if n_samples:
        print("Entrou no if")

        (
            reduced_Cl,
            reduced_Cp,
            reduced_t_mesh,
            reduced_x_mesh,
            reduced_src_mesh,
        ) = create_input_mesh(
            source,
            t_dom,
            x_dom,
            size_t,
            size_x,
            n_samples,
            Cl_fvm,
            Cp_fvm,
        )

        
        reduced_t_tc = torch.tensor(reduced_t_mesh, dtype=torch.float64).reshape(-1, 1)

        reduced_x_tc = torch.tensor(reduced_x_mesh, dtype=torch.float64).reshape(-1, 1)

        reduced_data_tc = (
            torch.cat([reduced_t_tc, reduced_x_tc], dim=1)
            .requires_grad_(True)
            .to(device)
        )

        reduced_src_tc = (
            torch.tensor(reduced_src_mesh, dtype=torch.float64)
            .reshape(-1, 1)
            .requires_grad_(True)
        )

        reduced_target = torch.tensor(
            np.array([reduced_Cl.flatten(), reduced_Cp.flatten()]).T,
            dtype=torch.float64,
        )

        return (
            initial_tc,
            center_x_tc,
            radius_tc,
            reduced_data_tc,
            reduced_src_tc,
            reduced_target,
            device,
        )

    else:
        (
            t_mesh,
            x_mesh,
            src_mesh,
        ) = create_input_mesh(source, t_dom, x_dom, size_t, size_x)

        t_tc = torch.tensor(t_mesh, dtype=torch.float64).reshape(-1, 1)

        x_tc = torch.tensor(x_mesh, dtype=torch.float64).reshape(-1, 1)

        data_tc = torch.cat([t_tc, x_tc], dim=1).requires_grad_(True).to(device)

        src_tc = (
            torch.tensor(src_mesh, dtype=torch.float64)
            .reshape(-1, 1)
            .requires_grad_(True)
        )

        target = torch.tensor(
            np.array([Cl_fvm.flatten(), Cp_fvm.flatten()]).T,
            dtype=torch.float64,
        )

        return (
            initial_tc,
            center_x_tc,
            radius_tc,
            data_tc,
            src_tc,
            target,
            device,
        )


def generate_initial_points(num_points, device, size_x):

    t = torch.zeros(num_points, 1, dtype=torch.float64)

    x = torch.rand(num_points, 1, dtype=torch.float64)

    a = 0
    a2 = 1 - (1 / size_x)
    b = 4
    c = 2

    C_init = torch.zeros((len(x), 2), dtype=torch.float64)

    C_init[:, 0] = torch.exp(-(((x.reshape(1, -1) - a) * b) ** 2)) / c

    C_init[:, 1] = torch.exp(-(((x.reshape(1, -1) - a2) * b) ** 2)) / c

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

    input_data = torch.cat([t, x], dim=1).to(device)

    pred = model(input_data)

    n = (
        torch.tensor([-1, 1], dtype=torch.float64)
        .repeat(len(pred) // 2, 1)
        .requires_grad_(True)
        .view(-1, 1)
        .to(device)
    )

    dCl_dx = torch.autograd.grad(
        pred[:, 0],
        x,
        torch.ones_like(pred[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCp_dx = torch.autograd.grad(
        pred[:, 1],
        x,
        torch.ones_like(pred[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    Cl_boundary = Dn * dCl_dx * n

    Cp_boundary = Db * dCp_dx

    # 4) Return them as one tensor, do NOT re-flag requires_grad
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


def generate_pde_source(original_source, h, batch, device):

    _, x_batch = batch.tensor_split(2, dim=1)  # [B, 1]
    x_domain = torch.arange(0, 1, h).view(-1, 1).to(device)  # [N, 1]

    # Identify active source locations in x_domain
    source_locs = x_domain[original_source.view(-1) == 1]  # shape [M, 1], where M ≤ N

    # Compute bounds
    l_bound = source_locs - h  # [M, 1]
    u_bound = source_locs + h  # [M, 1]

    # Broadcast and check
    x_batch_exp = x_batch[:, None, :]  # [B, 1, 1]
    l_bound_exp = l_bound[None, :, :]  # [1, M, 1]
    u_bound_exp = u_bound[None, :, :]  # [1, M, 1]

    # Check if x_batch[i] is within any [l_bound[j], u_bound[j]]
    in_range = (x_batch_exp > l_bound_exp) & (x_batch_exp < u_bound_exp)  # [B, M, 1]
    match = in_range.any(dim=1)  # [B, 1]

    # Generate new source
    new_source = torch.zeros_like(x_batch)
    new_source[match] = 1.0

    return new_source


def pde(
    batch,
    model,
    h,
    cb,
    phi,
    lambd_nb,
    Db,
    y_n,
    Cn_max,
    lambd_bn,
    mi_n,
    Dn,
    X_nb,
    device,
):

    t, x = batch

    input_data = torch.cat([t, x], dim=1).to(device)

    pred = model(input_data)

    dCl_dx = torch.autograd.grad(
        pred[:, 0],
        x,
        torch.ones_like(pred[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCp_dx = torch.autograd.grad(
        pred[:, 1],
        x,
        torch.ones_like(pred[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCl_dt = torch.autograd.grad(
        pred[:, 0],
        t,
        torch.ones_like(pred[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    dCp_dt = torch.autograd.grad(
        pred[:, 1],
        t,
        torch.ones_like(pred[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    d2Cl_dx2 = torch.autograd.grad(
        dCl_dx,
        x,
        torch.ones_like(dCl_dx),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    d2Cp_dx2 = torch.autograd.grad(
        dCp_dx,
        x,
        torch.ones_like(dCp_dx),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    qn = y_n * pred[:, 1].ravel() * (Cn_max - pred[:, 0])  # [1000, 1]

    rn = lambd_bn * pred[:, 0].ravel() * pred[:, 1] + mi_n * pred[:, 0]  # [1000, 1]

    Cl_eq = (
        Dn * d2Cl_dx2.ravel()
        - X_nb * ((dCl_dx * dCp_dx).ravel() + pred[:, 0] * d2Cp_dx2.ravel())
        - dCl_dt.ravel() * phi
    )  # All shapes [1000, 1]

    qb = cb * pred[:, 1]

    rb = lambd_nb * pred[:, 0].ravel() * pred[:, 1]

    Cp_eq = Db * d2Cp_dx2.ravel() - dCp_dt.ravel() * phi  # All shapes [1000, 1]

    return torch.cat([Cl_eq.reshape(-1, 1), Cp_eq.reshape(-1, 1)], dim=1)
