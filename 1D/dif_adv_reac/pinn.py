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
    shuffle=False,
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

    if shuffle:
        data_tc = data_tc.detach()
        target = target.detach()

        idx = torch.randperm(data_tc.shape[0])

        data_tc = data_tc[idx].clone()
        data_tc = target[idx].clone()

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


def evaluate_model(model, data_test, target_test, reference_time, device):
    """
    Avalia o modelo no conjunto de teste.

    Retorna:
        error:
            erro absoluto ponto a ponto

        test_time:
            tempo de inferência

        speed_up:
            speed up

        pred:
            predição do modelo
    """

    model.eval()

    data_test = data_test.to(device)

    target_test = target_test.cpu().detach().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():

        pred = model(data_test).cpu().detach().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    test_time = end - start

    # erro absoluto ponto a ponto
    error = np.abs(pred - target_test)

    speed_up = reference_time / test_time

    return (
        error,
        test_time,
        speed_up,
        pred,
    )


def generate_initial_points(num_points, device, b, c, a=0.3):
    """
    Gera os pontos iniciais para a PINN em t = 0.

    A concentração inicial de bactérias/patógenos segue:

        cb(x) = exp(-(((x - a) * b) ** 2)) / c

    assumindo que:
        C_init[:, 0] = Cl
        C_init[:, 1] = Cp ou Cb
    """

    t = torch.zeros(
        num_points, 1, dtype=torch.float64, device=device, requires_grad=True
    )

    x = torch.rand(
        num_points, 1, dtype=torch.float64, device=device, requires_grad=True
    )

    # Garante que b e c estejam no mesmo device e tipo
    b = torch.as_tensor(b, dtype=torch.float64, device=device)
    c = torch.as_tensor(c, dtype=torch.float64, device=device)
    a = torch.as_tensor(a, dtype=torch.float64, device=device)

    # Condição inicial gaussiana
    cb_init = torch.exp(-(((x - a) * b) ** 2)) / c

    C_init = torch.zeros(num_points, 2, dtype=torch.float64, device=device)

    # Mantém Cl inicial como zero
    C_init[:, 0] = 0.0

    # Aplica a condição inicial em Cp/Cb
    C_init[:, 1:2] = cb_init

    return (
        (t, x),
        C_init,
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


def pde_cl(
    batch,
    model,
    phi,
    gamma_n,
    Cn_max,
    lambd_bn,
    mi_n,
    Dn,
    X_nb,
    device,
    delta_cl,
    delta_cp,
    min_cl,
    min_cp,
    T_f,
    Cp0,
    L=1.0,
):
    t, x = batch

    # Variáveis adimensionais de entrada
    tau = (t.clone().detach().to(device) / T_f).requires_grad_(True)
    xi = (x.clone().detach().to(device) / L).requires_grad_(True)

    input_data = torch.cat([xi, tau], dim=1)

    pred = model(input_data)

    # Saídas normalizadas da rede
    Cl_hat = pred[:, 0:1]
    Cp_hat = pred[:, 1:2]

    # Variáveis físicas/desnormalizadas
    Cl = Cl_hat * delta_cl + min_cl
    Cp = Cp_hat * delta_cp + min_cp

    # Variáveis adimensionais
    Cl_bar = Cl / Cn_max
    Cp_bar = Cp / Cp0

    # Derivadas adimensionais
    dCl_dtau = torch.autograd.grad(
        Cl_bar,
        tau,
        grad_outputs=torch.ones_like(Cl_bar),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCl_dxi = torch.autograd.grad(
        Cl_bar,
        xi,
        grad_outputs=torch.ones_like(Cl_bar),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCp_dxi = torch.autograd.grad(
        Cp_bar,
        xi,
        grad_outputs=torch.ones_like(Cp_bar),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2Cl_dxi2 = torch.autograd.grad(
        dCl_dxi,
        xi,
        grad_outputs=torch.ones_like(dCl_dxi),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2Cp_dxi2 = torch.autograd.grad(
        dCp_dxi,
        xi,
        grad_outputs=torch.ones_like(dCp_dxi),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Números adimensionais
    diffusion_coef = Dn * T_f / (phi * L**2)
    chemotaxis_coef = X_nb * Cp0 * T_f / (phi * L**2)
    reaction_coef = lambd_bn * Cp0 * T_f / phi
    apoptosis_coef = mi_n * T_f / phi
    source_coef = gamma_n * Cp0 * T_f / phi

    # Termo de quimiotaxia adimensional
    chemotaxis_term = dCl_dxi * dCp_dxi + Cl_bar * d2Cp_dxi2

    # Resíduo adimensional da EDP dos leucócitos
    Cl_eq = (
        diffusion_coef * d2Cl_dxi2
        - chemotaxis_coef * chemotaxis_term
        - reaction_coef * Cl_bar * Cp_bar
        - apoptosis_coef * Cl_bar
        + source_coef * Cp_bar * (1 - Cl_bar)
        - dCl_dtau
    )

    return torch.cat(
        [Cl_eq.reshape(-1, 1), torch.zeros_like(Cl_eq.reshape(-1, 1))],
        dim=1,
    )


def pde_cp(
    batch,
    model,
    cb,
    phi,
    lambd_nb,
    Db,
    device,
    delta_cl,
    delta_cp,
    min_cl,
    min_cp,
    T_f,
    Cp0,
    Cn_max,
    L=1.0,
):
    t, x = batch

    # Variáveis adimensionais de entrada
    tau = (t.clone().detach().to(device) / T_f).requires_grad_(True)
    xi = (x.clone().detach().to(device) / L).requires_grad_(True)

    input_data = torch.cat([xi, tau], dim=1)

    pred = model(input_data)

    # Saídas normalizadas da rede
    Cl_hat = pred[:, 0:1]
    Cp_hat = pred[:, 1:2]

    # Variáveis físicas/desnormalizadas
    Cl = Cl_hat * delta_cl + min_cl
    Cp = Cp_hat * delta_cp + min_cp

    # Variáveis adimensionais
    Cl_bar = Cl / Cn_max
    Cp_bar = Cp / Cp0

    # Derivadas adimensionais
    dCp_dtau = torch.autograd.grad(
        Cp_bar,
        tau,
        grad_outputs=torch.ones_like(Cp_bar),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCp_dxi = torch.autograd.grad(
        Cp_bar,
        xi,
        grad_outputs=torch.ones_like(Cp_bar),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2Cp_dxi2 = torch.autograd.grad(
        dCp_dxi,
        xi,
        grad_outputs=torch.ones_like(dCp_dxi),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Números adimensionais
    diffusion_coef = Db * T_f / (phi * L**2)
    death_coef = lambd_nb * Cn_max * T_f / phi
    growth_coef = cb * T_f / phi

    # Resíduo adimensional da EDP dos patógenos
    Cp_eq = (
        diffusion_coef * d2Cp_dxi2
        - death_coef * Cl_bar * Cp_bar
        + growth_coef * Cp_bar
        - dCp_dtau
    )

    return torch.cat(
        [torch.zeros_like(Cp_eq.reshape(-1, 1)), Cp_eq.reshape(-1, 1)],
        dim=1,
    )


def pinn_training(
    n_epochs,
    batch_size,
    model,
    device,
    beta1,
    beta2,
    pinn_batch,
    center_x_tc,
    radius_tc,
    b,
    c,
    t_dom,
    Dn,
    X_nb,
    Db,
    cb,
    phi,
    lambd_nb,
    y_n,
    Cn_max,
    lambd_bn,
    mi_n,
    data_tc,
    target,
    delta_cl,
    delta_cp,
    min_cl,
    min_cp,
):

    trainer = Trainer(
        n_epochs=n_epochs,
        batch_size=batch_size,
        model=model,
        device=device,
        patience=5000,
        tolerance=0.01,
        betas=(beta1, beta2),
        print_steps=1e3,
        adaptive=True,
    )

    init_loss = LOSS(
        device=device,
        name="Inital",
        batch_size=pinn_batch,
        criterium="MSE",
    )

    init_loss.setBatchGenerator(
        generate_initial_points,
        b,
        c,
    )

    init_loss.setEvalFunction(
        initial_condition,
        device,
    )

    trainer.add_loss(init_loss)

    bnd_loss = LOSS(
        device=device,
        name="Boundary",
        batch_size=pinn_batch,
        criterium="MSE",
    )

    bnd_loss.setBatchGenerator(generate_boundary_points, t_dom[1])

    bnd_loss.setEvalFunction(boundary_condition, Dn, X_nb, Db, device)

    trainer.add_loss(bnd_loss)

    pde_cl_loss = LOSS(
        device=device,
        name="PDE leukocytes",
        batch_size=pinn_batch,
        criterium="MSE",
    )

    pde_cl_loss.setBatchGenerator(generate_pde_points, t_dom[1])

    pde_cl_loss.setEvalFunction(
        pde_cl,
        phi,
        y_n,
        Cn_max,
        lambd_bn,
        mi_n,
        Dn,
        X_nb,
        device,
        delta_cl,
        delta_cp,
        min_cl,
        min_cp,
        t_dom[-1],
        0.5,
    )

    trainer.add_loss(pde_cl_loss)

    pde_cp_loss = LOSS(
        device=device,
        name="PDE pathogens",
        batch_size=pinn_batch,
        criterium="MSE",
    )

    pde_cp_loss.setBatchGenerator(generate_pde_points, t_dom[1])

    pde_cp_loss.setEvalFunction(
        pde_cp,
        cb,
        phi,
        lambd_nb,
        Db,
        device,
        delta_cl,
        delta_cp,
        min_cl,
        min_cp,
        t_dom[-1],
        0.5,
        Cn_max,
    )

    trainer.add_loss(pde_cp_loss)

    data_loss = LOSS(
        device,
        name="Data Loss",
        batch_size=batch_size,
        criterium="MSE",
    )

    data_loss.add_data(
        data_tc,
        target,
    )

    trainer.add_loss(data_loss)

    start = time.time()

    model, loss_dict = trainer.train()

    end = time.time()

    pinn_time = end - start

    del trainer

    return model, loss_dict, pinn_time


def nn_training(
    n_epochs,
    batch_size,
    model,
    device,
    beta1,
    beta2,
    data_tc,
    target,
):

    trainer = Trainer(
        n_epochs=n_epochs,
        batch_size=batch_size,
        model=model,
        device=device,
        patience=5000,
        tolerance=0.01,
        betas=(beta1, beta2),
        print_steps=1e3,
        adaptive=True,
    )

    data_loss = LOSS(
        device,
        name="Data Loss",
        batch_size=batch_size,
        criterium="MSE",
    )

    data_loss.add_data(
        data_tc,
        target,
    )

    trainer.add_loss(data_loss)

    start = time.time()

    model, loss_dict = trainer.train()

    end = time.time()

    nn_time = end - start

    del trainer

    return model, loss_dict, nn_time


def main():
    return 0


if __name__ == "__main__":
    main()
