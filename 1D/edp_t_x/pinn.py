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
    size_t = int(((t_dom[1] - t_dom[0]) / (k)) + 1)

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
        x_dom[0], x_dom[-1], num=size_x, endpoint=False, dtype=np.float32
    )

    x_idx = np.linspace(0, size_x, num=size_x, endpoint=False, dtype=int)

    if n_samples:

        reduced_Cl, reduced_Cp, choosen_points = under_sampling(
            n_samples, Cl_fvm, Cp_fvm
        )

        t_np = np.linspace(
            t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32
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

    t_np = np.linspace(t_dom[0], t_dom[-1], num=size_t, endpoint=True, dtype=np.float32)

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

    (
        t_mesh,
        x_mesh,
        src_mesh,
    ) = create_input_mesh(source, t_dom, x_dom, size_t, size_x)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = "cpu"

    print("device:", device)

    initial_tc = (
        torch.tensor(initial_cond, dtype=torch.float32)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    center_x_tc = (
        torch.tensor(center_x, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    )

    radius_tc = (
        torch.tensor(radius, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    )

    t_tc = torch.tensor(t_mesh, dtype=torch.float32).reshape(-1, 1)

    x_tc = torch.tensor(x_mesh, dtype=torch.float32).reshape(-1, 1)

    data_tc = torch.cat([t_tc, x_tc], dim=1).requires_grad_(True).to(device)

    src_tc = (
        torch.tensor(src_mesh, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    )

    target = torch.tensor(
        np.array([Cl_fvm.flatten(), Cp_fvm.flatten()]).T,
        dtype=torch.float32,
    )

    if n_samples:

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

        reduced_t_tc = torch.tensor(reduced_t_mesh, dtype=torch.float32).reshape(-1, 1)

        reduced_x_tc = torch.tensor(reduced_x_mesh, dtype=torch.float32).reshape(-1, 1)

        reduced_data_tc = (
            torch.cat([reduced_t_tc, reduced_x_tc], dim=1)
            .requires_grad_(True)
            .to(device)
        )

        reduced_src_tc = (
            torch.tensor(reduced_src_mesh, dtype=torch.float32)
            .reshape(-1, 1)
            .requires_grad_(True)
        )

        reduced_target = torch.tensor(
            np.array([reduced_Cl.flatten(), reduced_Cp.flatten()]).T,
            dtype=torch.float32,
        )

        return (
            initial_tc,
            center_x_tc,
            radius_tc,
            data_tc,
            src_tc,
            target,
            reduced_data_tc,
            reduced_src_tc,
            reduced_target,
            device,
        )

    else:
        return (
            initial_tc,
            center_x_tc,
            radius_tc,
            data_tc,
            src_tc,
            target,
            device,
        )


def generate_pde_points(num_points, device):
    # Generate random (uniform) points in [0, 1) for time, x, and y
    t = torch.rand(num_points, 1, dtype=torch.float32) * 5

    x = torch.rand(num_points, 1, dtype=torch.float32)

    C = torch.zeros((len(x), 2), dtype=torch.float32)

    # Set requires_grad=True so we can compute PDE derivatives using autograd
    # Move each tensor to the specified device
    return (
        torch.cat([t.requires_grad_(True), x.requires_grad_(True)], dim=1).to(device),
        C.to(device),
    )


def generate_boundary_points(num_points, device):

    t = torch.rand(num_points, 1, dtype=torch.float32) * 5

    x = (
        torch.tensor([0.0, 1], dtype=torch.float32)
        .repeat(num_points // 2, 1)
        .view(-1, 1)
    )

    C = torch.zeros((len(x), 2), dtype=torch.float32)

    return (
        torch.cat([t.requires_grad_(True), x.requires_grad_(True)], dim=1).to(device),
        C.to(device),
    )


def generate_initial_points(num_points, device, center_x_tc, radius_tc, initial_tc):
    t = torch.zeros(num_points, 1, dtype=torch.float32)

    x = torch.rand(num_points, 1, dtype=torch.float32)

    euclidean_distances = ((x - center_x_tc) ** 2) ** 0.5

    inside_circle_mask = euclidean_distances <= radius_tc

    C_init = torch.zeros((len(x), 2), dtype=torch.float32)

    C_init[:, 1] = inside_circle_mask.ravel() * initial_tc.ravel()

    return (
        torch.cat([t.requires_grad_(True), x.requires_grad_(True)], dim=1).to(device),
        C_init.to(device),
    )


def boundary_condition(pred, batch, Dn, X_nb, Db, device):

    n = (
        torch.tensor([-1, 1], dtype=torch.float32)
        .repeat(len(pred) // 2, 1)
        .requires_grad_(True)
        .view(-1, 1)
        .to(device)
    )

    dCl = torch.autograd.grad(
        pred[:, 0],
        batch,
        torch.ones_like(pred[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCp = torch.autograd.grad(
        pred[:, 1],
        batch,
        torch.ones_like(pred[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Separar as derivadas parciais em duas colunas
    _, dCl_dx = dCl.tensor_split(2, dim=1)
    _, dCp_dx = dCp.tensor_split(2, dim=1)

    Cl_boundary = Dn * dCl_dx * n - X_nb * torch.matmul(pred[:, 0].ravel(), dCp_dx * n)

    Cp_boundary = Db * dCp_dx * n

    # 4) Return them as one tensor, do NOT re-flag requires_grad
    return torch.cat([Cl_boundary, Cp_boundary], dim=1)


def generate_pde_source(original_source, h, batch, device):

    _, x_batch = batch.tensor_split(2, dim=1)  # [B, 1]
    x_domain = torch.arange(0, 1, h).view(-1, 1).to(device)  # [N, 1]

    # Identify active source locations in x_domain
    source_locs = x_domain[original_source.view(-1) == 1]  # shape [M, 1], where M ≤ N
    source_locs

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
    pred,
    batch,
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
    original_source,
    device,
):

    dCl = torch.autograd.grad(
        pred[:, 0],
        batch,
        torch.ones_like(pred[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCp = torch.autograd.grad(
        pred[:, 1],
        batch,
        torch.ones_like(pred[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Separar as derivadas parciais em duas colunas
    dCl_dt, dCl_dx = dCl.tensor_split(2, dim=1)
    dCp_dt, dCp_dx = dCp.tensor_split(2, dim=1)

    d2Cl = torch.autograd.grad(
        dCl_dx,
        batch,
        torch.ones_like(dCl_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2Cp = torch.autograd.grad(
        dCp_dx,
        batch,
        torch.ones_like(dCp_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Separar as derivadas parciais em duas colunas
    _, d2Cl_dx2 = d2Cl.tensor_split(2, dim=1)
    _, d2Cp_dx2 = d2Cp.tensor_split(2, dim=1)

    # Calculating Cl value

    source = generate_pde_source(original_source, h, batch, device)

    qn = y_n * pred[:, 1:2] * (Cn_max - pred[:, 0:1]) * source  # [1000, 1]
    rn = lambd_bn * pred[:, 0:1] * pred[:, 1:2] + mi_n * pred[:, 0:1]  # [1000, 1]

    Cl_eq = (
        Dn * d2Cl_dx2 - X_nb * (pred[:, 1:2] * d2Cp_dx2 + dCl_dx * dCp_dx) - rn + qn
    ) - dCl_dt * phi  # All shapes [1000, 1]

    qb = cb * pred[:, 1:2]
    rb = lambd_nb * pred[:, 0:1] * pred[:, 1:2]

    Cp_eq = Db * d2Cp_dx2 - rb + qb - dCp_dt * phi  # All shapes [1000, 1]

    return torch.cat([Cl_eq, Cp_eq], dim=1)


class Trainer:

    def shuffle_data(self, *arrays):
        indices = np.random.permutation(arrays[0].shape[0])

        return tuple(array[indices] for array in arrays)

    def train_test_split(
        self,
        *arrays,
        test_size=0.5,
        shuffle=True,
    ):
        with torch.no_grad():
            if shuffle:
                arrays = self.shuffle_data(*arrays)

            # Determine train-test split index
            total_samples = arrays[0].shape[0]
            if 0 < test_size < 1:
                split_idx = total_samples - int(total_samples * test_size)
            elif isinstance(test_size, int) and 1 <= test_size < total_samples:
                split_idx = total_samples - test_size
            else:
                raise ValueError(
                    "Invalid test_size: must be a float (0 < x < 1) or int < len(data)"
                )

            # Perform the split for each array
            train_set = tuple(array[:split_idx].to(self.device) for array in arrays)
            test_set = tuple(array[split_idx:].to(self.device) for array in arrays)

            # Flatten and return
            return (*train_set, *test_set)

    def add_loss(self, loss_obj, weigth=1):
        self.losses.append(loss_obj)
        self.lossesW.append(weigth)

    def __init__(
        self,
        n_epochs,
        batch_size,
        model,
        device,
        constant_properties,
        target=[],
        data=[],
        patience=300,
        tolerance=1e-3,
        val_steps=None,
        print_steps=5000,
        validation=None,
        optimizer=None,
        scheduler=None,
    ):

        self.model = model.to(device)
        self.device = device
        self.constant_properties = constant_properties
        self.validation = validation
        self.tolerance = tolerance
        self.patience = patience
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler else None
        self.print_steps = print_steps
        self.losses = []
        self.lossesW = []

        if len(data) != 0 and len(target) != 0:

            self.n_batchs = int(ceil(len(data) / batch_size))

            self.val_steps = val_steps if val_steps else self.n_batchs

            self.n_it = int(n_epochs * self.n_batchs)

            if self.validation:
                (
                    self.data_train,
                    self.target_train,
                    self.data_test,
                    self.target_test,
                ) = self.train_test_split(
                    data,
                    target,
                    test_size=self.validation,
                )

            else:

                self.data_train = data
                self.data_test = None
                self.target_train = target
                self.target_test = None

            self.add_loss(
                RMSE(
                    self.data_train,
                    self.target_train,
                    device,
                    name="Data Loss",
                    batch_size=batch_size,
                    shuffle=False,
                )
            )

            return

        self.n_it = n_epochs
        self.val_steps = val_steps if val_steps else 1

        return

    def train(
        self,
    ):

        if self.losses == []:
            print("No loss function added")
            return

        loss_dict = {}

        for loss in self.losses:
            loss_dict[loss.name] = []

        loss_dict["val"] = []

        patience_count = 0
        val_loss = torch.tensor([1000])

        for it in range(self.n_it):
            start_time = time.time()  # Start timing the iteration

            self.model.zero_grad()
            self.optimizer.zero_grad()
            total_loss = 0
            losses = []

            for weighth, loss_obj in zip(self.lossesW, self.losses):

                loss = loss_obj.forward(self.model)

                total_loss += loss * weighth

                losses.append((loss * weighth).item())

                if it % self.n_batchs == 0:
                    loss_dict[loss_obj.name].append(loss.item())

            # Backward pass
            total_loss.backward()

            # Update weights
            self.optimizer.step()

            iteration_time = time.time() - start_time  # Calculate iteration duration

            if it / self.n_batchs % self.print_steps == 0:
                print(
                    "Iteration {}: total loss {:.4f}, losses: {}, learning rate: {:.10f}, time: {:.4f}s".format(
                        it // self.n_batchs,
                        total_loss.item(),
                        losses,
                        self.scheduler.get_last_lr()[0],
                        iteration_time,
                    )
                )

            self.scheduler.step(total_loss)

            # Computing validation loss

            if it % self.val_steps:
                with torch.no_grad():
                    val_old = val_loss

                    val_loss = torch.mean(
                        torch.sum(
                            ((self.target_test - self.model(self.data_test))) ** 2,
                            dim=1,
                        )
                        ** 0.5
                    )

                if it % self.n_batchs == 0:
                    loss_dict["val"].append(val_loss.item())

                if self.tolerance and self.validation:

                    if (
                        abs(val_old.item() - val_loss.item()) / val_old.item()
                        < self.tolerance
                    ):
                        patience_count += 1

                    else:
                        patience_count = 0

                    if patience_count >= self.patience:

                        print(
                            "Iteration {}: total loss {:.4f}, losses: {}, learning rate: {:.10f}, time: {:.4f}s".format(
                                it // self.n_batchs,
                                total_loss.item(),
                                losses,
                                self.scheduler.get_last_lr()[0],
                                iteration_time,
                            )
                        )

                        print("Early break!")

                        break

        return self.model, loss_dict
