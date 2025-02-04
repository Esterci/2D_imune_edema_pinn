import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle as pk
import matplotlib.pyplot as plt
import json
from glob import glob

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

arch_str = "Tanh--32__Tanh--32__Tanh--32__Tanh--32"


def get_infection_site(struct_name):

    center_str = (struct_name).split("__")[-2].split("(")[-1].split(")")[0].split(",")

    center = (float(center_str[0]), float(center_str[1]))

    radius = float(struct_name.split("__")[-1].split("--")[-1].split(".pkl")[0])

    return center, radius


def read_files(path):
    file_list = sorted(glob(path + "/*"))

    speed_up_list = []
    Cb_list = []
    Cn_list = []

    for file in file_list:

        variable = lambda a: a.split("/")[-1].split("__")[0]

        if variable(file) == "Cl":
            Cn_list.append(file)

        elif variable(file) == "Cp":
            Cb_list.append(file)

        elif variable(file) == "speed_up":
            speed_up_list.append(file)

    return Cn_list, Cb_list, speed_up_list


def format_array(Cb_list, Cn_list):

    for i, (Cb_file, Cn_file) in enumerate(zip(Cb_list, Cn_list)):
        with open(Cb_file, "rb") as f:
            new_Cb = pk.load(f)

        with open(Cn_file, "rb") as f:
            new_Cn = pk.load(f)

        sim_shape = new_Cb.shape

        if i == 0:
            Cb = np.zeros(
                (len(Cb_list), sim_shape[0], sim_shape[1], sim_shape[2], sim_shape[3])
            )

            Cn = np.zeros(
                (len(Cn_list), sim_shape[0], sim_shape[1], sim_shape[2], sim_shape[3])
            )

            center_x = np.zeros(len(Cb_list))

            center_y = np.zeros(len(Cb_list))

            radius_array = np.zeros(len(Cb_list))

        Cb[i, :, :, :, :] = new_Cb

        Cn[i, :, :, :, :] = new_Cn

        center, radius = get_infection_site(Cb_file)

        center_x[i], center_y[i] = center

        radius_array[i] = radius

    return Cb, Cn, center_x, center_y, radius_array


def get_mesh_properties(
    x_dom,
    y_dom,
    t_dom,
    h,
    k,
    central_ini_cond,
    ini_cond_var,
    n_ini,
    verbose=True,
):

    size_x = int(((x_dom[1] - x_dom[0]) / (h)))
    size_y = int(((y_dom[1] - y_dom[0]) / (h)))
    size_t = int(((t_dom[1] - t_dom[0]) / (k)) + 1)

    initial_cond = np.linspace(
        central_ini_cond * (1 - ini_cond_var),
        central_ini_cond * (1 + ini_cond_var),
        num=n_ini,
        endpoint=True,
        dtype=np.float16,
    )

    if verbose:
        print(
            "Steps in time = {:d}\nSteps in space_x = {:d}\nSteps in space_y = {:d}\n".format(
                size_t,
                size_x,
                size_y,
            )
        )

    return (size_x, size_y, size_t, initial_cond)


def create_input_mesh(
    t_dom,
    x_dom,
    y_dom,
    size_t,
    size_x,
    size_y,
    center_x_array,
    center_y_array,
    initial_cond,
    radius_array,
):
    t_np = np.linspace(t_dom[0], t_dom[1], num=size_t, endpoint=False, dtype=np.float32)
    x_np = np.linspace(x_dom[0], x_dom[1], num=size_x, endpoint=False, dtype=np.float32)
    y_np = np.linspace(y_dom[0], y_dom[1], num=size_y, endpoint=False, dtype=np.float32)
    infection_idx = np.linspace(
        0, len(center_x_array), num=len(center_x_array), endpoint=False, dtype=np.int32
    )

    # Change first with second dimension for np.meshgrid match with
    # torch.mashgrid and C flattening logic

    initial_mesh, infection_mesh, t_mesh, x_mesh, y_mesh = np.meshgrid(
        initial_cond, infection_idx, t_np, x_np, y_np
    )

    center_x_mesh = np.zeros(infection_mesh.ravel().shape)
    center_y_mesh = np.zeros(infection_mesh.ravel().shape)
    radius_mesh = np.zeros(infection_mesh.ravel().shape)

    for i, idx in enumerate(infection_mesh.ravel()):

        center_x_mesh[i] = center_x_array[idx]
        center_y_mesh[i] = center_y_array[idx]
        radius_mesh[i] = radius_array[idx]

    return (
        initial_mesh,
        center_x_mesh,
        center_y_mesh,
        radius_mesh,
        t_mesh,
        x_mesh,
        y_mesh,
    )


def allocates_training_mesh(
    t_dom,
    x_dom,
    y_dom,
    size_t,
    size_x,
    size_y,
    center_x_array,
    center_y_array,
    initial_cond,
    radius_array,
    Cb_fvm,
    Cn_fvm,
):

    (
        initial_mesh,
        center_x_mesh,
        center_y_mesh,
        radius_mesh,
        t_mesh,
        x_mesh,
        y_mesh,
    ) = create_input_mesh(
        t_dom,
        x_dom,
        y_dom,
        size_t,
        size_x,
        size_y,
        center_x_array,
        center_y_array,
        initial_cond,
        radius_array,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = "cpu"

    print("device:", device)

    initial_tc = (
        torch.tensor(initial_mesh, dtype=torch.float16)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    center_x_tc = (
        torch.tensor(center_x_mesh, dtype=torch.float32)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    center_y_tc = (
        torch.tensor(center_y_mesh, dtype=torch.float32)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    radius_tc = (
        torch.tensor(radius_mesh, dtype=torch.float32)
        .reshape(-1, 1)
        .requires_grad_(True)
    )

    t_tc = torch.tensor(t_mesh, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)

    x_tc = torch.tensor(x_mesh, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)

    y_tc = torch.tensor(y_mesh, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)

    target = torch.tensor(
        np.hstack((Cn_fvm.reshape(-1, 1), Cb_fvm.reshape(-1, 1))), dtype=torch.float32
    )

    return (
        initial_tc,
        center_x_tc,
        center_y_tc,
        radius_tc,
        t_tc,
        x_tc,
        y_tc,
        target,
        device,
    )


def generate_model(arch_str):
    hidden_layers = arch_str.split("__")

    modules = []

    for params in hidden_layers:
        if len(params) != 0:
            activation, out_neurons = params.split("--")

            if len(modules) == 0:
                if activation == "Linear":
                    modules.append(
                        activation_dict[activation](7, int(out_neurons)).float()
                    )

                else:
                    modules.append(nn.Linear(7, int(out_neurons)).float())
                    modules.append(activation_dict[activation]().float())

            else:
                if activation == "Linear":
                    modules.append(
                        activation_dict[activation](
                            int(in_neurons), int(out_neurons)
                        ).float()
                    )

                else:
                    modules.append(nn.Linear(int(in_neurons), int(out_neurons)).float())
                    modules.append(activation_dict[activation]().float())

            in_neurons = out_neurons

    modules.append(nn.Linear(int(in_neurons), 2).float())

    return nn.Sequential(*modules)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


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


def shuffle_data(center_x, center_y, radius, initial, t, x, y, target):
    Data_num = np.arange(x.shape[0])
    np.random.shuffle(Data_num)

    return (
        center_x[Data_num],
        center_y[Data_num],
        radius[Data_num],
        initial[Data_num],
        t[Data_num],
        x[Data_num],
        y[Data_num],
        target[Data_num],
    )


def train_test_split(
    center_x,
    center_y,
    radius,
    initial,
    t,
    x,
    y,
    target,
    device,
    test_size=0.5,
    shuffle=True,
):
    with torch.no_grad():
        if shuffle:
            center_x, center_y, radius, initial, t, x, y, target = shuffle_data(
                center_x, center_y, radius, initial, t, x, y, target
            )

        if test_size < 1:
            train_ratio = len(x) - int(len(x) * test_size)
            center_x_train, center_x_test = (
                center_x[:train_ratio],
                center_x[train_ratio:],
            )
            center_y_train, center_y_test = (
                center_y[:train_ratio],
                center_y[train_ratio:],
            )
            radius_train, radius_test = radius[:train_ratio], radius[train_ratio:]
            initial_train, initial_test = initial[:train_ratio], initial[train_ratio:]
            t_train, t_test = t[:train_ratio], t[train_ratio:]
            x_train, x_test = x[:train_ratio], x[train_ratio:]
            y_train, y_test = y[:train_ratio], y[train_ratio:]
            target_train, target_test = target[:train_ratio], target[train_ratio:]
            return (
                center_x_train.requires_grad_(True).to(device),
                center_x_test.requires_grad_(True).to(device),
                center_y_train.requires_grad_(True).to(device),
                center_y_test.requires_grad_(True).to(device),
                radius_train.requires_grad_(True).to(device),
                radius_test.requires_grad_(True).to(device),
                initial_train.requires_grad_(True).to(device),
                initial_test.requires_grad_(True).to(device),
                t_train.requires_grad_(True).to(device),
                t_test.requires_grad_(True).to(device),
                x_train.requires_grad_(True).to(device),
                x_test.requires_grad_(True).to(device),
                y_train.requires_grad_(True).to(device),
                y_test.requires_grad_(True).to(device),
                target_train.requires_grad_(True).to(device),
                target_test.requires_grad_(True).to(device),
            )
        elif test_size in range(1, len(x)):
            center_x_train, center_x_test = (
                center_x[test_size:],
                center_x[:test_size],
            )
            center_y_train, center_y_test = (
                center_y[test_size:],
                center_y[:test_size],
            )
            radius_train, radius_test = radius[test_size:], radius[:test_size]
            initial_train, initial_test = initial[test_size:], initial[:test_size]
            t_train, t_test = t[test_size:], t[:test_size]
            x_train, x_test = x[test_size:], x[:test_size]
            y_train, y_test = y[test_size:], y[:test_size]
            target_train, target_test = target[test_size:], target[:test_size]
            return (
                center_x_train.requires_grad_(True).to(device),
                center_x_test.requires_grad_(True).to(device),
                center_y_train.requires_grad_(True).to(device),
                center_y_test.requires_grad_(True).to(device),
                radius_train.requires_grad_(True).to(device),
                radius_test.requires_grad_(True).to(device),
                initial_train.requires_grad_(True).to(device),
                initial_test.requires_grad_(True).to(device),
                t_train.requires_grad_(True).to(device),
                t_test.requires_grad_(True).to(device),
                x_train.requires_grad_(True).to(device),
                x_test.requires_grad_(True).to(device),
                y_train.requires_grad_(True).to(device),
                y_test.requires_grad_(True).to(device),
                target_train.requires_grad_(True).to(device),
                target_test.requires_grad_(True).to(device),
            )


def generate_training_points(num_points, device):
    center_x = torch.rand(num_points, 1, dtype=torch.float32)
    center_y = torch.rand(num_points, 1, dtype=torch.float32)
    radius = torch.rand(num_points, 1, dtype=torch.float32) * (0.2 - 0.1) + 0.1
    initial = torch.rand(num_points, 1, dtype=torch.float32) * (0.6 - 0.4) + 0.4
    t = torch.rand(num_points, 1, dtype=torch.float32) * 10
    x = torch.rand(num_points, 1, dtype=torch.float32)
    y = torch.rand(num_points, 1, dtype=torch.float32)

    return (
        center_x.requires_grad_(True).to(device),
        center_y.requires_grad_(True).to(device),
        radius.requires_grad_(True).to(device),
        initial.requires_grad_(True).to(device),
        t.requires_grad_(True).to(device),
        x.requires_grad_(True).to(device),
        y.requires_grad_(True).to(device),
    )


def generate_boundary_points(num_points, device):
    x_boundary = torch.tensor([0.0, 1], dtype=torch.float32).repeat(num_points // 2, 1)
    y_boundary = torch.rand(num_points, dtype=torch.float32)

    if torch.rand(1) > 0.5:
        x_boundary, y_boundary = y_boundary, x_boundary
        n = torch.tensor([[0.0, -1.0], [0.0, 1.0]], dtype=torch.float32).repeat(
            num_points // 2, 1
        )
    else:
        n = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32).repeat(
            num_points // 2, 1
        )

    return (
        x_boundary.view(-1, 1).requires_grad_(True).to(device),
        y_boundary.view(-1, 1).requires_grad_(True).to(device),
        n.requires_grad_(True).to(device),
    )


def generate_boundary_points(num_points, device):
    x_boundary = torch.tensor([0.0, 1], dtype=torch.float32).repeat(num_points // 2, 1)
    y_boundary = torch.rand(num_points, dtype=torch.float32)

    if torch.rand(1) > 0.5:
        x_boundary, y_boundary = y_boundary, x_boundary
        n = torch.tensor([[0.0, -1.0], [0.0, 1.0]], dtype=torch.float32).repeat(
            num_points // 2, 1
        )

    else:
        n = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32).repeat(
            num_points // 2, 1
        )

    return (
        x_boundary.view(-1, 1).requires_grad_(True).to(device),
        y_boundary.view(-1, 1).requires_grad_(True).to(device),
        n.requires_grad_(True).to(device),
    )


def initial_condition_points(data_input):

    center_x_tc= data_input[:,1]
    center_y_tc= data_input[:,2]
    radius_tc= data_input[:,3]
    initial_tc = data_input[:,0]
    x_tc= data_input[:,5]
    y_tc= data_input[:,6]

    # Calculate squared distances from each point to the circle centers
    squared_distances = (x_tc - center_x_tc) ** 2 + (y_tc - center_y_tc) ** 2

    # Create a mask for points inside the circle
    inside_circle_mask = squared_distances <= radius_tc**2

    # Initialize the tensor and set the values for points inside the circle
    C_init = torch.zeros((len(x_tc), 2),dtype=torch.float32)
    C_init[:, 1] = inside_circle_mask.ravel() * initial_tc.ravel()

    return C_init


def boundary_condition(
    model, device, center_x, center_y, radius, initial, t_b, x_b, y_b, n, Dn, X_nb, Db
):

    input_boundary = (
        torch.cat(
            [
                center_x,
                center_y,
                radius,
                initial,
                t_b,
                x_b,
                y_b,
            ],
            dim=1,
        )
        .to(device)
        .requires_grad_(True)
    )

    Cp, Cl = model(input_boundary).tensor_split(2, dim=1)

    del input_boundary
    nx, ny = n.tensor_split(2, dim=1)

    if nx[0].item() != 0:
        dCp_dx = torch.autograd.grad(
            Cp,
            x_b,
            grad_outputs=torch.ones_like(Cp),
            create_graph=True,
            retain_graph=True,
        )

        dCl_dx = torch.autograd.grad(
            Cl,
            x_b,
            grad_outputs=torch.ones_like(Cl),
            create_graph=True,
            retain_graph=True,
        )

        Cl_boundary = torch.mul(
            ((Dn * dCl_dx[0]) - X_nb * torch.mul(Cl, dCp_dx[0])), nx
        )

        Cp_boundary = torch.mul((Db * dCp_dx[0]), nx)

        return torch.cat([Cl_boundary, Cp_boundary], dim=1)

    else:
        dCp_dy = torch.autograd.grad(
            Cp,
            y_b,
            grad_outputs=torch.ones_like(Cp),
            create_graph=True,
            retain_graph=True,
        )

        dCl_dy = torch.autograd.grad(
            Cl,
            y_b,
            grad_outputs=torch.ones_like(Cl),
            create_graph=True,
            retain_graph=True,
        )

        Cl_boundary = torch.mul(
            ((Dn * dCl_dy[0]) - X_nb * torch.mul(Cl, dCp_dy[0])), ny
        )

        Cp_boundary = torch.mul((Db * dCp_dy[0]), ny)

        return torch.cat([Cl_boundary, Cp_boundary], dim=1)
    
def pde(
    model,
    center_x,
    center_y,
    radius,
    initial,
    t,
    x,
    y,
    cb,
    lambd_nb,
    Db,
    y_n,
    Cn_max,
    lambd_bn,
    mi_n,
    Dn,
    X_nb,
):

    Cl, Cp = model(
        torch.cat(
            [
                center_x,
                center_y,
                radius,
                initial,
                t,
                x,
                y,
            ],
            dim=1,
        )
    ).tensor_split(2, dim=1)

    # Calculating Cp value

    dCp_dx, dCp_dy = torch.autograd.grad(
        Cp,
        [x, y],
        grad_outputs=torch.ones_like(Cp),
        create_graph=True,
        retain_graph=True,
    )

    dCp_dx_2 = torch.autograd.grad(
        dCp_dx,
        x,
        grad_outputs=torch.ones_like(dCp_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCp_dy_2 = torch.autograd.grad(
        dCp_dy,
        y,
        grad_outputs=torch.ones_like(dCp_dy),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCp_dt = torch.autograd.grad(
        Cp,
        t,
        grad_outputs=torch.ones_like(Cp),
        create_graph=True,
        retain_graph=True,
    )[0]

    qb = cb * Cp
    rb = lambd_nb * torch.mul(Cl, Cp)

    Cp_eq = Db * (dCp_dx_2 + dCp_dy_2) - rb + qb - dCp_dt

    # Calculating Cl value

    dCl_dx, dCl_dy = torch.autograd.grad(
        Cl,
        [x, y],
        grad_outputs=torch.ones_like(Cl),
        create_graph=True,
        retain_graph=True,
    )

    dCl_dx_2 = torch.autograd.grad(
        dCl_dx,
        x,
        grad_outputs=torch.ones_like(dCl_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCl_dy_2 = torch.autograd.grad(
        dCl_dy,
        y,
        grad_outputs=torch.ones_like(dCl_dy),
        create_graph=True,
        retain_graph=True,
    )[0]

    dCl_dt = torch.autograd.grad(
        Cl,
        t,
        grad_outputs=torch.ones_like(Cl),
        create_graph=True,
        retain_graph=True,
    )[0]

    qn = y_n * torch.mul(Cp, (Cn_max - Cl))
    rn = lambd_bn * torch.mul(Cl, Cp) + mi_n * Cl

    Cl_eq = (
        Dn * (dCl_dx_2 + dCl_dy_2)
        - X_nb
        * (
            (torch.mul(dCl_dx, dCp_dx) + torch.mul(Cl, dCp_dx_2))
            + (torch.mul(dCl_dy, dCp_dy) + torch.mul(Cl, dCp_dy_2))
        )
        - rn
        + qn
    ) - dCl_dt

    return torch.cat([Cl_eq, Cp_eq], dim=1)


class train:
    def __init__(
        self,
        n_epochs,
        batch_size,
        decay_rate,
        model,
        center_x_tc,
        center_y_tc,
        radius_tc,
        initial_tc,
        t_tc,
        x_tc,
        y_tc,
        target,
        device,
        n_points,
        constant_properties,
        norm_weights=None,
        validation=None,
        tolerance=None,
        patience=10,
    ):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.model = model.to(device)
        self.device = device
        self.n_points = n_points
        self.constant_properties = constant_properties
        self.norm_weights = norm_weights
        self.validation = validation
        self.tolerance = tolerance
        self.patience = patience

        if self.validation:
            (
                self.center_x_train,
                self.center_x_test,
                self.center_y_train,
                self.center_y_test,
                self.radius_train,
                self.radius_test,
                self.initial_train,
                self.initial_test,
                self.t_train,
                self.t_test,
                self.x_train,
                self.x_test,
                self.y_train,
                self.y_test,
                self.target_train,
                self.target_test,
            ) = train_test_split(
                center_x_tc,
                center_y_tc,
                radius_tc,
                initial_tc,
                t_tc,
                x_tc,
                y_tc,
                target,
                device,
                test_size=self.validation,
            )

        else:
            self.center_x_train = center_x_tc.to(device)
            self.center_x_test = None
            self.center_y_train = center_y_tc.to(device)
            self.center_y_test = None
            self.radius_train = radius_tc.to(device)
            self.radius_test = None
            self.initial_train = initial_tc.to(device)
            self.initial_test = None
            self.t_train = t_tc.to(device)
            self.t_test = None
            self.x_train = x_tc.to(device)
            self.x_test = None
            self.y_train = y_tc.to(device)
            self.y_test = None
            self.target_train = target.to(device)
            self.target_test = None

        self.test_data = (
            torch.cat(
                [
                    self.center_x_test,
                    self.center_y_test,
                    self.radius_test,
                    self.initial_test,
                    self.t_test,
                    self.x_test,
                    self.y_test,
                ],
                dim=1,
            )
            .requires_grad_(True)
            .to(device)
        )

        pass

    def loss_func(
        self,
    ):
        self.optimizer.zero_grad()

        self.batch = torch.cat(
            [
                self.center_x_train,
                self.center_y_train,
                self.radius_train,
                self.initial_train,
                self.t_train,
                self.x_train,
                self.y_train,
            ],
            dim=1,
        )[self.i : self.i + self.batch_size, :]

        C_initial_batch = initial_condition_points(self.batch).to(self.device)

        # Computing intial loss
        t_initial = torch.zeros((self.batch.shape[0], 1), dtype=torch.float32).to(
            self.device
        )

        mesh_ini = torch.cat(
            [self.batch[:, :5], t_initial, self.batch[:, 6:]],
            dim=1,
        )

        C_initial_pred = self.model(mesh_ini)

        self.loss_initial = self.criterion(C_initial_batch, C_initial_pred)

        # Computing pde loss

        center_x, center_y, radius, initial, t, x, y = generate_training_points(
            self.n_points, self.device
        )

        predicted_pde = pde(
            self.model,
            center_x,
            center_y,
            radius,
            initial,
            t,
            x,
            y,
            self.constant_properties["cb"],
            self.constant_properties["lambd_nb"],
            self.constant_properties["Db"],
            self.constant_properties["y_n"],
            self.constant_properties["Cn_max"],
            self.constant_properties["lambd_bn"],
            self.constant_properties["mi_n"],
            self.constant_properties["Dn"],
            self.constant_properties["X_nb"],
        )

        self.loss_pde = self.criterion(
            predicted_pde,
            torch.zeros_like(predicted_pde),
        )

        # Computing boundary loss

        x_bnd, y_bnd, n_bnd = generate_boundary_points(self.n_points, self.device)

        predicted_boundary = boundary_condition(
            self.model,
            self.device,
            center_x,
            center_y,
            radius,
            initial,
            t,
            x_bnd,
            y_bnd,
            n_bnd,
            self.constant_properties["Dn"],
            self.constant_properties["X_nb"],
            self.constant_properties["Db"],
        )

        self.loss_boundary = self.criterion(
            predicted_boundary,
            torch.zeros_like(predicted_boundary),
        )

        # Computing data loss

        C_pred = self.model(self.batch.to(self.device))

        self.loss_data = self.criterion(
            C_pred, self.target_train[self.i : self.i + self.batch_size, :]
        )

        del C_pred

        self.loss = (
            10 * self.loss_initial
            + self.loss_pde
            + self.loss_boundary
            + self.loss_data * 10
        )

        self.loss.backward()

        return self.loss

    def execute(
        self,
    ):
        self.criterion = nn.MSELoss()

        dt_min, dt_max = self.norm_weights if self.norm_weights else (0, 1)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-7)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=self.decay_rate
        )
        C_pde_loss_it = torch.zeros(self.n_epochs)
        C_data_loss_it = torch.zeros(self.n_epochs)
        C_boundary_loss_it = torch.zeros(self.n_epochs)
        C_initial_loss_it = torch.zeros(self.n_epochs)
        val_loss_it = torch.zeros(self.n_epochs)

        patience_count = 0
        val_loss = torch.tensor([1000])

        for epoch in range(self.n_epochs):
            for bt, self.i in enumerate(range(0, len(self.x_train), self.batch_size)):

                self.optimizer.step(self.loss_func)

            self.lr_scheduler.step()

            # Computing validation loss

            if self.validation:
                with torch.no_grad():
                    val_old = val_loss
                    val_loss = self.criterion(
                        self.target_test, self.model(self.test_data)
                    )

            C_pde_loss_it[epoch] = self.loss_pde.item()
            C_boundary_loss_it[epoch] = self.loss_boundary.item()
            C_initial_loss_it[epoch] = self.loss_initial.item()
            C_data_loss_it[epoch] = self.loss_data.item()
            val_loss_it[epoch] = val_loss.item() if self.validation else 0

            if ((epoch + 1) % 10) == 0 or (epoch == 0):
                print(
                    f"Finished epoch {epoch+1}, latest loss {self.loss}, validation loss {val_loss.item()}"
                    if self.validation
                    else f"Finished epoch {epoch+1}, latest loss {self.loss}"
                )

            if self.tolerance:

                if (
                    abs(val_old.item() - val_loss.item()) / val_old.item()
                    < self.tolerance
                ):
                    patience_count += 1

                else:
                    patience_count = 0

                if patience_count >= self.patience:

                    C_pde_loss_it = C_pde_loss_it[:epoch]
                    C_boundary_loss_it = C_boundary_loss_it[:epoch]
                    C_initial_loss_it = C_initial_loss_it[:epoch]
                    C_data_loss_it = C_data_loss_it[:epoch]
                    val_loss_it = val_loss_it[:epoch]

                    print("Early break!")

                    break

        return (
            self.model,
            C_pde_loss_it,
            C_boundary_loss_it,
            C_initial_loss_it,
            C_data_loss_it,
            val_loss_it,
        )