import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle as pk
import argparse
import glob
import sys


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
                    modules.append(activation_dict[activation](3, int(out_neurons)))

                else:
                    modules.append(nn.Linear(3, int(out_neurons)))
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


def gerenate_training_points(num_points, device):
    x = torch.rand(num_points, 1).to(device)
    y = torch.rand(num_points, 1).to(device)
    t = torch.rand(num_points, 1).to(device) * 10

    return x.requires_grad_(True), y.requires_grad_(True), t.requires_grad_(True)


def gerenate_boundary_points(num_points, device):
    x_boundary = torch.tensor([0.0, 1.0]).repeat(num_points // 2)
    y_boundary = torch.rand(num_points)

    if torch.rand(1) > 0.5:
        x_boundary, y_boundary = y_boundary, x_boundary
        n = torch.tensor([[0.0, -1.0], [0.0, 1.0]]).repeat(num_points // 2, 1)
    else:
        n = torch.tensor([[-1.0, 0.0], [1.0, 0.0]]).repeat(num_points // 2, 1)

    return (
        x_boundary.to(device).requires_grad_(True).view(-1, 1),
        y_boundary.to(device).requires_grad_(True).view(-1, 1),
        n.to(device),
    )


def initial_condition(x, y):

    Cl = torch.full_like(x, 0)

    Cp = torch.full_like(x, 0)

    for i, (xx, yy) in enumerate(zip(x, y)):
        if ((xx >= 0.5) and (xx <= 0.6)) and ((yy >= 0.5) and (yy <= 0.6)):
            Cp[i] = 0.2

    return torch.cat([Cl, Cp], dim=1)


def boundary_condition(t_b, x_b, y_b, n, model, Dn, X_nb, Db):

    input_data = torch.cat([t_b, x_b, y_b], dim=1)

    Cp, Cl = model(input_data).tensor_split(2, dim=1)

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


def pde(model, t, x, y, cb, lambd_nb, Db, y_n, Cn_max, lambd_bn, mi_n, Dn, X_nb):

    Cl, Cp = model(torch.cat([t, x, y], dim=1)).tensor_split(2, dim=1)

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
        device,
        data_input,
        t,
        x,
        y,
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
        self.data_input = data_input
        self.t = t
        self.x = x
        self.y = y
        self.n_points = n_points
        self.constant_properties = constant_properties
        self.norm_weights = norm_weights
        self.validation = validation
        self.tolerance = tolerance
        self.patience = patience
        pass

    def loss_func(
        self,
    ):
        # Computing intial loss
        t_initial = torch.zeros_like(self.train_t[self.i : self.i + self.batch_size])

        mesh_ini = torch.cat(
            [
                t_initial,
                self.train_x[self.i : self.i + self.batch_size],
                self.train_y[self.i : self.i + self.batch_size],
            ],
            dim=1,
        )

        C_initial_pred = model(mesh_ini)

        self.loss_initial = self.criterium(
            self.C_initial[self.i : self.i + self.batch_size], C_initial_pred
        )

        # Computing pde loss

        x_pde, y_pde, t_pde = gerenate_training_points(self.n_points, device)

        predicted_pde = pde(
            model,
            t_pde,
            x_pde,
            y_pde,
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

        self.loss_pde = self.criterium(
            predicted_pde,
            torch.zeros_like(predicted_pde),
        )

        # Computing boundary loss

        x_bnd, y_bnd, n_bnd = gerenate_boundary_points(self.n_points, device)

        predicted_boundary = boundary_condition(
            t_pde,
            x_bnd,
            y_bnd,
            n_bnd,
            model,
            self.constant_properties["Dn"],
            self.constant_properties["X_nb"],
            self.constant_properties["Db"],
        )

        self.loss_boundary = self.criterium(
            predicted_boundary,
            torch.zeros_like(predicted_boundary),
        )

        # Computing data loss

        C_pred = model(self.train_data_input[self.i : self.i + self.batch_size])

        self.loss_data = self.criterium(
            self.train_data[self.i : self.i + self.batch_size], C_pred
        )

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
        self.criterium = nn.MSELoss()

        dt_min, dt_max = self.norm_weights if self.norm_weights else (0, 1)

        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=self.decay_rate
        )

        if self.validation:
            (
                self.train_data,
                self.test_data,
                self.train_t,
                self.test_t,
                train_dom,
                test_dom,
            ) = train_test_split(
                self.data_input,
                self.t,
                torch.cat([self.x, self.y], dim=1),
                test_size=self.validation,
            )
            self.train_x, self.train_y = train_dom.split(1, dim=1)
            self.test_x, self.test_y = test_dom.split(1, dim=1)
            self.train_data_input = torch.cat(
                [self.train_t, self.train_x, self.train_y], dim=1
            )
            self.test_data_input = torch.cat(
                [self.test_t, self.test_x, self.test_y], dim=1
            )

        else:
            self.train_data = data_input
            self.test_data_input = None
            self.train_t = self.t
            self.test_t = None
            self.train_x = self.x
            self.test_x = None
            self.train_y = self.y
            self.test_y = None
            self.test_data = None
            self.train_data_input = torch.cat(
                [self.t, self.train_x, self.train_y], dim=1
            )

        C_pde_loss_it = torch.zeros(self.n_epochs).to(device)
        C_data_loss_it = torch.zeros(self.n_epochs).to(device)
        C_boundary_loss_it = torch.zeros(self.n_epochs).to(device)
        C_initial_loss_it = torch.zeros(self.n_epochs).to(device)
        self.C_initial = initial_condition(self.train_x, self.train_y).to(device)
        val_loss_it = torch.zeros(self.n_epochs).to(device)

        patience_count = 0
        val_loss = torch.tensor([1000])

        for epoch in range(self.n_epochs):
            for self.i in range(0, len(self.train_t), self.batch_size):

                self.optimizer.zero_grad()

                self.optimizer.step(self.loss_func)
                self.lr_scheduler.step()

            # Computing validation loss

            if self.validation:
                with torch.no_grad():
                    val_old = val_loss
                    val_loss = self.criterium(
                        self.test_data, model(self.test_data_input)
                    )

            C_pde_loss_it[epoch] = self.loss_pde.item()
            C_boundary_loss_it[epoch] = self.loss_boundary.item()
            C_initial_loss_it[epoch] = self.loss_initial.item()
            C_data_loss_it[epoch] = self.loss_data.item()
            val_loss_it[epoch] = val_loss.item() if self.validation else 0

            if ((epoch + 1) % 100) == 0 or (epoch == 0):
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
            model,
            C_pde_loss_it,
            C_boundary_loss_it,
            C_initial_loss_it,
            C_data_loss_it,
            val_loss_it,
        )


if __name__ == "__main__":

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
        default=None,
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

    h = param_dict["h"]
    k = param_dict["k"]
    Db = param_dict["Db"]
    Dn = param_dict["Dn"]
    phi = param_dict["phi"]
    ksi = param_dict["ksi"]
    cb = param_dict["cb"]
    lambd_nb = param_dict["lambd_nb"]
    mi_n = param_dict["mi_n"]
    lambd_bn = param_dict["lambd_bn"]
    y_n = param_dict["y_n"]
    Cn_max = param_dict["Cn_max"]
    X_nb = param_dict["X_nb"]
    x_dom_min = param_dict["x_dom_min"]
    x_dom_max = param_dict["x_dom_max"]
    y_dom_min = param_dict["y_dom_min"]
    y_dom_max = param_dict["y_dom_max"]
    t_dom_min = param_dict["t_dom_min"]
    t_dom_max = param_dict["t_dom_max"]

    pinn_file = "epochs_{}__batch_{}__arch_".format(n_epochs, batch_size) + arch_str

    # Check if has already run

    pinn_sim_done = list(
        map(lambda txt: txt.split("/")[-1].split(".")[0], glob.glob("pinn_sim/*"))
    )

    if pinn_file in pinn_sim_done:
        sys.exit(404)

    size_x = int(((x_dom_max - x_dom_min) / (h))) + 1
    size_y = int(((y_dom_max - y_dom_min) / (h))) + 1
    size_t = int(((t_dom_max - t_dom_min) / (k))) + 1

    with open("fdm_sim/Cp__" + struct_name + ".pkl", "rb") as f:
        Cb_fdm = pk.load(f)

    with open("fdm_sim/Cl__" + struct_name + ".pkl", "rb") as f:
        Cn_fdm = pk.load(f)

    t_np = np.linspace(
        t_dom_min, t_dom_max, num=size_t, endpoint=True, dtype=np.float32
    )
    x_np = np.linspace(
        x_dom_min, x_dom_max, num=size_x, endpoint=True, dtype=np.float32
    )
    y_np = np.linspace(
        y_dom_min, y_dom_max, num=size_y, endpoint=True, dtype=np.float32
    )

    xx, tt, yy = np.meshgrid(
        x_np,
        t_np,
        y_np,
    )

    data_input_np = np.array([Cn_fdm.flatten(), Cb_fdm.flatten()]).T

    if torch.cuda.is_available():
        device = torch.device("cuda")
        t_tc = (
            torch.tensor(tt, dtype=torch.float32, requires_grad=True)
            .reshape(-1, 1)
            .to(device)
        )
        x_tc = (
            torch.tensor(xx, dtype=torch.float32, requires_grad=True)
            .reshape(-1, 1)
            .to(device)
        )
        y_tc = (
            torch.tensor(yy, dtype=torch.float32, requires_grad=True)
            .reshape(-1, 1)
            .to(device)
        )
        data_input = torch.tensor(data_input_np, dtype=torch.float32).to(device)

    else:
        device = torch.device("cpu")
        t_tc = torch.tensor(tt, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
        x_tc = torch.tensor(xx, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
        y_tc = torch.tensor(yy, dtype=torch.float32, requires_grad=True).reshape(-1, 1)
        data_input = torch.tensor(data_input_np, dtype=torch.float32)

    print(device)

    del xx
    del yy
    del tt

    decay_rate = 0.999

    constant_properties = {
        "h": h,
        "k": k,
        "Db": Db,
        "Dn": Dn,
        "phi": phi,
        "ksi": ksi,
        "cb": cb,
        "lambd_nb": lambd_nb,
        "mi_n": mi_n,
        "lambd_bn": lambd_bn,
        "y_n": y_n,
        "Cn_max": Cn_max,
        "X_nb": X_nb,
    }

    trainer = train(
        n_epochs=n_epochs,
        batch_size=batch_size,
        decay_rate=decay_rate,
        model=model,
        device=device,
        data_input=data_input,
        x=x_tc,
        y=y_tc,
        t=t_tc,
        n_points=batch_size,
        constant_properties=constant_properties,
        validation=0.1,
        tolerance=0.001,
        patience=60,
    )

    (
        model,
        C_pde_loss_it,
        C_boundary_loss_it,
        C_initial_loss_it,
        C_data_loss_it,
        val_loss_it,
    ) = trainer.execute()

    with open("learning_curves/C_pde_loss_it__" + pinn_file + ".pkl", "wb") as f:
        pk.dump(C_pde_loss_it.cpu().numpy(), f)

    with open("learning_curves/C_data_loss_it__" + pinn_file + ".pkl", "wb") as f:
        pk.dump(C_data_loss_it.cpu().numpy(), f)

    with open("learning_curves/C_boundary_loss_it__" + pinn_file + ".pkl", "wb") as f:
        pk.dump(C_boundary_loss_it.cpu().numpy(), f)

    with open("learning_curves/C_initial_loss_it__" + pinn_file + ".pkl", "wb") as f:
        pk.dump(C_initial_loss_it.cpu().numpy(), f)

    with open("learning_curves/val_loss_it__" + pinn_file + ".pkl", "wb") as f:
        pk.dump(val_loss_it.cpu().numpy(), f)

    torch.set_num_threads(1)

    with open("fdm_sim/time__" + struct_name + ".pkl", "rb") as f:
        time_fdm = pk.load(f)

    model_cpu = model.to("cpu")

    speed_up = []

    mesh = torch.cat([t_tc, x_tc, y_tc], dim=1).to("cpu")

    torch.set_num_threads(1)

    for i in range(len(time_fdm)):

        pinn_start = time.time()

        with torch.no_grad():
            Cl_pinn, Cp_pinn = model_cpu(mesh).split(1, dim=1)

        pinn_end = time.time()

        pinn_time = pinn_end - pinn_start

        speed_up.append(time_fdm[i] / pinn_time)

    mean_speed_up = np.mean(speed_up)
    std_speed_up = np.std(speed_up)

    rmse = np.mean(
        [
            ((Cl_p[0] - Cl_f) ** 2 + (Cp_p[0] - Cp_f) ** 2) ** 0.5
            for Cl_p, Cp_p, Cl_f, Cp_f in zip(
                Cl_pinn, Cp_pinn, Cn_fdm.flatten(), Cb_fdm.flatten()
            )
        ]
    )

    max_ae = np.max(
        [
            [((Cl_p[0] - Cl_f) ** 2) ** 0.5, ((Cp_p[0] - Cp_f) ** 2) ** 0.5]
            for Cl_p, Cp_p, Cl_f, Cp_f in zip(
                Cl_pinn, Cp_pinn, Cn_fdm.flatten(), Cb_fdm.flatten()
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

    with open("pinn_sim/" + pinn_file + ".pkl", "wb") as f:
        pk.dump(output, f)
