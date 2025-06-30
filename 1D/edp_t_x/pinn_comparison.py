import pickle as pk
import os
import json
from pinn import *
import argparse

dtype = torch.float32

arch_str = args_dict["arch_str"]

beta1 = args_dict["beta1"]

beta2 = args_dict["beta2"]

n_epochs = int(1e4)

batch_size = int(1e4)

hidden_layer = [int(n_neurons) for n_neurons in arch_str.split("__")[1:]]

arch_str = ""

for hd in hidden_layer:
    arch_str += "__" + str(hd)

# Opening JSON file
with open("control_dicts/constant_properties.json", "r") as openfile:
    # Reading from json file
    constant_properties = json.load(openfile)

Db = constant_properties["Db"]
Dn = constant_properties["Dn"]
phi = constant_properties["phi"]
cb = constant_properties["cb"]
lambd_nb = constant_properties["lambd_nb"]
mi_n = constant_properties["mi_n"]
lambd_bn = constant_properties["lambd_bn"]
y_n = constant_properties["y_n"]
Cn_max = constant_properties["Cn_max"]
X_nb = constant_properties["X_nb"]
central_ini_cond = constant_properties["central_ini_cond"]

# Opening JSON file
with open("control_dicts/mesh_properties.json", "r") as openfile:
    # Reading from json file
    mesh_properties = json.load(openfile)

h = mesh_properties["h"]
k = mesh_properties["k"]
x_dom = mesh_properties["x_dom"]
y_dom = mesh_properties["y_dom"]
t_dom = mesh_properties["t_dom"]

Cl_list, Cp_list, speed_up_list = read_files("fvm_sim")

Cp_fvm, Cl_fvm, center, radius = format_array(Cp_list[0], Cl_list[0])

size_x, size_y, size_t = get_mesh_properties(x_dom, y_dom, t_dom, h, k)

with open("source_points/lymph_vessels.pkl", "rb") as f:
    leu_source_points = pk.load(f)

(
    initial_tc,
    center_x_tc,
    radius_tc,
    data_tc,
    src_tc,
    target,
    device,
) = allocates_training_mesh(
    t_dom,
    x_dom,
    size_t,
    size_x,
    center[0],
    central_ini_cond,
    radius,
    Cp_fvm,
    Cl_fvm,
    leu_source_points,
)

model = FullyConnectedNetwork(2, 2, hidden_layer, dtype=dtype)

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(beta1, beta2))

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.9999,
    patience=1000,
    threshold=1e-3,
    threshold_mode="rel",
    cooldown=0,
    min_lr=1e-5,
    eps=1e-08,
)

trainer_pinn = Trainer(
    n_epochs=n_epochs,
    batch_size=batch_size,
    model=model,
    device=device,
    target=target,
    data=data_tc,
    patience=5000,
    tolerance=0.01,
    validation=0.2,
    optimizer=optimizer,
    scheduler=lr_scheduler,
    print_steps=1e10,
    constant_properties=constant_properties,
)

init_loss = LOSS_INITIAL(
    batch_size=batch_size,
    device=device,
    loss="RMSE",
    name="LossInital",
)

init_loss.setBatchGenerator(generate_initial_points, center_x_tc, radius_tc, initial_tc)

trainer_pinn.add_loss(init_loss)

bnd_loss = LOSS_PINN(
    batch_size=batch_size,
    device=device,
    loss="RMSE",
    name="LossBoundary",
)

bnd_loss.setBatchGenerator(generate_boundary_points)

bnd_loss.setPinnFunction(boundary_condition, Dn, X_nb, Db, device)

trainer_pinn.add_loss(bnd_loss)

pde_loss = LOSS_PINN(
    batch_size=batch_size,
    device=device,
    loss="RMSE",
    name="LossPDE",
)

pde_loss.setBatchGenerator(generate_pde_points)

original_source = torch.tensor(leu_source_points).to(device)

pde_loss.setPinnFunction(
    pde,
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
)

trainer_pinn.add_loss(pde_loss, 5)

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(beta1, beta2))

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.9999,
    patience=1000,
    threshold=1e-3,
    threshold_mode="rel",
    cooldown=0,
    min_lr=1e-5,
    eps=1e-08,
)

trainer_nn = Trainer(
    n_epochs=n_epochs,
    batch_size=batch_size,
    model=model,
    device=device,
    target=target,
    data=data_tc,
    patience=5000,
    tolerance=0.01,
    validation=0.2,
    optimizer=optimizer,
    scheduler=lr_scheduler,
    print_steps=1e10,
    constant_properties=constant_properties,
)

if __name__ == "__main__":

    model_pinn, loss_dict = trainer_pinn.train()

    model_nn, loss_dict = trainer_pinn.train()
