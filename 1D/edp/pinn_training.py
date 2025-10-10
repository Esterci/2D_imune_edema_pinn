import pickle as pk
import os
import json
from pinn import *
import argparse

# Parsing model parameters

parser = argparse.ArgumentParser(description="", add_help=False)

parser = argparse.ArgumentParser()

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
    "-b1",
    "--beta1",
    type=float,
    action="store",
    dest="beta1",
    required=True,
    default=None,
    help="",
)

parser.add_argument(
    "-b2",
    "--beta2",
    type=float,
    action="store",
    dest="beta2",
    required=True,
    default=None,
    help="",
)


if __name__ == "__main__":

    args = parser.parse_args()

    args_dict = vars(args)

    arch_str = args_dict["arch_str"]

    beta1 = args_dict["beta1"]

    beta2 = args_dict["beta2"]

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

    n_epochs = int(1e4)

    batch_size = int(1.2e3)

    dtype = torch.float64

    model = generate_model(arch_str, 2, 2)

    pinn_file = "beta1_{}__beta2_{}".format(beta1, beta2) + arch_str

    print("\n" + pinn_file)

    print("=" * 20)

    print(
        "Number of parameters",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(beta1, beta2))

    trainer = Trainer(
        n_epochs=n_epochs,
        batch_size=batch_size,
        model=model,
        device=device,
        # target=target,
        # data=data_tc,
        patience=5000,
        tolerance=0.01,
        # validation=0.2,
        optimizer=optimizer,
        print_steps=1e3,
    )

    init_loss = LOSS(
        device=device,
        name="Inital",
        batch_size=batch_size,
    )

    init_loss.setBatchGenerator(
        generate_initial_points, center_x_tc, radius_tc, initial_tc
    )

    init_loss.setEvalFunction(initial_condition, device)

    trainer.add_loss(init_loss, 10)

    bnd_loss = LOSS(
        device=device,
        name="Boundary",
        batch_size=batch_size,
    )

    bnd_loss.setBatchGenerator(generate_boundary_points, t_dom[1])

    bnd_loss.setEvalFunction(boundary_condition, Dn, X_nb, Db, device)

    trainer.add_loss(bnd_loss)

    pde_loss = LOSS(
        device=device,
        name="PDE",
        batch_size=batch_size,
    )

    pde_loss.setBatchGenerator(generate_pde_points, t_dom[1])

    pde_loss.setEvalFunction(
        pde, h, cb, phi, lambd_nb, Db, y_n, Cn_max, lambd_bn, mi_n, Dn, X_nb, device
    )

    trainer.add_loss(pde_loss)

    model, loss_dict = trainer.train()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    cwd = os.getcwd()

    torch.save(model.state_dict(), cwd + "/nn_parameters/" + pinn_file + ".pt")

    with open("learning_curves/comp.pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(loss_dict, openfile)

    del model
    del trainer
