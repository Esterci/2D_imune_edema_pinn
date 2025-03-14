import numpy
import torch
import pickle as pk
import os
import json
from pinn import *
import argparse

# Parsing model parameters

parser = argparse.ArgumentParser(description="", add_help=False)
parser = argparse.ArgumentParser()


parser.add_argument(
    "-d",
    "--decay_rate",
    type=float,
    action="store",
    dest="decay_rate",
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
    "-l",
    "--lr_rate",
    type=float,
    action="store",
    dest="lr_rate",
    required=True,
    default=None,
    help="",
)


if __name__ == "__main__":

    args = parser.parse_args()

    args_dict = vars(args)

    decay_rate = args_dict["decay_rate"]
    arch_str = args_dict["arch_str"]
    lr_rate = args_dict["lr_rate"]

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

    (
        initial_tc,
        center_x_tc,
        center_y_tc,
        radius_tc,
        t_tc,
        x_tc,
        y_tc,
        target,
        reduced_t_tc,
        reduced_x_tc,
        reduced_y_tc,
        reduced_target,
        device,
    ) = allocates_training_mesh(
        t_dom,
        x_dom,
        y_dom,
        size_t,
        size_x,
        size_y,
        center[0],
        center[1],
        central_ini_cond,
        radius,
        Cp_fvm,
        Cl_fvm,
        3000,
    )

    model = generate_model(arch_str).to(device).apply(init_weights)

    print(model)

    val = 0.2
    batch_size = int(len(t_tc) * (1 - val) / 10)

    trainer = train(
        n_epochs=300,
        batch_size=batch_size,
        decay_rate=decay_rate,
        model=model,
        initial_tc=initial_tc,
        center_x_tc=center_x_tc,
        center_y_tc=center_y_tc,
        radius_tc=radius_tc,
        t_tc=reduced_t_tc,
        x_tc=reduced_x_tc,
        y_tc=reduced_y_tc,
        target=reduced_target,
        device=device,
        n_points=batch_size,
        constant_properties=constant_properties,
        validation=val,
        tolerance=0.02,
        patience=20,
        normalize=True,
        lr_rate=lr_rate,
    )

    (
        model,
        C_pde_loss_it,
        C_boundary_loss_it,
        C_initial_loss_it,
        C_data_loss_it,
        val_loss_it,
    ) = trainer.execute()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    pinn_file = (
        "decay_rates_{:.4}__lr_rates_{:.4}__arch_".format(decay_rate, lr_rate)
        + arch_str
    )

    cwd = os.getcwd()

    torch.save(model.state_dict(), cwd + "/nn_parameters/" + pinn_file + ".pt")

    with open("learning_curves/C_pde_loss_it_" + pinn_file + ".pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(C_pde_loss_it.cpu().numpy(), openfile)

    with open(
        "learning_curves/C_boundary_loss_it_" + pinn_file + ".pkl", "wb"
    ) as openfile:
        # Reading from json file
        pk.dump(C_boundary_loss_it.cpu().numpy(), openfile)

    with open(
        "learning_curves/C_initial_loss_it_" + pinn_file + ".pkl", "wb"
    ) as openfile:
        # Reading from json file
        pk.dump(C_initial_loss_it.cpu().numpy(), openfile)

    with open("learning_curves/C_data_loss_it_" + pinn_file + ".pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(C_data_loss_it.cpu().numpy(), openfile)

    with open("learning_curves/val_loss_it_" + pinn_file + ".pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(val_loss_it.cpu().numpy(), openfile)

    del model
    del trainer
