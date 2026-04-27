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

    print(center, radius, central_ini_cond)

    (
        initial_tc,
        center_x_tc,
        radius_tc,
        data_tc,
        target,
        reduced_data_tc,
        reduced_target,
        device,
    ) = allocates_training_mesh(
        t_dom,
        x_dom,
        size_t,
        size_x,
        center[0],
        central_ini_cond,
        radius,
        Cl_fvm,
        Cp_fvm,
        samples_percent=0.05,
    )

    n_epochs = int(1e4)

    batch_size = int(len(reduced_data_tc) / 10)

    pinn_batch = int(len(reduced_data_tc) * 100)

    dtype = torch.float64

    model = generate_model(arch_str, 2, 2)

    pinn_file = "beta1_{}__beta2_{}".format(beta1, beta2) + arch_str

    print("\n" + pinn_file)

    print("=" * 20)

    print(
        "Number of parameters",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

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

    init_loss_cl = LOSS(
        device=device,
        name="Inital",
        batch_size=pinn_batch,
        criterium="MSE",
    )

    init_loss_cl.setBatchGenerator(
        generate_initial_points, center_x_tc, radius_tc, initial_tc
    )

    init_loss_cl.setEvalFunction(
        initial_condition_cl,
        center_x_tc,
        radius_tc,
        initial_tc,
        device,
    )

    trainer.add_loss(init_loss_cl, 10)

    init_loss_cp = LOSS(
        device=device,
        name="Inital",
        batch_size=pinn_batch,
        criterium="MSE",
    )

    init_loss_cp.setBatchGenerator(
        generate_initial_points, center_x_tc, radius_tc, initial_tc
    )

    init_loss_cp.setEvalFunction(initial_condition_cp, device)

    trainer.add_loss(init_loss_cp, 10)

    bnd_loss_cl = LOSS(
        device=device,
        name="Boundary Cl",
        batch_size=pinn_batch,
        criterium="MSE",
    )

    bnd_loss_cl.setBatchGenerator(generate_boundary_points, t_dom[1])

    bnd_loss_cl.setEvalFunction(boundary_condition_cl, Dn, X_nb, device)

    trainer.add_loss(bnd_loss_cl)

    bnd_loss_cp = LOSS(
        device=device,
        name="Boundary",
        batch_size=pinn_batch,
        criterium="MSE",
    )

    bnd_loss_cp.setBatchGenerator(generate_boundary_points, t_dom[1])

    bnd_loss_cp.setEvalFunction(boundary_condition_cp, Dn, device)

    trainer.add_loss(bnd_loss_cp)

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
    )

    trainer.add_loss(pde_cp_loss)

    reduced_data_tc = reduced_data_tc.detach()
    reduced_target = reduced_target.detach()

    idx = torch.randperm(reduced_data_tc.shape[0])

    reduced_data_tc_sh = reduced_data_tc[idx].clone()
    reduced_target_sh = reduced_target[idx].clone()

    data_loss = LOSS(
        device,
        name="Data Loss",
        batch_size=batch_size,
        criterium="MSE",
    )

    data_loss.add_data(
        reduced_data_tc_sh,
        reduced_target_sh,
    )

    trainer.add_loss(data_loss)

    model, loss_dict = trainer.train()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    cwd = os.getcwd()

    torch.save(model.state_dict(), cwd + "/nn_parameters/" + pinn_file + ".pt")

    with open("learning_curves/" + pinn_file + ".pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(loss_dict, openfile)

    del model
    del trainer
