import pickle as pk
import os
import json
from pinn import *
from utils import *
import argparse
from sklearn.model_selection import KFold
import numpy as np
import copy

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

parser.add_argument(
    "-n",
    "--pinn_name",
    type=str,
    action="store",
    dest="pinn_name",
    required=True,
    default=None,
    help="",
)


def main():

    args = parser.parse_args()

    args_dict = vars(args)

    arch_str = args_dict["arch_str"]

    beta1 = args_dict["beta1"]

    beta2 = args_dict["beta2"]

    pinn_name = args_dict["pinn_name"]

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

    reference_time = read_speed_ups(speed_up_list)[0]["serial_time"]

    print(center, radius, central_ini_cond)

    (
        initial_tc,
        center_x_tc,
        radius_tc,
        data_tc,
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
        Cl_fvm,
        Cp_fvm,
    )

    n_epochs = int(1e4)

    batch_size = int(len(data_tc) / 10)

    pinn_batch = int(1e4)

    k_folds = 5

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    pinn_error = []
    nn_error = []

    pinn_train_time = []
    pinn_test_time = []
    pinn_speed_up = []

    nn_train_time = []
    nn_test_time = []
    nn_speed_up = []

    pinn_min_error = np.inf
    nn_min_error = np.inf

    pinn_best_loss_dict = None
    nn_best_loss_dict = None

    pinn_best_param = None
    nn_best_param = None

    pinn_best_pred = None
    nn_best_pred = None

    for fold, (train_idx, test_idx) in enumerate(kfold.split(data_tc)):

        print("\n" + "=" * 30)
        print(f"Fold {fold + 1}/{k_folds}")
        print("=" * 30)

        data_train = data_tc[train_idx].clone().detach()
        target_train = target[train_idx].clone().detach()

        data_test = data_tc[test_idx].clone().detach()
        target_test = target[test_idx].clone().detach()

        batch_size = max(int(len(data_train) / 10), 1)

        # =====================================================
        # PINN
        # =====================================================

        pinn_model = generate_model(arch_str, 2, 2)

        print("\nPINN:", pinn_name)
        print(
            "Number of parameters:",
            sum(p.numel() for p in pinn_model.parameters() if p.requires_grad),
        )

        pinn_model, pinn_loss_dict, train_time = pinn_training(
            n_epochs,
            batch_size,
            pinn_model,
            device,
            beta1,
            beta2,
            pinn_batch,
            center_x_tc,
            radius_tc,
            initial_tc,
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
            data_train,
            target_train,
        )

        error, test_time, speed_up, pred = evaluate_model(
            pinn_model, data_test, target_test, reference_time, device
        )

        pinn_error.append(error)

        pinn_train_time.append(train_time)
        pinn_test_time.append(test_time)
        pinn_speed_up.append(speed_up)

        fold_mae = np.mean(error)

        if fold_mae <= pinn_min_error:

            pinn_min_error = fold_mae

            pinn_best_loss_dict = pinn_loss_dict

            pinn_best_param = copy.deepcopy(pinn_model.state_dict())

            pinn_best_pred = pred

        # =====================================================
        # NN
        # =====================================================

        nn_model = generate_model(arch_str, 2, 2)

        print("\nNN:", pinn_name)
        print(
            "Number of parameters:",
            sum(p.numel() for p in nn_model.parameters() if p.requires_grad),
        )

        nn_model, nn_loss_dict, train_time = nn_training(
            n_epochs,
            batch_size,
            nn_model,
            device,
            beta1,
            beta2,
            data_train,
            target_train,
        )

        error, test_time, speed_up, pred = evaluate_model(
            nn_model, data_test, target_test, reference_time, device
        )

        nn_error.append(error)

        nn_train_time.append(train_time)
        nn_test_time.append(test_time)
        nn_speed_up.append(speed_up)

        fold_mae = np.mean(error)

        if fold_mae <= nn_min_error:

            nn_min_error = fold_mae

            nn_best_loss_dict = nn_loss_dict

            nn_best_param = copy.deepcopy(nn_model.state_dict())

            nn_best_pred = pred

        del pinn_model
        del nn_model

    pinn_error_flat = np.concatenate([error.flatten() for error in pinn_error])

    nn_error_flat = np.concatenate([error.flatten() for error in nn_error])

    pinn_fold_mae = np.array([np.mean(error) for error in pinn_error])

    nn_fold_mae = np.array([np.mean(error) for error in nn_error])

    pinn_fold_max_ae = np.array([np.max(error) for error in pinn_error])

    nn_fold_max_ae = np.array([np.max(error) for error in nn_error])

    pinn_output = {
        "pinn_mean_ae": float(np.mean(pinn_error_flat)),
        "pinn_std_ae": float(np.std(pinn_error_flat, ddof=1)),
        "pinn_max_ae": float(np.max(pinn_error_flat)),
        "pinn_mean_fold_mae": float(np.mean(pinn_fold_mae)),
        "pinn_std_fold_mae": float(np.std(pinn_fold_mae, ddof=1)),
        "pinn_max_fold_mae": float(np.max(pinn_fold_mae)),
        "pinn_mean_fold_max_ae": float(np.mean(pinn_fold_max_ae)),
        "pinn_max_fold_max_ae": float(np.max(pinn_fold_max_ae)),
        "pinn_mean_train_time": float(np.mean(pinn_train_time)),
        "pinn_std_train_time": float(np.std(pinn_train_time, ddof=1)),
        "pinn_mean_test_time": float(np.mean(pinn_test_time)),
        "pinn_std_test_time": float(np.std(pinn_test_time, ddof=1)),
        "pinn_mean_speed_up": float(np.mean(pinn_speed_up)),
        "pinn_std_speed_up": float(np.std(pinn_speed_up)),
        "pinn_mae_per_fold": pinn_fold_mae.tolist(),
        "pinn_max_ae_per_fold": pinn_fold_max_ae.tolist(),
    }

    nn_output = {
        "nn_mean_ae": float(np.mean(nn_error_flat)),
        "nn_std_ae": float(np.std(nn_error_flat, ddof=1)),
        "nn_max_ae": float(np.max(nn_error_flat)),
        "nn_mean_fold_mae": float(np.mean(nn_fold_mae)),
        "nn_std_fold_mae": float(np.std(nn_fold_mae, ddof=1)),
        "nn_max_fold_mae": float(np.max(nn_fold_mae)),
        "nn_mean_fold_max_ae": float(np.mean(nn_fold_max_ae)),
        "nn_max_fold_max_ae": float(np.max(nn_fold_max_ae)),
        "nn_mean_train_time": float(np.mean(nn_train_time)),
        "nn_std_train_time": float(np.std(nn_train_time, ddof=1)),
        "nn_mean_test_time": float(np.mean(nn_test_time)),
        "nn_std_test_time": float(np.std(nn_test_time, ddof=1)),
        "nn_mean_speed_up": float(np.mean(nn_speed_up)),
        "nn_std_speed_up": float(np.std(nn_speed_up)),
        "nn_mae_per_fold": nn_fold_mae.tolist(),
        "nn_max_ae_per_fold": nn_fold_max_ae.tolist(),
    }

    torch.save(pinn_best_param, "nn_parameters/pinn__" + pinn_name + ".pt")

    with open("learning_curves/pinn__" + pinn_name + ".pkl", "wb") as openfile:
        pk.dump(pinn_best_loss_dict, openfile)

    with open("nn_sim/pinn__output_" + pinn_name + ".pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(pinn_output, openfile)

    with open("nn_sim/pinn__prediction_" + pinn_name + ".pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(pinn_best_pred, openfile)

    torch.save(nn_best_param, "nn_parameters/nn__" + pinn_name + ".pt")

    with open("learning_curves/nn__" + pinn_name + ".pkl", "wb") as openfile:
        pk.dump(nn_best_loss_dict, openfile)

    with open("nn_sim/nn__output_" + pinn_name + ".pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(nn_output, openfile)

    with open("nn_sim/nn__prediction_" + pinn_name + ".pkl", "wb") as openfile:
        # Reading from json file
        pk.dump(nn_best_pred, openfile)

    return 0


if __name__ == "__main__":

    main()
