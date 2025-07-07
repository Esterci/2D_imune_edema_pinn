from glob import glob
import time
import pickle as pk
import os
import json
from pinn import *
from fisiocomPinn.Net import *


def load_model(file_name, device):
    cwd = os.getcwd()

    hidden_layer = [
        int(n_neurons)
        for n_neurons in file_name.split("beta2_")[-1].split(".pt")[0].split("__")[1:]
    ]

    dtype = torch.float32
    model = FullyConnectedNetwork(2, 2, hidden_layer, dtype=dtype)

    model.load_state_dict(
        torch.load(cwd + "/" + file_name, weights_only=True, map_location=device)
    )

    print(model.eval())

    return model.to(device)


def read_speed_ups(speed_up_list):
    speed_up_obj = {}
    for i, file in enumerate(speed_up_list):
        with open(file, "rb") as f:
            speed_up_obj[i] = pk.load(f)

    return speed_up_obj


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

    del mesh_properties

with open("source_points/lymph_vessels.pkl", "rb") as f:
    leu_source_points = pk.load(f)


arch_str = "32__32__32__32__32__32__32"

beta1 = 0.825

beta2 = 0.99

n_epochs = int(1e4)

hidden_layer = [int(n_neurons) for n_neurons in arch_str.split("__")[1:]]

dtype = torch.float32

samples_list = np.linspace(1e2, 1.5e3, num=10, endpoint=True, dtype=int)


if __name__ == "__main__":

    Cl_list, Cp_list, speed_up_list = read_files("fvm_sim")

    Cp_fvm, Cl_fvm, center, radius = format_array(Cp_list[0], Cl_list[0])

    size_x, size_y, size_t = get_mesh_properties(x_dom, y_dom, t_dom, h, k)

    for n_samples in samples_list:

        print("\n==== Runing Tests for {} samples ====".format(n_samples))

        (
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
            n_samples=n_samples,
        )

        print(len(data_tc))

        batch_size = int(len(reduced_data_tc) / 10)

        target = target.cpu().detach()

        speed_up_obj = read_speed_ups(speed_up_list)

        metrics = {
            "n_samples": n_samples,
            "mean_pinn_time": [],
            "std_pinn_time": 0,
            "pinn_rmse": 0,
            "pin_mae": 0,
            "mean_nn_time": [],
            "std_nn_time": 0,
            "nn_rmse": 0,
            "pin_mae": 0,
        }

        output = {}

        error_pinn = np.zeros((len(speed_up_obj.keys()), len(target)))

        error_nn = np.zeros((len(speed_up_obj.keys()), len(target)))

        min_error_pinn = 1000000

        min_error_nn = 1000000

        for i in speed_up_obj.keys():

            print("\n    => Iteration {}".format(i))

            pinn_model = FullyConnectedNetwork(2, 2, hidden_layer, dtype=dtype)

            optimizer = optim.Adam(
                pinn_model.parameters(), lr=1e-3, betas=(beta1, beta2)
            )

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

            pinn_trainer = Trainer(
                n_epochs=n_epochs,
                batch_size=batch_size,
                model=pinn_model,
                device=device,
                target=reduced_target,
                data=reduced_data_tc,
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

            init_loss.setBatchGenerator(
                generate_initial_points, center_x_tc, radius_tc, initial_tc
            )

            pinn_trainer.add_loss(init_loss)

            bnd_loss = LOSS_PINN(
                batch_size=batch_size,
                device=device,
                loss="RMSE",
                name="LossBoundary",
            )

            bnd_loss.setBatchGenerator(generate_boundary_points)

            bnd_loss.setPinnFunction(boundary_condition, Dn, X_nb, Db, device)

            pinn_trainer.add_loss(bnd_loss)

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

            pinn_trainer.add_loss(pde_loss, 5)

            start = time.time()

            pinn_model, _ = pinn_trainer.train()

            end = time.time()

            pinn_time = end - start

            mesh = data_tc.to(device)

            with torch.no_grad():
                pred_pinn_dev = pinn_model(mesh)

            pred_pinn = pred_pinn_dev.cpu().detach().numpy()

            aux = ((pred_pinn - target.cpu().detach().numpy()) ** 2) ** 0.5

            error_pinn[i] = aux[:, 0] + aux[:, 1]

            local_pinn_rmse = np.mean(aux[:, 0] + aux[:, 1])

            if local_pinn_rmse < min_error_pinn:
                min_error_pinn = local_pinn_rmse

                output["pred_pinn"] = pred_pinn

            metrics["mean_pinn_time"].append(pinn_time)

            nn_model = FullyConnectedNetwork(2, 2, hidden_layer, dtype=dtype)

            optimizer = optim.Adam(nn_model.parameters(), lr=1e-3, betas=(beta1, beta2))
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

            trainer = Trainer(
                n_epochs=n_epochs,
                batch_size=batch_size,
                model=nn_model,
                device=device,
                target=reduced_target,
                data=reduced_data_tc,
                patience=5000,
                tolerance=0.01,
                validation=0.2,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                print_steps=1e10,
                constant_properties=constant_properties,
            )

            start = time.time()

            nn_model, loss_dict = trainer.train()

            end = time.time()

            nn_time = end - start

            with torch.no_grad():
                pred_nn_dev = nn_model(mesh)

            pred_nn = pred_nn_dev.cpu().detach().numpy()

            aux = ((pred_nn - target.cpu().detach().numpy()) ** 2) ** 0.5

            error_nn[i] = aux[:, 0] + aux[:, 1]

            local_nn_rmse = np.mean(aux[:, 0] + aux[:, 1])

            if local_nn_rmse < min_error_nn:
                min_error_nn = local_nn_rmse

                output["pred_nn"] = pred_nn

                with open(
                    "learning_curves/comp_" + str(n_samples) + ".pkl", "wb"
                ) as openfile:
                    # Reading from json file
                    pk.dump(loss_dict, openfile)

            metrics["mean_nn_time"].append(nn_time)

        output["target"] = target.cpu().detach().numpy()

        metrics["std_pinn_time"] = np.std(metrics["mean_pinn_time"])
        metrics["mean_pinn_time"] = np.mean(metrics["mean_pinn_time"])
        metrics["pinn_rmse"] = np.mean(error_pinn.flatten())
        metrics["pinn_mae"] = np.mean(error_pinn.flatten())

        metrics["std_nn_time"] = np.std(metrics["mean_nn_time"])
        metrics["mean_nn_time"] = np.mean(metrics["mean_nn_time"])
        metrics["nn_rmse"] = np.mean(error_nn.flatten())
        metrics["nn_mae"] = np.mean(error_nn.flatten())

        with open("pinn_sim/comp_output_" + str(n_samples) + ".pkl", "wb") as openfile:
            # Reading from json file
            pk.dump(output, openfile)

        with open("pinn_sim/comp_metrics_" + str(n_samples) + ".pkl", "wb") as openfile:
            # Reading from json file
            pk.dump(metrics, openfile)
