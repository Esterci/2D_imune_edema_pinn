from glob import glob
import time
import pickle as pk
import os
import json
from pinn import *

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
ini_cond_var = constant_properties["ini_cond_var"]


# Opening JSON file
with open("control_dicts/mesh_properties.json", "r") as openfile:
    # Reading from json file
    mesh_properties = json.load(openfile)

h = mesh_properties["h"]
k = mesh_properties["k"]
x_dom = mesh_properties["x_dom"]
y_dom = mesh_properties["y_dom"]
t_dom = mesh_properties["t_dom"]


Cn_list, Cb_list, speed_up_list = read_files("fvm_sim")

Cp_fvm, Cl_fvm, center_x_array, center_y_array, radius_array = format_array(
    [Cb_list[5]], [Cn_list[5]]
)


simp_Cp_fvm = simplify_mx(Cp_fvm)
simp_Cl_fvm = simplify_mx(Cl_fvm)


size_x, size_y, size_t, initial_cond = get_mesh_properties(
    x_dom, y_dom, t_dom, h, k, central_ini_cond, ini_cond_var, Cp_fvm.shape[1]
)


(
    initial_tc,
    center_x_tc,
    center_y_tc,
    radius_tc,
    t_tc,
    x_tc,
    y_tc,
    target,
    device,
) = allocates_training_mesh(
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
    simp_Cp_fvm,
    simp_Cl_fvm,
)


nn_list = glob("nn_parameters/*")
run_list = list(
    map(
        lambda file: file.split("pinn_sim/")[-1]
        .split(".pkl")[0]
        .split("prediction_")[-1],
        glob("pinn_sim/*"),
    )
)

total = len(nn_list)

for nn_num, nn_file in enumerate(nn_list):

    pinn_file = nn_file.split("nn_parameters/")[-1].split(".pt")[0]

    print(f"\n{nn_num+1} of {total}")

    print("=" * 20)

    print("PINN:", pinn_file)

    if pinn_file in run_list:
        print("Already evaluated")

    else:
        model = load_model(nn_file, device)

        speed_up_obj = read_speed_ups(speed_up_list)

        output = {
            "mean_speed_up": [],
            "std_speed_up": 0,
            "mean_speed_comp_up": [],
            "std_speed_comp_up": 0,
            "mean_speed_up_pinn": [],
            "std_speed_up_pinn": 0,
            "mean_serial_time": [],
            "std_serial_time": 0,
            "mean_cuda_time": [],
            "std_cuda_time": 0,
            "mean_pinn_time": [],
            "std_pinn_time": 0,
        }

        prediction = {}

        error = np.zeros((len(speed_up_obj.keys()), len(target)))
        target_np = target.cpu().detach().numpy()

        for i in speed_up_obj.keys():

            start = time.time()

            mesh = torch.cat([t_tc, x_tc, y_tc], dim=1).to(device)

            with torch.no_grad():
                pred_pinn_dev = model(mesh)

            pred_pinn = pred_pinn_dev.cpu().detach().numpy()

            end = time.time()

            pinn_time = end - start

            speed_up_obj[i]["pinn_time"] = pinn_time

            speed_up_obj[i]["speed_up_pinn"] = (
                speed_up_obj[i]["serial_time"] / pinn_time
            )

            output["mean_speed_up"].append(speed_up_obj[i]["speed_up"])

            output["mean_speed_comp_up"].append(speed_up_obj[i]["speed_comp_up"])

            output["mean_speed_up_pinn"].append(speed_up_obj[i]["speed_up_pinn"])

            output["mean_serial_time"].append(speed_up_obj[i]["serial_time"])

            output["mean_cuda_time"].append(speed_up_obj[i]["cuda_time"])

            output["mean_pinn_time"].append(speed_up_obj[i]["pinn_time"])

            aux = ((pred_pinn - target_np) ** 2) ** 0.5

            error[i] = aux[:, 0] + aux[:, 1]

        rmse = np.mean(error.flatten())
        max_ae = np.max(error.flatten())

        print("Erro absoluto médio", rmse)
        print("Erro absoluto máximo", max_ae)

        output["rmse"] = rmse
        output["max_ae"] = max_ae

        output["std_speed_up"] = np.std(output["mean_speed_up"])
        output["std_speed_comp_up"] = np.std(output["mean_speed_comp_up"])
        output["std_speed_up_pinn"] = np.std(output["mean_speed_up_pinn"])
        output["std_serial_time"] = np.std(output["mean_serial_time"])
        output["std_cuda_time"] = np.std(output["mean_cuda_time"])
        output["std_pinn_time"] = np.std(output["mean_pinn_time"])

        output["mean_speed_up"] = np.mean(output["mean_speed_up"])
        output["mean_speed_comp_up"] = np.mean(output["mean_speed_comp_up"])
        output["mean_speed_up_pinn"] = np.mean(output["mean_speed_up_pinn"])
        output["mean_serial_time"] = np.mean(output["mean_serial_time"])
        output["mean_cuda_time"] = np.mean(output["mean_cuda_time"])
        output["mean_pinn_time"] = np.mean(output["mean_pinn_time"])

        print(
            "Speed Up: {} +/-{}".format(output["mean_speed_up"], output["std_speed_up"])
        )
        print(
            "Compilation Speed Up: {} +/-{}".format(
                output["mean_speed_comp_up"], output["std_speed_comp_up"]
            )
        )
        print(
            "PINN Speed Up: {} +/-{}".format(
                output["mean_speed_up_pinn"], output["std_speed_up_pinn"]
            )
        )

        prediction["pred_pinn"] = pred_pinn
        prediction["target"] = target_np

        with open("pinn_sim/output_" + pinn_file + ".pkl", "wb") as openfile:
            # Reading from json file
            pk.dump(output, openfile)

        with open("pinn_sim/prediction_" + pinn_file + ".pkl", "wb") as openfile:
            # Reading from json file
            pk.dump(prediction, openfile)
