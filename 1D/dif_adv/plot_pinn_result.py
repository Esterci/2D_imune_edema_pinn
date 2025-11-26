import pickle as pkl
from plots import plot_comparison
import argparse
import json
from utils import init_mesh
from pinn import read_files, format_array, get_mesh_properties
import numpy as np


# Parsing model parameters

parser = argparse.ArgumentParser()

parser.add_argument(
    "-f",
    "--file_name",
    type=str,
    action="store",
    dest="file_name",
    required=True,
    default=None,
    help="FVM description used to generate file_name",
)


if __name__ == "__main__":

    args = parser.parse_args()

    args_dict = vars(args)

    file_name = args_dict["file_name"]

    with open("learning_curves/" + file_name + ".pkl", "rb") as f:
        loss_dict = pkl.load(f)

    # Load mesh properties from JSON file
    with open("control_dicts/mesh_properties.json", "r") as openfile:
        mesh_properties = json.load(openfile)

    with open("pinn_sim/prediction_" + file_name + ".pkl", "rb") as f:
        # Reading from json file
        prediction = pkl.load(f)

    with open("source_points/lymph_vessels.pkl", "rb") as f:
        leu_source_points = pkl.load(f)

    h = mesh_properties["h"]
    k = mesh_properties["k"]
    x_dom = mesh_properties["x_dom"]
    y_dom = mesh_properties["y_dom"]
    t_dom = mesh_properties["t_dom"]

    Cl_list, Cp_list, speed_up_list = read_files("fvm_sim")

    Cp_fvm, Cl_fvm, center, radius = format_array(Cp_list[0], Cl_list[0])

    size_x, size_y, size_t = get_mesh_properties(x_dom, y_dom, t_dom, h, k)

    pred_pinn = prediction["pred_pinn"]

    target = prediction["target"]

    Cl_pinn_device, Cp_pinn_device = np.split(pred_pinn, 2, axis=1)

    Cl_pinn_np = Cl_pinn_device.reshape(Cp_fvm.shape)

    Cp_pinn_np = Cp_pinn_device.reshape(Cp_fvm.shape)

    plot_comparison(
        size_t,
        size_x,
        t_dom,
        x_dom,
        Cp_fvm,
        Cl_fvm,
        Cp_pinn_np,
        Cl_pinn_np,
        leu_source_points,
    )
