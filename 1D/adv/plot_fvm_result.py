import pickle as pkl
from plots import plot_results
import argparse
import json
from utils import init_mesh

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

    with open("fvm_sim/Cl__" + file_name + ".pkl", "rb") as openfile:
        Cl = pkl.load(openfile)

    with open("fvm_sim/Cp__" + file_name + ".pkl", "rb") as openfile:
        Cp = pkl.load(openfile)

    # Load mesh properties from JSON file
    with open("control_dicts/mesh_properties.json", "r") as openfile:
        mesh_properties = json.load(openfile)

    h = mesh_properties["h"]
    k = mesh_properties["k"]
    x_dom = mesh_properties["x_dom"]
    y_dom = mesh_properties["y_dom"]
    t_dom = mesh_properties["t_dom"]

    # Generate random initial condition parameters
    center = (0.35, 0)
    radius = 0.15

    # Initialize mesh and related properties

    size_x, size_y, size_t, leu_source_points, struct_name = init_mesh(
        x_dom,
        y_dom,
        t_dom,
        h,
        k,
        center,
        radius,
        create_source=False,
    )

    plot_results(size_t, size_x, t_dom, x_dom, Cp, Cl, leu_source_points)
