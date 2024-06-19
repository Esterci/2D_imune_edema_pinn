import numpy as np
import argparse
import pickle as pk


def fb(Cb, Cn, cb, lambd_nb):
    return (cb - lambd_nb * Cn) * Cb


def fn(Cb, Cn, Cn_max, y_n, lambd_bn, mi_n):
    return y_n * Cb * (Cn_max - Cn) - lambd_bn * Cn * Cb - mi_n * Cn


# Parsing model parameters


def argsDictionary(args):

    var_dict = vars(args)

    structure_name = []

    for var in var_dict:
        if not var == "file":
            hp_n_values = str(var) + "-" * 2 + str(var_dict[var])
            structure_name.append(hp_n_values)

    structure_name = ("_" * 2).join(structure_name).split("__save")[0]

    return var_dict, structure_name


def pde():

    import time as tm

    # Computing fdm model

    size_tt = int(((t_upper - t_lower) / (k)))

    Cn_final = np.zeros((size_tt))
    Cb_final = np.zeros((size_tt))

    Cn_new = 0
    Cb_new = 0.2

    for time in range(size_tt):

        Cn_old = Cn_new
        Cb_old = Cb_new

        Cb_new = (k / phi) * (fb(Cb_old, Cn_old, cb, lambd_nb)) + Cb_old

        Cn_new = (k / phi) * (fn(Cb_old, Cn_old, Cn_max, y_n, lambd_bn, mi_n)) + Cn_old

        Cb_final[time] = Cb_new
        Cn_final[time] = Cn_new

    if save:
        with open("edo_fdm_sim/Cp__" + struct_name + ".pkl", "wb") as f:
            pk.dump(Cb_final, f)

        with open("edo_fdm_sim/Cl__" + struct_name + ".pkl", "wb") as f:
            pk.dump(Cn_final, f)

    return Cb_final, Cn_final

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="", add_help=False)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k",
        "--k",
        type=float,
        action="store",
        dest="k",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-p",
        "--phi",
        type=float,
        action="store",
        dest="phi",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-ksi",
        "--ksi",
        type=float,
        action="store",
        dest="ksi",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-c",
        "--cb",
        type=float,
        action="store",
        dest="cb",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-cmax",
        "--Cn_max",
        type=float,
        action="store",
        dest="Cn_max",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-l_nb",
        "--lambd_nb",
        type=float,
        action="store",
        dest="lambd_nb",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-m",
        "--mi_n",
        type=float,
        action="store",
        dest="mi_n",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-l_bn",
        "--lambd_bn",
        type=float,
        action="store",
        dest="lambd_bn",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-y",
        "--y_n",
        type=float,
        action="store",
        dest="y_n",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-tl",
        "--t_lower",
        type=float,
        action="store",
        dest="t_lower",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-tu",
        "--t_upper",
        type=float,
        action="store",
        dest="t_upper",
        required=True,
        default=None,
        help="",
    )

    parser.add_argument(
        "-s",
        "--save",
        type=bool,
        action="store",
        dest="save",
        required=False,
        default=False,
        help="",
    )

    args = parser.parse_args()

    args_dict, struct_name = argsDictionary(args)

    # initializaing fdm model

    k = args_dict["k"]
    phi = args_dict["phi"]
    ksi = args_dict["ksi"]
    cb = args_dict["cb"]
    Cn_max = args_dict["Cn_max"]
    lambd_nb = args_dict["lambd_nb"]
    mi_n = args_dict["mi_n"]
    lambd_bn = args_dict["lambd_bn"]
    y_n = args_dict["y_n"]
    t_lower = args_dict["t_lower"]
    t_upper = args_dict["t_upper"]
    save = args_dict["save"]
    
    pde()
