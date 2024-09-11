import os
import glob
from itertools import product
import numpy as np

sim_list = glob.glob("edo_pinn_sim/*")

file = "k--1__phi--0.2__ksi--0.0__cb--0.15__Cn_max--0.55__lambd_nb--1.8__mi_n--0.2__lambd_bn--0.1__y_n--0.1__t_lower--0.0__t_upper--10.0"
# n_hd_layers = [1, 2, 3]
n_hd_layers = [2]  # , 3]
n_neurons = [2**2, 2**3, 2**4]
activation_func = [
    "LeakyReLU",
    "Sigmoid",
    "Elu",
    "Tanh",
    "ReLU",
    "SiLU",
]

batch_size = [(1000, 300)]  # , (50000, 1400)]

possible_layers = list(product(activation_func, n_neurons))

count = 0

# writing jobs

for n_l in n_hd_layers:
    layers_combinations = product(possible_layers, repeat=n_l)

    for layers_comb in product(possible_layers, repeat=n_l):
        arch_str = ""

        for layer in layers_comb:
            arch_str += layer[0] + "--" + str(layer[1]) + "__"

        for batch in batch_size:
            pinn_name = (
                "edo_pinn_sim/"
                + "epochs_{}__batch_{}__arch_".format(batch[1], batch[0])
                + arch_str
            ) + ".pkl"

            if pinn_name not in sim_list:
                print("=" * 20)
                print(pinn_name)
                print("\n")

                os.system(
                    (
                        "time python3 edo_pinn_model.py "
                        + "-f "
                        + file
                        + " -n "
                        + str(int(batch[1]))
                        + " -b "
                        + str(int(batch[0]))
                        + " -a "
                        + arch_str
                        + " -v "
                        + str(0.2)
                    )
                )

            count += 1
