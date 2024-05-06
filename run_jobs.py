import os
import glob
from itertools import product

file_list = glob.glob("edo_pinn_sim/*")

file = "k--0.0001__phi--0.2__ksi--0.0__cb--0.15__Cn_max--0.55__lambd_nb--1.8__mi_n--0.2__lambd_bn--0.1__y_n--0.1__t_lower--0.0__t_upper--10.0"
n_hd_layers = [1, 2, 3, 4]
n_neurons = [2**2, 2**3, 2**4, 2**5]
activation_func = ["Elu", "LeakyReLU", "Sigmoid", "Softplus", "Tanh", "Linear"]

n_epochs = [1000, 2500, 5000]
batch_size = [10000, 5000, 1000]

possible_layers = list(product(activation_func, n_neurons))

for n_l in n_hd_layers:
    for layers_comb in product(possible_layers, repeat=n_l):
        arch_str = ""

        for layer in layers_comb:
            arch_str += layer[0] + "--" + str(layer[1]) + "__"

        for n_e in n_epochs:
            for batch in batch_size:

                pinn_file = (
                    "edo_pinn_sim/epochs_{}__batch_{}__arch_".format(n_e, batch) + arch_str + ".pkl"
                ) 

                if pinn_file not in file_list:
                    print(pinn_file)

                    os.system(
                        (
                            "python3 edo_pinn_model.py "
                            + "-f "
                            + file
                            + " -n "
                            + str(int(n_e))
                            + " -b "
                            + str(int(batch))
                            + " -a "
                            + arch_str
                        )
                    )
