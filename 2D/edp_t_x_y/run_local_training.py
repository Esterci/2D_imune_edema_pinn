import os
import glob
from itertools import product
import numpy as np

sim_list = glob.glob("nn_parameters/*")

n_hd_layers = [4]
n_neurons = [2**2]  # , 2**3, 2**4]

activation_func = [
    # "LeakyReLU",
    # "Sigmoid",
    # "Elu",
    "Tanh",
    # "ReLU",
    # "SiLU",
]


batch_size = [(108000, 300)]  # , (50000, 1400)]

possible_layers = list(product(activation_func, n_neurons))

# writing jobs

for n_l in n_hd_layers:
    layers_combinations = product(possible_layers, repeat=n_l)

    for layers_comb in product(possible_layers, repeat=n_l):
        arch_str = ""

        for layer in layers_comb:
            arch_str += layer[0] + "--" + str(layer[1]) + "__"

        for batch in batch_size:
            pinn_name = (
                "nn_parameters/"
                + "epochs_{}__batch_{}__arch_".format(batch[1], batch[0])
                + arch_str
            ) + ".pt"

            print("=" * 20)
            print(pinn_name)
            print("\n")

            if pinn_name not in sim_list:

                os.system(
                    (
                        "time python3 pinn_training.py "
                        + " -n "
                        + str(int(batch[1]))
                        + " -b "
                        + str(int(batch[0]))
                        + " -a "
                        + arch_str
                    )
                )
            
            else:
                print("Already Trained")
