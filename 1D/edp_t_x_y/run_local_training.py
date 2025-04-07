import os
import glob
from itertools import product
import numpy as np

sim_list = glob.glob("nn_parameters/*")

n_hd_layers = [3]

n_neurons = [2**3, 2**4, 2**5]

activation_func = [
    "Tanh",
    "Softplus",
    "SiLU",
]

possible_layers = list(product(activation_func, n_neurons))

decay_rates = np.linspace(0.95, 0.999, num=3, endpoint=True, dtype=np.float32)

lr_rates = np.linspace(1e-4, 1e-3, num=3, endpoint=True, dtype=np.float32)

count = 0

# writing jobs

for n_l in n_hd_layers:
    layers_combinations = product(possible_layers, repeat=n_l)

    for layers_comb in product(possible_layers, repeat=n_l):
        arch_str = ""

        for layer in layers_comb:
            arch_str += layer[0] + "--" + str(layer[1]) + "__"

        for lr_rate in lr_rates:

            for decay_rate in decay_rates:
                pinn_name = (
                    "nn_parameters/"
                    + "decay_rates_{:.4}__lr_rates_{:.4}__arch_".format(
                        decay_rate, lr_rate
                    )
                    + arch_str
                    + ".pt"
                )

                if pinn_name not in sim_list:

                    os.system(
                        (
                            "time python3 pinn_training.py "
                            + " -d "
                            + str(decay_rate)
                            + " -l "
                            + str(lr_rate)
                            + " -a "
                            + arch_str
                        )
                    )

                    count += 1

                else:
                    print("Already Trained")
