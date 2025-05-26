import os
import glob
from itertools import product
import numpy as np
import time

sim_list = glob.glob("nn_parameters/*")

n_hd_layers = [3]

n_neurons = [2**3, 2**4, 2**5]

activation_func = [
    "Tanh",
    "Softplus",
    "SiLU",
]

betas1 = np.linspace(0.6, 0.9, num=5, endpoint=True, dtype=np.float32)

betas2 = np.linspace(0.99, 0.9999, num=5, endpoint=True, dtype=np.float32)

count = 0

# writing jobs

for n_l in n_hd_layers:
    layers_combinations = list(product(n_neurons, repeat=n_l))

    for layers_comb in layers_combinations:

        arch_str = ""

        for layer in layers_comb:
            arch_str += "__" + str(layer)

        for b1 in betas1:
            for b2 in betas2:

                pinn_name = "beta1_{}__beta2_{}".format(b1, b2) + arch_str + ".pt"

                if pinn_name not in sim_list:

                    start = time.time()

                    os.system(
                        (
                            "time python3 pinn_training.py "
                            + " -a "
                            + str(arch_str)
                            + " -b1 "
                            + str(b1)
                            + " -b2 "
                            + str(b2)
                        )
                    )

                    count += 1

                    print(pinn_name, "- run time: ", time.time() - start)

                else:
                    print("Already Trained")
                    
                break
            break
        break
    break

print(count)
