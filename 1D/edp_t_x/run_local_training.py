import os
import glob
import numpy as np
import time

sim_list = glob.glob("nn_parameters/*")

n_hd_layers = [6]

n_neurons = [2**5]

betas1 = np.linspace(0.6, 0.9, num=5, endpoint=True, dtype=np.float32)

betas2 = np.linspace(0.99, 0.9999, num=5, endpoint=True, dtype=np.float32)

count = 0

# writing jobs

for n_l in n_hd_layers:
    for n_n in n_neurons:

        arch_str = ""

        for _ in range(n_l):
            arch_str += "__" + str(n_n)

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


print(count)
