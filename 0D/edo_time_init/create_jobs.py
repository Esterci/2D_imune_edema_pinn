import os
import glob
from itertools import product
import numpy as np


def add_line(line, out):
    # Open the file in append & read mode ('a+')

    with open(out, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)

        # If file is not empty then append '\n'
        data = file_object.read(100)

        if len(data) > 0:
            file_object.write("\n")

        # Append text at the end of file
        file_object.write(line)


v_gpu = [
    "MIG-d14ab5f9-ca27-5f8a-bfe1-57e541d0328f",
    "MIG-03c65507-2388-56b3-a2d9-b30c8b5723ee",
    "MIG-f184e443-af81-5f32-bad0-527cd20eb031",
    "MIG-8721230f-e004-50bd-b720-915f56b60dc6",
    "MIG-a444fcc0-f725-530b-9ffb-97805cefb734",
    "MIG-10685134-19fb-5361-83da-7bdc9b8242ba",
    "MIG-a5ff4856-76ba-5d4a-bc36-d6c908a95b14",
    "MIG-d65b56b1-2519-5354-96ae-aec5f0e41128",
]

file = "k--0.1__phi--0.2__ksi--0.0__cb--0.15__Cn_max--0.55__lambd_nb--1.8__mi_n--0.2__lambd_bn--0.1__y_n--0.1__t_lower--0.0__t_upper--10.0"
n_hd_layers = [3, 4]
n_neurons = [2**3, 2**4, 2**5]
activation_func = [
    "Elu",
    "Tanh",
    "ReLU",
    "SiLU",
]

batch_size = [(1000, 300)]

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
                "epochs_{}__batch_{}__arch_".format(batch[1], batch[0]) + arch_str
            )

            if count % 20 == 0:
                add_line("#!/bin/bash", "jobs/pinn_" + str(count // 20) + ".job")
                add_line(
                    "#----------------------------------------------------------",
                    "jobs/pinn_" + str(count // 20) + ".job",
                )
                add_line("# Job name", "jobs/pinn_" + str(count // 20) + ".job")
                add_line(
                    "#PBS -N pinn_" + str(count // 20),
                    "jobs/pinn_" + str(count // 20) + ".job",
                )
                add_line(
                    "#PBS -e error_files/pinn_" + str(count // 20) + ".e",
                    "jobs/pinn_" + str(count // 20) + ".job",
                )
                add_line(
                    "#PBS -o output_files/pinn_" + str(count // 20) + ".o",
                    "jobs/pinn_" + str(count // 20) + ".job",
                )
                add_line(
                    "# Run time (hh:mm:ss) - 3 hrs",
                    "jobs/pinn_" + str(count // 20) + ".job",
                )
                add_line(
                    "#PBS -l walltime=3:00:00", "jobs/pinn_" + str(count // 20) + ".job"
                )
                add_line(
                    "#----------------------------------------------------------",
                    "jobs/pinn_" + str(count // 20) + ".job",
                )
                add_line(
                    "#PBS -l nodes=compute-1-1:ppn=1",
                    "jobs/pinn_" + str(count // 20) + ".job",
                )
                add_line(
                    "# Change to submission directory",
                    "jobs/pinn_" + str(count // 20) + ".job",
                )
                add_line("cd $PBS_O_WORKDIR", "jobs/pinn_" + str(count // 20) + ".job")
                add_line("cat $PBS_NODEFILE", "jobs/pinn_" + str(count // 20) + ".job")
                add_line(
                    "# Launch Thiago-based executable",
                    "jobs/pinn_" + str(count // 20) + ".job",
                )

                add_line(
                    "export CUDA_VISIBLE_DEVICES=" + v_gpu[count // 20 % len(v_gpu)],
                    "jobs/pinn_" + str(count // 20) + ".job",
                )

            add_line(
                "time ~/.conda/envs/torch_gpu/bin/python3 edo_pinn_model.py "
                + "-f "
                + file
                + " -n "
                + str(int(batch[1]))
                + " -b "
                + str(int(batch[0]))
                + " -a "
                + arch_str
                + " -g "
                + str(v_gpu[count // 20 % len(v_gpu)])
                + " -v "
                + str(0.2),
                "jobs/pinn_" + str(count // 20) + ".job",
            )

            count += 1

if count % 20 != 0:
    jobs = (count // 20) + 1

else:
    jobs = count / 20

print(jobs)
