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
    "MIG-d65b56b1-2519-5354-96ae-aec5f0e41128",
    "MIG-0dd7cc8d-bef1-51bb-8790-19fb03bacf66",
    "MIG-67a1ee11-1ab0-511a-ac0f-c81bdbd05f7e",
    "MIG-f184e443-af81-5f32-bad0-527cd20eb031",
    "MIG-8721230f-e004-50bd-b720-915f56b60dc6",
    "MIG-a444fcc0-f725-530b-9ffb-97805cefb734",
    "MIG-10685134-19fb-5361-83da-7bdc9b8242ba",
    "MIG-a5ff4856-76ba-5d4a-bc36-d6c908a95b14",
]

file = "phi--0.2__ksi--0.0__cb--0.15__Cn_max--0.55__lambd_nb--1.8__mi_n--0.2__lambd_bn--0.1__y_n--0.1__t_lower--0.0__t_upper--10.0"

net_arch = {
    1: {
        "arch_str": "Tanh--32__SiLU--16__",
        "epochs": "500",
        "batch": "10000",
    },
    2: {
        "arch_str": "SiLU--8__LeakyReLU--4__",
        "epochs": "500",
        "batch": "10000",
    },
}

# writing jobs

for count, arch in enumerate(net_arch):
    add_line("#!/bin/bash", "jobs/pinn_" + str(count) + ".job")
    add_line(
        "#----------------------------------------------------------",
        "jobs/pinn_" + str(count) + ".job",
    )
    add_line("# Job name", "jobs/pinn_" + str(count) + ".job")
    add_line(
        "#PBS -N pinn_" + str(count),
        "jobs/pinn_" + str(count) + ".job",
    )
    add_line(
        "#PBS -e error_files/pinn_" + str(count) + ".e",
        "jobs/pinn_" + str(count) + ".job",
    )
    add_line(
        "#PBS -o output_files/pinn_" + str(count) + ".o",
        "jobs/pinn_" + str(count) + ".job",
    )
    add_line(
        "# Run time (hh:mm:ss) - 3 hrs",
        "jobs/pinn_" + str(count) + ".job",
    )
    add_line("#PBS -l walltime=3:00:00", "jobs/pinn_" + str(count) + ".job")
    add_line(
        "#----------------------------------------------------------",
        "jobs/pinn_" + str(count) + ".job",
    )
    add_line(
        "#PBS -l nodes=compute-1-1:ppn=1",
        "jobs/pinn_" + str(count) + ".job",
    )
    add_line(
        "# Change to submission directory",
        "jobs/pinn_" + str(count) + ".job",
    )
    add_line("cd $PBS_O_WORKDIR", "jobs/pinn_" + str(count) + ".job")
    add_line("cat $PBS_NODEFILE", "jobs/pinn_" + str(count) + ".job")
    add_line(
        "# Launch Thiago-based executable",
        "jobs/pinn_" + str(count) + ".job",
    )

    add_line(
        "export CUDA_VISIBLE_DEVICES=" + v_gpu[count % len(v_gpu)],
        "jobs/pinn_" + str(count) + ".job",
    )

    add_line(
        "time ~/.conda/envs/torch_gpu/bin/python3 edo_pinn_decrease.py "
        + "-f "
        + file
        + " -n "
        + net_arch[arch]["epochs"]
        + " -b "
        + net_arch[arch]["batch"]
        + " -a "
        + net_arch[arch]["arch_str"],
        "jobs/pinn_" + str(count) + ".job",
    )

print(count + 1)
