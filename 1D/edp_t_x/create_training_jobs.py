from itertools import product
import numpy as np
import glob


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


sim_list = glob.glob("nn_parameters/*")


chunck_size = 1300

v_gpu = [
    "MIG-bbc0f904-20ba-5ff0-aa76-754fa93730ba",
    "MIG-5e5f0459-057d-5c61-b065-68892d1d27df",
    "MIG-6d453f0c-ca44-53ae-947c-52497b81d66b",
    "MIG-fad1c2a5-0979-5cbb-b4db-06a50630487c",
]

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
                    if count % chunck_size == 0:
                        add_line(
                            "#!/bin/bash",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "#----------------------------------------------------------",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "# Job name",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "#PBS -N pinn_" + str(count // chunck_size),
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "#PBS -e error_files/pinn_"
                            + str(count // chunck_size)
                            + ".e",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "#PBS -o output_files/pinn_"
                            + str(count // chunck_size)
                            + ".o",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "# Run time (hh:mm:ss) - 100:00 hr",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "#PBS -l walltime=100:00:00",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "#----------------------------------------------------------",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "#PBS -l nodes=compute-1-0:ppn=1",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "# Change to submission directory",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "cd $PBS_O_WORKDIR",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "cat $PBS_NODEFILE",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )
                        add_line(
                            "# Launch Thiago-based executable",
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )

                        add_line(
                            "export CUDA_VISIBLE_DEVICES="
                            + v_gpu[count // chunck_size % len(v_gpu)],
                            "jobs/pinn_" + str(count // chunck_size) + ".job",
                        )

                    add_line(
                        "time ~/.conda/envs/pyTourch/bin/python3 pinn_training.py "
                        + " -d "
                        + str(decay_rate)
                        + " -l "
                        + str(lr_rate)
                        + " -a "
                        + arch_str,
                        "jobs/pinn_" + str(count // chunck_size) + ".job",
                    )

                    count += 1

if count % chunck_size != 0:
    jobs = (count // chunck_size) + 1

else:
    jobs = count / chunck_size

print(jobs)
