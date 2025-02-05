from itertools import product
import os


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


chunck_size = 2

node = "nodes=compute-1-0"

os.system("rm jobs/pinn_*")

v_gpu = [
    "GPU-fd7e14c3-91ce-6c4b-e736-393c0d0537ef",
    "GPU-49723f5b-3680-6d21-0357-4b7bf88ad0e7",
    # "MIG-f184e443-af81-5f32-bad0-527cd20eb031",
    # "MIG-8721230f-e004-50bd-b720-915f56b60dc6",
    # "MIG-a444fcc0-f725-530b-9ffb-97805cefb734",
    # "MIG-10685134-19fb-5361-83da-7bdc9b8242ba",
    # "MIG-a5ff4856-76ba-5d4a-bc36-d6c908a95b14",
    # "MIG-d65b56b1-2519-5354-96ae-aec5f0e41128",
]

n_hd_layers = [3, 4]

n_neurons = [2**3, 2**4, 2**5]

activation_func = [
    "Elu",
    "Tanh",
    "ReLU",
    "SiLU",
]

batch_size = [(6e5, 300)]

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

            if count % chunck_size == 0:
                add_line(
                    "#!/bin/bash",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "#----------------------------------------------------------",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "# Job name",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "#PBS -N pinn_" + str(count // chunck_size),
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "#PBS -e error_files/pinn_" + str(count // chunck_size) + ".e",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "#PBS -o output_files/pinn_" + str(count // chunck_size) + ".o",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "# Run time (hh:mm:ss) - 4:00 hr",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "#PBS -l walltime=4:00:00",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "#----------------------------------------------------------",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "#PBS -l " + node + ":ppn=1",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "# Change to submission directory",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "cd $PBS_O_WORKDIR",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "cat $PBS_NODEFILE",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )
                add_line(
                    "# Launch Thiago-based executable",
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )

                add_line(
                    "export CUDA_VISIBLE_DEVICES="
                    + v_gpu[count // chunck_size % len(v_gpu)],
                    "jobs/pinn_train_" + str(count // chunck_size) + ".job",
                )

            add_line(
                "time ~/.conda/envs/pyTourch/bin/python3 edp_pinn_model.py "
                + " -n "
                + str(int(batch[1]))
                + " -b "
                + str(int(batch[0]))
                + " -a "
                + arch_str,
                "jobs/pinn_train_" + str(count // chunck_size) + ".job",
            )

            count += 1

            if count == 2:

                break


if count % chunck_size != 0:
    jobs = (count // chunck_size) + 1

else:
    jobs = count / chunck_size

print(jobs)
