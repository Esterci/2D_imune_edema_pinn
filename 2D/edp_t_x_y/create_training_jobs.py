from itertools import product


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


chunck_size = 100

v_gpu = [
    "MIG-a444fcc0-f725-530b-9ffb-97805cefb734",
    "MIG-10685134-19fb-5361-83da-7bdc9b8242ba",
    "MIG-a5ff4856-76ba-5d4a-bc36-d6c908a95b14",
    "MIG-d65b56b1-2519-5354-96ae-aec5f0e41128",
    "MIG-275c5d7e-981d-5f7a-b45b-0659ba9ad13a",
    "MIG-3aad3b21-c6f1-5b32-9d6b-1341d2b38d11",
    "MIG-f946a009-bfbb-5335-89ba-7f3ac431bf10",
]

n_hd_layers = [4]

n_neurons = [2**3, 2**4, 2**5]

activation_func = [
    "Elu",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "SiLU",
]

batch_size = [(108000, 300)]

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
                "nn_parameters/"
                + "epochs_{}__batch_{}__arch_".format(batch[1], batch[0])
                + arch_str
            ) + ".pt"

            if pinn_name not in sim_list:
                if count % chunck_size == 0:
                    add_line(
                        "#!/bin/bash", "jobs/pinn_" + str(count // chunck_size) + ".job"
                    )
                    add_line(
                        "#----------------------------------------------------------",
                        "jobs/pinn_" + str(count // chunck_size) + ".job",
                    )
                    add_line(
                        "# Job name", "jobs/pinn_" + str(count // chunck_size) + ".job"
                    )
                    add_line(
                        "#PBS -N pinn_" + str(count // chunck_size),
                        "jobs/pinn_" + str(count // chunck_size) + ".job",
                    )
                    add_line(
                        "#PBS -e error_files/pinn_" + str(count // chunck_size) + ".e",
                        "jobs/pinn_" + str(count // chunck_size) + ".job",
                    )
                    add_line(
                        "#PBS -o output_files/pinn_" + str(count // chunck_size) + ".o",
                        "jobs/pinn_" + str(count // chunck_size) + ".job",
                    )
                    add_line(
                        "# Run time (hh:mm:ss) - 4:00 hr",
                        "jobs/pinn_" + str(count // chunck_size) + ".job",
                    )
                    add_line(
                        "#PBS -l walltime=4:00:00",
                        "jobs/pinn_" + str(count // chunck_size) + ".job",
                    )
                    add_line(
                        "#----------------------------------------------------------",
                        "jobs/pinn_" + str(count // chunck_size) + ".job",
                    )
                    add_line(
                        "#PBS -l nodes=compute-1-1:ppn=1",
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
                    "time ~/.conda/envs/pyTourch/bin/python3 edp_pinn_model.py "
                    + " -n "
                    + str(int(batch[1]))
                    + " -b "
                    + str(int(batch[0]))
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
