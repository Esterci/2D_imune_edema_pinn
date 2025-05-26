from itertools import product


def add_line(line, out, line_break=True):
    # Open the file in append & read mode ('a+')

    with open(out, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)

        # If file is not empty then append '\n'
        data = file_object.read(100)

        if len(data) > 0 and line_break:
            file_object.write("\n")

        # Append text at the end of file
        file_object.write(line)


v_gpu = [
    "MIG-a444fcc0-f725-530b-9ffb-97805cefb734",
    "MIG-10685134-19fb-5361-83da-7bdc9b8242ba",
    "MIG-a5ff4856-76ba-5d4a-bc36-d6c908a95b14",
    "MIG-d65b56b1-2519-5354-96ae-aec5f0e41128",
    "MIG-275c5d7e-981d-5f7a-b45b-0659ba9ad13a",
    "MIG-3aad3b21-c6f1-5b32-9d6b-1341d2b38d11",
    "MIG-f946a009-bfbb-5335-89ba-7f3ac431bf10",
]


add_line("#!/bin/bash", "jobs/fvm_comp.job")
add_line(
    "#----------------------------------------------------------",
    "jobs/fvm_comp.job",
)
add_line("# Job name", "jobs/fvm_comp.job")
add_line(
    "#PBS -N fvm_comp",
    "jobs/fvm_comp.job",
)
add_line(
    "#PBS -e error_files/fvm_comp.e",
    "jobs/fvm_comp.job",
)
add_line(
    "#PBS -o output_files/fvm_comp.o",
    "jobs/fvm_comp.job",
)
add_line(
    "# Run time (hh:mm:ss) - 10:00 hr",
    "jobs/fvm_comp.job",
)
add_line(
    "#PBS -l walltime=10:00:00",
    "jobs/fvm_comp.job",
)
add_line(
    "#----------------------------------------------------------",
    "jobs/fvm_comp.job",
)
add_line(
    "#PBS -l nodes=compute-1-0:ppn=128",
    "jobs/fvm_comp.job",
)
add_line(
    "# Change to submission directory",
    "jobs/fvm_comp.job",
)
add_line(
    "cd $PBS_O_WORKDIR",
    "jobs/fvm_comp.job",
)
add_line(
    "cat $PBS_NODEFILE",
    "jobs/fvm_comp.job",
)
add_line(
    "# Launch Thiago-based executable",
    "jobs/fvm_comp.job",
)

count = 0

for i in range(33):

    if i != 0 and (i % len(v_gpu) == len(v_gpu) - 1 or i == 32):
        add_line(
            "export CUDA_VISIBLE_DEVICES="
            + v_gpu[i % len(v_gpu)]
            + " && ~/.conda/envs/torch-numba-11/bin/python3 fvm_comparison.py;",
            "jobs/fvm_comp.job",
        )

    else:
        add_line(
            "export CUDA_VISIBLE_DEVICES="
            + v_gpu[i % len(v_gpu)]
            + " && ~/.conda/envs/torch-numba-11/bin/python3 fvm_comparison.py & ",
            "jobs/fvm_comp.job",
        )
