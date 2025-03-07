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
    "UUID: MIG-bbc0f904-20ba-5ff0-aa76-754fa93730ba",
    "UUID: MIG-5e5f0459-057d-5c61-b065-68892d1d27df",
    "UUID: MIG-6d453f0c-ca44-53ae-947c-52497b81d66b",
    "UUID: MIG-fad1c2a5-0979-5cbb-b4db-06a50630487c",
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
