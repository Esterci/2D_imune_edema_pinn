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
    "GPU-fd7e14c3-91ce-6c4b-e736-393c0d0537ef",
    "GPU-49723f5b-3680-6d21-0357-4b7bf88ad0e7",
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
    "#PBS -l nodes=compute-1-1:ppn=128",
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

    if i % 2 == 0 and i != 32:
        add_line(
            "export CUDA_VISIBLE_DEVICES="
            + v_gpu[i % len(v_gpu)]
            + " & time ~/.conda/envs/torch-numba/bin/python3 fvm_comparison.py && ",
            "jobs/fvm_comp.job",
        )

    elif i == 32:
        add_line(
            "export CUDA_VISIBLE_DEVICES="
            + v_gpu[i % len(v_gpu)]
            + " & time ~/.conda/envs/torch-numba/bin/python3 fvm_comparison.py;",
            "jobs/fvm_comp.job",
            True,
        )

    else:
        add_line(
            "export CUDA_VISIBLE_DEVICES="
            + v_gpu[i % len(v_gpu)]
            + " & time ~/.conda/envs/torch-numba/bin/python3 fvm_comparison.py;",
            "jobs/fvm_comp.job",
            False,
        )
