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


for i in range(33):
    add_line(
        "#!/bin/bash", "jobs/fvm_comp_" + str(i) + ".job"
    )
    add_line(
        "#----------------------------------------------------------",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "# Job name", "jobs/fvm_comp_" + str(i) + ".job"
    )
    add_line(
        "#PBS -N fvm_comp_" + str(i),
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "#PBS -e error_files/fvm_comp_" + str(i) + ".e",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "#PBS -o output_files/fvm_comp_" + str(i) + ".o",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "# Run time (hh:mm:ss) - 10:00 hr",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "#PBS -l walltime=10:00:00",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "#----------------------------------------------------------",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "#PBS -l nodes=compute-1-1:ppn=128",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "# Change to submission directory",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "cd $PBS_O_WORKDIR",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "cat $PBS_NODEFILE",
        "jobs/fvm_comp_" + str(i) + ".job",
    )
    add_line(
        "# Launch Thiago-based executable",
        "jobs/fvm_comp_" + str(i) + ".job",
    )

    add_line(
        "time ~/.conda/envs/torch-numba/bin/python3 fvm_comparison.py ",
        "jobs/fvm_comp_" + str(i) + ".job",
    )

