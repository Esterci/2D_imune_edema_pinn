import argparse
import os


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


def write_setup():
    if os.path.exists("jobs/pinn_testing.job"):
        os.remove("jobs/pinn_testing.job")

    else:
        print("The fvm job does not exist")

    add_line("#!/bin/bash", "jobs/pinn_testing.job")
    add_line(
        "#----------------------------------------------------------",
        "jobs/pinn_testing.job",
    )
    add_line("# Job name", "jobs/pinn_testing.job")
    add_line(
        "#PBS -N pinn_testing",
        "jobs/pinn_testing.job",
    )
    add_line(
        "#PBS -e error_files/pinn_testing.e",
        "jobs/pinn_testing.job",
    )
    add_line(
        "#PBS -o output_files/pinn_testing.o",
        "jobs/pinn_testing.job",
    )
    add_line(
        "# Run time (hh:mm:ss) - 10:00 hr",
        "jobs/pinn_testing.job",
    )
    add_line(
        "#PBS -l walltime=10:00:00",
        "jobs/pinn_testing.job",
    )
    add_line(
        "#----------------------------------------------------------",
        "jobs/pinn_testing.job",
    )
    add_line(
        "#PBS -l nodes=compute-1-1:ppn=128",
        "jobs/pinn_testing.job",
    )
    add_line(
        "# Change to submission directory",
        "jobs/pinn_testing.job",
    )
    add_line(
        "cd $PBS_O_WORKDIR",
        "jobs/pinn_testing.job",
    )
    add_line(
        "cat $PBS_NODEFILE",
        "jobs/pinn_testing.job",
    )
    add_line(
        "# Launch Thiago-based executable",
        "jobs/pinn_testing.job",
    )


if __name__ == "__main__":

    write_setup()

    add_line(
        "~/.conda/envs/torch-numba-11/bin/python3 pinn_comparison.py;",
        "jobs/pinn_testing.job",
    )
