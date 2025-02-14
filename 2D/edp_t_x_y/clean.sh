#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 [cleaning|fvm|pinn-training|pinn-inference|all]"
    exit 1
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    usage
fi

# Process the provided argument
case "$1" in
    cleaning)
        echo "Cleaning job, error, and output files..."
        rm -f error_files/*.e
        rm -f jobs/*.job
        rm -f output_files/*.o
        ;;
    fvm)
        echo "Cleaning FVM simulation files..."
        rm -f fvm_sim/*.pkl
        ;;
    pinn-training)
        echo "Cleaning PINN training files..."
        rm -f learning_curves/*.pkl
        rm -f nn_parameters/*.pt
        ;;
    pinn-inference)
        echo "Cleaning PINN inference files..."
        rm -f pinn_sim/*.pkl
        ;;
    all)
        echo "Cleaning all files..."
        rm -f error_files/*.e
        rm -f fvm_sim/*.pkl
        rm -f jobs/*.job
        rm -f learning_curves/*.pkl
        rm -f nn_parameters/*.pt
        rm -f output_files/*.o
        rm -f pinn_sim/*.pkl
        ;;
    *)
        usage
        ;;
esac

echo "Cleanup completed."