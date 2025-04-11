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
        find error_files/ -type f -name "*.e" -delete
        find jobs/ -type f -name "*.job" -delete
        find output_files/ -type f -name "*.o" -delete
        ;;
    fvm)
        echo "Cleaning FVM simulation files..."
        find fvm_sim/ -type f -name "*.pkl" -delete
        find fvm_animations/ -type f -name "*.mp4" -delete
        ;;
    pinn-training)
        echo "Cleaning PINN training files..."
        find learning_curves/ -type f -name "*.pkl" -delete
        find nn_parameters/ -type f -name "*.pt" -delete
        ;;
    pinn-inference)
        echo "Cleaning PINN inference files..."
        find pinn_sim/ -type f -name "*.pkl" -delete
        ;;
    all)
        echo "Cleaning all files..."
        find error_files/ -type f -name "*.e" -delete
        find fvm_sim/ -type f -name "*.pkl" -delete
        find jobs/ -type f -name "*.job" -delete
        find learning_curves/ -type f -name "*.pkl" -delete
        find nn_parameters/ -type f -name "*.pt" -delete
        find output_files/ -type f -name "*.o" -delete
        find pinn_sim/ -type f -name "*.pkl" -delete
        ;;
    *)
        usage
        ;;
esac

echo "Cleanup completed."