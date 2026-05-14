#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 [cleaning|fvm|pinn|pinn-training|pinn-inference|all]"
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
    nn)
        echo "Cleaning all NN files..."
        find learning_curves/ -type f -name "*.pkl" -delete
        find nn_parameters/ -type f -name "*.pt" -delete
        find nn_sim/ -type f -name "*.pkl" -delete
        find experiments/ -type f -name "*.pkl" -delete
        ;;
    nn-training)
        echo "Cleaning NN training files..."
        find learning_curves/ -type f -name "*.pkl" -delete
        find nn_parameters/ -type f -name "*.pt" -delete
        ;;
    nn-inference)
        echo "Cleaning NN inference files..."
        find nn_sim/ -type f -name "*.pkl" -delete
        ;;
    nn-experiment)
        echo "Cleaning NN experiments files..."
        find experiments/ -type f -name "*.pkl" -delete
        ;;
    all)
        echo "Cleaning all files..."
        find error_files/ -type f -name "*.e" -delete
        find jobs/ -type f -name "*.job" -delete
        find output_files/ -type f -name "*.o" -delete
        find fvm_sim/ -type f -name "*.pkl" -delete
        find fvm_animations/ -type f -name "*.mp4" -delete
        find learning_curves/ -type f -name "*.pkl" -delete
        find nn_parameters/ -type f -name "*.pt" -delete
        find nn_sim/ -type f -name "*.pkl" -delete
        ;;
    *)
        usage
        ;;
esac

echo "Cleanup completed."