#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 [fvm|pinn-training|pinn-inference|control|jobs|scale|source|all]"
    exit 1
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    usage
fi

# Define the base path for remote files
REMOTE_PATH="/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x"
REMOTE="$CLUSTER_USER@$CLUSTER_DNS:$REMOTE_PATH"

# Process the provided argument
case "$1" in
    fvm)
        echo "Downloading FVM simulation files..."
        scp "$REMOTE/fvm_animations/*" ./fvm_animations
        scp "$REMOTE/fvm_sim/*" ./fvm_sim
        ;;
    pinn-training)
        echo "Downloading PINN training files..."
        scp "$REMOTE/learning_curves/*" ./learning_curves
        scp "$REMOTE/nn_parameters/*" ./nn_parameters
        ;;
    pinn-inference)
        echo "Downloading PINN inference files..."
        scp "$REMOTE/pinn_sim/*" ./pinn_sim
        ;;
    pinn-comparison)
        echo "Downloading PINN comparison files..."
        scp "$REMOTE/pinn_sim/comp_*" ./pinn_sim
        ;;
    control)
        echo "Downloading control_dicts..."
        scp "$REMOTE/control_dicts/*" ./control_dicts
        scp "$REMOTE/scale_weights/*" ./scale_weights
        scp "$REMOTE/source_points/*" ./source_points
        ;;
    jobs)
        echo "Downloading job and output files..."
        scp "$REMOTE/jobs/*" ./jobs
        scp "$REMOTE/error_files/*" ./error_files
        scp "$REMOTE/output_files/*" ./output_files
        ;;
    all)
        echo "Downloading all project files..."
        scp "$REMOTE/control_dicts/*" ./control_dicts
        scp "$REMOTE/error_files/*" ./error_files
        scp "$REMOTE/fvm_animations/*" ./fvm_animations
        scp "$REMOTE/fvm_sim/*" ./fvm_sim
        scp "$REMOTE/jobs/*" ./jobs
        scp "$REMOTE/learning_curves/*" ./learning_curves
        scp "$REMOTE/nn_parameters/*" ./nn_parameters
        scp "$REMOTE/output_files/*" ./output_files
        scp "$REMOTE/pinn_sim/*" ./pinn_sim
        scp "$REMOTE/scale_weights/*" ./scale_weights
        scp "$REMOTE/source_points/*" ./source_points
        ;;
    *)
        usage
        ;;
esac

echo "Download completed."
