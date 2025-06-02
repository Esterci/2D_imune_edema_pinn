#!/bin/bash

# Path to the shell profile (adjust if using zsh, fish, etc.)
PROFILE_FILE="$HOME/.bashrc"

# Function to add or update environment variables in the profile file
add_or_update_env_var() {
    VAR_NAME=$1
    VAR_VALUE=$2

    if grep -q "^export $VAR_NAME=" "$PROFILE_FILE"; then
        sed -i "s|^export $VAR_NAME=.*|export $VAR_NAME=\"$VAR_VALUE\"|" "$PROFILE_FILE"
    else
        echo "export $VAR_NAME=\"$VAR_VALUE\"" >> "$PROFILE_FILE"
    fi
}

# Prompt the user for input
read -rp "Enter the cluster username (CLUSTER_USER): " CLUSTER_USER
read -rp "Enter the cluster DNS address (CLUSTER_DNS): " CLUSTER_DNS
read -rp "Enter the name of the bucket for PINN data (PINN_BUCKET): " PINN_BUCKET
read -rp "Enter the local path to the PINN Git repository (LOCAL_PINN_GIT): " LOCAL_PINN_GIT

# Export to current session
export CLUSTER_USER="$CLUSTER_USER"
export CLUSTER_DNS="$CLUSTER_DNS"
export PINN_BUCKET="$PINN_BUCKET"
export LOCAL_PINN_GIT="$LOCAL_PINN_GIT"

# Save permanently in the shell profile
add_or_update_env_var "CLUSTER_USER" "$CLUSTER_USER"
add_or_update_env_var "CLUSTER_DNS" "$CLUSTER_DNS"
add_or_update_env_var "PINN_BUCKET" "$PINN_BUCKET"
add_or_update_env_var "LOCAL_PINN_GIT" "$LOCAL_PINN_GIT"

# Final output
echo -e "\n✅ Environment variables have been successfully added/updated in $PROFILE_FILE"
echo "🔁 Run 'source $PROFILE_FILE' to apply the changes to the current terminal session."
