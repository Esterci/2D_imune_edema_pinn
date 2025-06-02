#!/bin/bash

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/control_dicts/* ./control_dicts
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/error_files/* ./error_files
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/fvm_animations/* ./fvm_animations
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/fvm_sim/* ./fvm_sim
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/jobs/* ./jobs
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/learning_curves/* ./learning_curves
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/nn_parameters/* ./nn_parameters
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/output_files/* ./output_files
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/pinn_sim/* ./pinn_sim
