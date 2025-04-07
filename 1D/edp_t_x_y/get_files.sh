#!/bin/bash

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/control_dicts/* ./control_dicts
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/error_files/* ./error_files
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/fvm_sim/* ./fvm_sim
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/jobs/* ./jobs
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/learning_curves/* ./learning_curves
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/nn_parameters/* ./nn_parameters
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/output_files/* ./output_files
scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/pinn_sim/* ./pinn_sim
