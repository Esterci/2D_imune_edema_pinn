#!/bin/bash

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_time/jobs/* ./jobs

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_time/error_files/* ./error_files

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_time/output_files/* ./output_files

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_time/pinn_sim/* ./pinn_sim

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_time/learning_curves/* ./learning_curves


