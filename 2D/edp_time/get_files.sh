#!/bin/bash

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/jobs/* ./jobs

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/error_files/* ./error_files

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/output_files/* ./output_files

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/edo_pinn_sim/* ./edo_pinn_sim

scp $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/learning_curves/* ./learning_curves


