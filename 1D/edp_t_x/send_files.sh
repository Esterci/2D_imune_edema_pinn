#!/bin/bash

scp ./control_dicts/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/control_dicts
scp ./jobs/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/jobs
scp ./source_points/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/1D/edp_t_x/source_points