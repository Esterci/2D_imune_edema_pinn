#!/bin/bash

scp ./control_dicts/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/control_dicts
scp ./jobs/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/jobs
scp ./source_points/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_t_x_y/source_points