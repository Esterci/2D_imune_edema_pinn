#!/bin/bash

scp ./jobs/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_time/jobs
scp ./fdm_sim/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/2D_imune_edema_pinn/2D/edp_time/fdm_sim