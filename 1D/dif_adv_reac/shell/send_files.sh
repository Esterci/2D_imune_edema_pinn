#!/bin/bash
scp ./control_dicts/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/Repositories/2D_imune_edema_pinn/1D/dif_adv_reac/control_dicts/;
scp ./jobs/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/Repositories/2D_imune_edema_pinn/1D/dif_adv_reac/jobs/;
scp ./source_points/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/Repositories/2D_imune_edema_pinn/1D/dif_adv_reac/source_points/;
scp ./fvm_animations/* $CLUSTER_USER@$CLUSTER_DNS:/home/thiago.esterci/Repositories/2D_imune_edema_pinn/1D/dif_adv_reac/fvm_animations/;


