# Geting files to bucket 

mgc object-storage objects download-all --src=$PINN_BUCKET --dst=$LOCAL_PINN_GIT 

d=1721243952

#################

unzip -o $LOCAL_PINN_GIT/jobs_$d.zip

#################

unzip -o $LOCAL_PINN_GIT/error_files_$d.zip

#################

unzip -o $LOCAL_PINN_GIT/output_files_$d.zip

#################

unzip -o $LOCAL_PINN_GIT/edo_pinn_sim_$d.zip

#################

unzip -o $LOCAL_PINN_GIT/edo_fdm_sim_$d.zip

#################

unzip -o $LOCAL_PINN_GIT/learning_curves_$d.zip