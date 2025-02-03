# Geting files to bucket 

mgc object-storage objects download-all --src=$PINN_BUCKET --dst=$LOCAL_PINN_GIT 

d=1727462177

#################

unzip -o $LOCAL_PINN_GIT/control_dicts_$d.zip


#################

unzip -o $LOCAL_PINN_GIT/error_files_$d.zip


#################

unzip -o $LOCAL_PINN_GIT/fdm_sim_$d.zip


#################

unzip -o $LOCAL_PINN_GIT/fvm_sim_$d.zip


#################

unzip -o $LOCAL_PINN_GIT/jobs_$d.zip



#################

unzip -o $LOCAL_PINN_GIT/learning_curves_$d.zip


#################

unzip -o $LOCAL_PINN_GIT/nn_parameters_$d.zip


#################

unzip -o $LOCAL_PINN_GIT/output_files_$d.zip


#################

unzip -o $LOCAL_PINN_GIT/pinn_sim_$d.zip


#################

unzip -o $LOCAL_PINN_GIT/source_points_$d.zip