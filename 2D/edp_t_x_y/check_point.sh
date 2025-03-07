d=$(date +%s)


#################

zip -r control_dicts_$d.zip control_dicts/ 

mv control_dicts_$d.zip $LOCAL_PINN_GIT

#################

zip -r error_files_$d.zip error_files/ 

mv error_files_$d.zip $LOCAL_PINN_GIT

#################

zip -r fdm_sim_$d.zip fdm_sim/ 

mv fdm_sim_$d.zip $LOCAL_PINN_GIT

#################

zip -r fvm_sim_$d.zip fvm_sim/ 

mv fvm_sim_$d.zip $LOCAL_PINN_GIT

#################

zip -r jobs_$d.zip jobs/ 

mv jobs_$d.zip $LOCAL_PINN_GIT


#################

zip -r learning_curves_$d.zip learning_curves/ 

mv learning_curves_$d.zip $LOCAL_PINN_GIT

#################

zip -r nn_parameters_$d.zip nn_parameters/ 

mv nn_parameters_$d.zip $LOCAL_PINN_GIT

#################

zip -r output_files_$d.zip output_files/ 

mv output_files_$d.zip $LOCAL_PINN_GIT

#################

zip -r pinn_sim_$d.zip pinn_sim/ 

mv pinn_sim_$d.zip $LOCAL_PINN_GIT

#################

zip -r scale_weights_$d.zip scale_weights/ 

mv scale_weights_$d.zip $LOCAL_PINN_GIT

#################

zip -r source_points_$d.zip source_points/ 

mv source_points_$d.zip $LOCAL_PINN_GIT

# Sending files to bucket 

mgc object-storage objects sync --local=$LOCAL_PINN_GIT --bucket=$PINN_BUCKET