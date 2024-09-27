d=$(date +%s)

#################

zip -r jobs_$d.zip jobs/ 

mv jobs_$d.zip $LOCAL_PINN_GIT

#################

zip -r error_files_$d.zip error_files/ 

mv error_files_$d.zip $LOCAL_PINN_GIT

#################

zip -r output_files_$d.zip output_files/ 

mv output_files_$d.zip $LOCAL_PINN_GIT

#################

zip -r edo_pinn_sim_$d.zip pinn_sim/ 

mv edo_pinn_sim_$d.zip $LOCAL_PINN_GIT

#################

zip -r learning_curves_$d.zip learning_curves/ 

mv learning_curves_$d.zip $LOCAL_PINN_GIT

# Sending files to bucket 

mgc object-storage objects sync --local=$LOCAL_PINN_GIT --bucket=$PINN_BUCKET