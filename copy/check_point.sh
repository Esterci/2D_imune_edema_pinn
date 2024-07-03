d=$(date +%s)

#################

zip -r jobs_$d.zip jobs/ 

mv jobs_$d.zip check_points/

#################

zip -r error_files_$d.zip error_files/ 

mv error_files_$d.zip check_points/

#################

zip -r output_files_$d.zip output_files/ 

mv output_files_$d.zip check_points/

#################

zip -r edo_pinn_sim_$d.zip edo_pinn_sim/ 

mv edo_pinn_sim_$d.zip check_points/

#################

zip -r learning_curves_$d.zip learning_curves/ 

mv learning_curves_$d.zip check_points/