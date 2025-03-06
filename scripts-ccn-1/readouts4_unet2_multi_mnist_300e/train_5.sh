#!/bin/bash

# SETUP ENVIRONMENT
module load gcc/13.2.0-fasrc01
source ~/.bashrc
mamba activate traveling_waves_integrate

# Load useful variables
source base_vars.sh

# Go back to the main folder
cd ..
cd ..

# Useful variables
c_mid=5

seeds=(30 31 32 33 34 35 36 37 38 39)
new_seeds=()
# Loop through the original list and add the number
for curr_seed in "${seeds[@]}"; do
  new_seeds+=($((curr_seed + seed)))
done
seeds=("${new_seeds[@]}")
extensions=(1 2 3 4 5 6 7 8 9 10)
length=${#seeds[@]}

for ((i = 0; i < length; i++)); do
    seed=${seeds[i]}
    extension=${extensions[i]}

    run_name_no_attachment="${dataset}/${model_type}/${extension}/${c_mid}channels"
    run_name="${attachment}/${run_name_no_attachment}"
    run_dir="${base_folder}/${run_name_no_attachment}"

    # RUN EXPERIMENTS
    python main.py wandb.project="${project_name}" hydra.run.dir="${run_dir}" \
params.run_name="${run_name}" \
params.model_type="${model_type}" params.dataset="${dataset}" params.N=${N} \
params.batch_size=${batch_size} params.seed=${seed} params.max_iters=${max_iters} \
params.num_classes=${num_classes} params.dt=${dt} \
params.hidden_channels=${hidden_channels} params.cell_type=${cell_type} params.readout_type=${readout_type} \
params.c_mid=${c_mid} params.min_epochs=${min_epochs} params.max_epochs=${max_epochs} \
params.lr=${lr} params.min_iters=${min_iters} params.max_iters=${max_iters} params.rnn_kernel=${rnn_kernel} \
params.kernel_init=${kernel_init} params.cp_path=${cp_path} params.save_model=${save_model} \
params.num_channels_plot=${num_channels_plot} params.normalize=${normalize} params.optimizer=${optimizer} \
params.weight_decay=${weight_decay} params.num_layers=${num_layers} \
params.training_patience=${training_patience} params.training_tolerance=${training_tolerance}
    
done

# DEACTIVATE ENVIRONMENT
mamba deactivate