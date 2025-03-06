#!/bin/bash
#SBATCH -c 16
#SBATCH -t 3-00:00
#SBATCH -p kempner
#SBATCH --mem=250g
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append 
#SBATCH -o %j.out 
#SBATCH -e %j.err
#SBATCH --mail-type=FAIL 
#SBATCH --account=kempner_ba_lab

# SETUP ENVIRONMENT
module load gcc/13.2.0-fasrc01
source ~/.bashrc
mamba activate slot_attention6

# RUN EXPERIMENT
python produce_scores.py
#python produce_scores_multi-mnist.py

# DEACTIVATE ENVIRONMENT
mamba deactivate