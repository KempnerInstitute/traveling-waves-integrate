#!/bin/bash

# SETUP ENVIRONMENT
module load gcc/13.2.0-fasrc01
source ~/.bashrc
mamba activate traveling_waves_integrate

# RUN EXPERIMENT
python main.py

# DEACTIVATE ENVIRONMENT
mamba deactivate