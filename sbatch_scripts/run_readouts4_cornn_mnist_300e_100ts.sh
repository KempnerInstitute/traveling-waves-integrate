#!/bin/bash

cd scripts-ccn-1/readouts4_mnist_300e_100ts

sbatch train_cornn_fft.sh
sbatch train_cornn_linear.sh
sbatch train_cornn_last.sh
sbatch train_cornn_mean_time.sh
sbatch train_cornn_max_time.sh