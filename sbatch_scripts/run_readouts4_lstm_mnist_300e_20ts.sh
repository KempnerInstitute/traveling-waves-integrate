#!/bin/bash

cd scripts-ccn-1/readouts4_lstm_mnist_300e_20ts

sbatch train_fft.sh
sbatch train_linear.sh
sbatch train_last.sh
sbatch train_mean_time.sh
sbatch train_max_time.sh