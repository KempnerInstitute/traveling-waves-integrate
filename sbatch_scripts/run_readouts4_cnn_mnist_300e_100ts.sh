#!/bin/bash

cd scripts-ccn-1/readouts4_cnn_mnist_300e_100ts

sbatch train_2.sh
sbatch train_4.sh
sbatch train_8.sh
sbatch train_16.sh
sbatch train_32.sh