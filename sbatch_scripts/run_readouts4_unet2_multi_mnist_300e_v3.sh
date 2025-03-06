#!/bin/bash

cd scripts-ccn-1/readouts4_unet2_multi_mnist_300e_v2

sbatch train_2.sh
sbatch train_3.sh
sbatch train_4.sh