#!/bin/bash

cd scripts-ccn-1/readouts4_unet2_multi_mnist_300e

sbatch train_2.sh
sbatch train_3.sh
sbatch train_4.sh
sbatch train_5.sh