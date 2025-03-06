import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import os
import glob
import pdb

from utils import make_model, set_random_seed
from trainer import eval_metrics
from dataset import load_data
from dataset_config import DATASET_CONFIG

import pdb


SCORE_FILE = "results/scores_test_multi-mnist.csv"

CORNN_FOLDERS = [
    "ccn17/multi_mnist/cornn_model2/1/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/2/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/3/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/5/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/8/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/9/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/11/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/12/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/13/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/14/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/15/linear_smaller4_0-100iters_16hc/",
    "ccn17/multi_mnist/cornn_model2/16/linear_smaller4_0-100iters_16hc/"
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    loss_foreground = nn.CrossEntropyLoss(ignore_index=0)
    
    # Load data
    data_config1 = DATASET_CONFIG['multi_mnist']
    _, _, testset1 = load_data('multi_mnist', data_config1)
    testsets = {'multi_mnist' : testset1}

    # Setup folders to load nets
    base_folder = 'experiments'
    unet_folder = 'ccn14/multi_mnist/unet2'
    extensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # load cornn configs and folders
    configs = []
    folders = []
    for folder in CORNN_FOLDERS:
        folder = f"{base_folder}/{folder}"
        search_pattern = os.path.join(folder, '**', '.hydra', 'config.yaml')
        for file_path in glob.iglob(search_pattern, recursive=True):
            if os.path.isfile(file_path):
                configs.append(load_yaml_file(file_path))
                folders.append(os.path.dirname(os.path.dirname(file_path)) + "/")

    # Load unet configs and folders
    for ext in extensions:
        folder = f"{base_folder}/{unet_folder}/{ext}"
        search_pattern = os.path.join(folder, '**', '.hydra', 'config.yaml')
        for file_path in glob.iglob(search_pattern, recursive=True):
            if os.path.isfile(file_path):
                configs.append(load_yaml_file(file_path))
                folders.append(os.path.dirname(os.path.dirname(file_path)) + "/")

    # Load runs for the last 2 sets of seeds
    unet_folder = 'ccn14_v2/multi_mnist/unet2'
    extensions = [11, 12]
    for ext in extensions:
        folder = f"{base_folder}/{unet_folder}/{ext}"
        search_pattern = os.path.join(folder, '**', '.hydra', 'config.yaml')
        for file_path in glob.iglob(search_pattern, recursive=True):
            if os.path.isfile(file_path):
                configs.append(load_yaml_file(file_path))
                folders.append(os.path.dirname(os.path.dirname(file_path)) + "/")

    # Load model and evaluate
    csv_path = SCORE_FILE
    os.system(f"rm {csv_path}")
    set_random_seed(600)
    for i, config in enumerate(configs):
        testset = testsets[config['dataset']]
        full_run_name = folders[i]
        run_name = folders[i][len(base_folder) + 1:]
        data_config = DATASET_CONFIG[config['dataset']]
        model = load_model(full_run_name, config, device, data_config)
        extension = full_run_name.split("/")[-3]
        #set_random_seed(config['seed'])
        evaluate_model_and_update_csv(model, run_name, extension, config, device,
                                      loss_func, loss_foreground, testset,
                                      epoch=None, batch_size=64, csv_file=csv_path)


def evaluate_model_and_update_csv(model, run_name, ext, config, device,
                                  loss_func, loss_foreground, testset,
                                  epoch, batch_size, csv_file):
    """
    Evaluates an already-loaded model and updates (or creates) a CSV with the results.
    
    Parameters:
        model: The already loaded model.
        run_name (str): A name/identifier for the model run.
        config: Configuration object containing run metadata.
        device: Device on which to run evaluation.
        loss_func: Loss function for evaluation.
        loss_foreground: Additional loss function for foreground.
        testset: Dataset to run evaluation on.
        epoch (int): Epoch number (used for evaluation if needed).
        batch_size (int): Batch size for evaluation.
        csv_file (str): Path to CSV file to update or create.
    """
    # If CSV exists, load it. Otherwise, create a new DataFrame with the required columns.
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        columns = [
            "run_name",
            "dataset",
            "model_type",
            "c_mid",
            "extension",
            "loss",
            "iou",
            "acc",
            "loss_foreground",
            "iou_foreground",
            "acc_foreground"
        ]
        df = pd.DataFrame(columns=columns)

    # Run the evaluation.
    metrics = eval_metrics(model, loss_func, loss_foreground, testset, device, epoch, batch_size)
    
    # Create a new row with the evaluation metrics and configuration details.
    new_row = {
        "run_name": run_name,
        "dataset": config['dataset'],
        "model_type": config['model_type'],
        "c_mid": config['c_mid'],
        "extension": ext,
        "loss": metrics['total_loss'],
        "iou": metrics['total_iou'],
        "acc": metrics['total_acc'],
        "loss_foreground": metrics['foreground_loss'],
        "iou_foreground": metrics['foreground_iou'],
        "acc_foreground": metrics['foreground_acc']
    }
    
    # Append the new row to the DataFrame.
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the updated DataFrame to CSV.
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")

# Function to load a YAML file
def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)['params']
    
# Load saved models
def load_model(cp_folder, config, device, data_config):
    net = make_model(
        device,
        config['model_type'],
        config['num_classes'],
        config['N'],
        config['dt'],
        config['min_iters'],
        config['max_iters'],
        data_config['channels'],
        config['c_mid'],
        config['hidden_channels'],
        config['rnn_kernel'],
        data_config['img_size'],
        config['kernel_init'],
        cell_type=config['cell_type'],
        num_layers=config['num_layers'],
        readout_type=config['readout_type'],
    )
    net.load_state_dict(torch.load(cp_folder + "cp.pt", map_location=torch.device('cpu')), strict=False)
    net.eval()
    return net.to(device)
    

if __name__ == "__main__":
    main()