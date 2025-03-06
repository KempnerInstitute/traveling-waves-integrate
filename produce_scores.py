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


SCORE_FILE = "results/scores_test2.csv"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    loss_foreground = nn.CrossEntropyLoss(ignore_index=0)
    
    # Load data
    data_config1 = DATASET_CONFIG['new_tetronimoes']
    data_config2 = DATASET_CONFIG['mnist']
    _, _, testset1 = load_data('new_tetronimoes', data_config1)
    _, _, testset2 = load_data('mnist', data_config2)
    testsets = {'new_tetronimoes' : testset1,
                'mnist' : testset2}

    # Setup folders
    base_folder = 'experiments'
    ccns = ['ccn8', 'ccn11', 'ccn8_rerun'] # ccn9
    datasets = ['mnist', 'new_tetronimoes']
    models = ['cornn_model2', 'conv_recurrent2', 'baseline1_flexible']
    extensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # ccn8_rerun/mnist/cornn_model2/{7, 8, 9, 10} --->       4
    # ccn8/mnist/cornn_model2/{ext 1 - 6} ------------>     46
    # ccn8/new_tetronimoes/cornn_model2/{ext} ------>       50
    # ccn8/mnist/conv_recurrent2/{ext} --->                 50
    # ccn8/new_tetronimoes/conv_recurrent2/{ext} --->       50
    # ccn11/new_tetronimoes/baseline1_flexible/{ext} --->   50
    # ccn11/mnist/baseline1_flexible/{ext} --->             50

    # Load configs and run folders
    configs = []
    folders = []
    for ccn in ccns:
        for dataset in datasets:
            for model in models:
                # Skip
                if ccn == 'ccn8' and model == 'baseline1_flexible':
                    continue
                if ccn == 'ccn11' and model != 'baseline1_flexible':
                    continue
                if ccn == 'ccn8_rerun' and model != 'cornn_model2':
                    continue
                for ext in extensions:
                    folder = f"{base_folder}/{ccn}/{dataset}/{model}/{ext}"
                    search_pattern = os.path.join(folder, '**', '.hydra', 'config.yaml')
                    for file_path in glob.iglob(search_pattern, recursive=True):
                        # Skip
                        if ccn == 'ccn8' and model == 'conv_recurrent2' and '20iters' not in file_path:
                            continue
                        if ccn == 'ccn8' and model == 'cornn_model2' and 'max_time' in file_path and dataset == 'mnist' and ext == 7:
                            continue
                        # Check if it's a file
                        if os.path.isfile(file_path):
                            configs.append(load_yaml_file(file_path))
                            folders.append(os.path.dirname(os.path.dirname(file_path)) + "/")
    #pdb.set_trace()
    # Load model and evaluate
    csv_path = SCORE_FILE
    set_random_seed(500)
    for i, config in enumerate(configs):
        testset = testsets[config['dataset']]
        full_run_name = folders[i]
        run_name = folders[i][len(base_folder) + 1:]
        data_config = DATASET_CONFIG[config['dataset']]
        model = load_model(full_run_name, config, device, data_config)
        extension = full_run_name.split("/")[-3]
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
            "readout_type",
            "max_iters",
            "num_layers",
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
        "readout_type": config['readout_type'],
        "max_iters": config['max_iters'],
        "num_layers": config['num_layers'],
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