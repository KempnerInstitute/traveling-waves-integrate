import torch
import wandb
import hydra
import omegaconf


from utils import make_model, set_random_seed, save_model, load_model
from trainer import train
from dataset import ShapeDataset, load_data
from dataset_config import DATASET_CONFIG


@hydra.main(config_path=".", config_name="config")
def main(config):
    # Setup wandb
    wandb_config = omegaconf.OmegaConf.to_container(
        config.params, resolve=True, throw_on_missing=True
    )
    wandb.init(entity=config.wandb.entity, project=config.wandb.project,
               config=wandb_config, name=config.params.run_name)

    # Setup
    config = config.params
    data_config = DATASET_CONFIG[config.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_random_seed(config.seed)

    # Load data
    trainset, valset, testset = load_data(config.dataset, data_config)
    
    set_random_seed(config.seed)

    # Make model
    net = make_model(device, config.model_type,
                     config.num_classes, config.N, config.dt, config.min_iters,
                     config.max_iters, data_config['channels'], config.c_mid, config.hidden_channels, config.rnn_kernel,
                     data_config['img_size'], 
                     config.kernel_init, config.cell_type, config.num_layers, config.readout_type)
    
    # Train
    net = train(net, config.dataset, trainset, valset, testset, device, min_epochs=config.min_epochs, max_epochs=config.max_epochs, lr=config.lr,
                batch_size=config.batch_size, model_type=config.model_type,
                num_channels_plot=config.num_channels_plot, 
                optimizer=config.optimizer, weight_decay=config.weight_decay,
                num_classes=config.num_classes, cp_path=config.cp_path, test_seed=config.seed + 1,
                patience=config.training_patience, tolerance=config.training_tolerance)
    
    # Save FFT and other images

if __name__ == "__main__":
    main()