"""!@file utils.py
@brief Utility functions for the training diffusion models.

@details This file contains utility functions including
- read_parameters: read parameters from a .ini file
- create_folder: create folder for saving the dataset generated samples, losses, and model
- save_losses: save the loss values to a .txt file
- visualize_losses: visualize the loss values for each epoch

@author Yichi Zhang (yz870) on 19/03/2024
"""

import os
import configparser as cfg
import matplotlib.pyplot as plt

def read_parameters(file_path: str) -> dict:
    """!@brief Read parameters from a .ini file.

    @param file_path: The path to the .ini file.
    @return params dictionary containing the parameters.
    """
    params = {}
    config = cfg.ConfigParser()
    config.read(file_path)
    for section in config.sections():
        params[section] = {}
    
    # section: hyperparameters_preprocess
    params['hyperparameters_preprocess']['normalise'] = config.getboolean('hyperparameters_preprocess', 'normalise', fallback=True)
    params['hyperparameters_preprocess']['augmentation'] = config.getboolean('hyperparameters_preprocess', 'augmentation', fallback=True)

    # section: hyperparameters_model
    params['hyperparameters_model']['hidden_channels'] = config.get('hyperparameters_model', 'hidden_channels', fallback='32,64,64,32')
    params['hyperparameters_model']['hidden_channels'] = list(map(int, params['hyperparameters_model']['hidden_channels'].split(',')))
    params['hyperparameters_model']['hidden_channels'] = tuple(params['hyperparameters_model']['hidden_channels'])
    params['hyperparameters_model']['betas'] = config.get('hyperparameters_model', 'betas', fallback='1e-4, 0.02')
    params['hyperparameters_model']['betas'] = list(map(float, params['hyperparameters_model']['betas'].split(',')))
    params['hyperparameters_model']['betas'] = tuple(params['hyperparameters_model']['betas'])
    params['hyperparameters_model']['n_T'] = config.getint('hyperparameters_model', 'n_T', fallback=1000)
    params['hyperparameters_model']['activation'] = config.get('hyperparameters_model', 'activation', fallback='GeLU')
    params['hyperparameters_model']['variance_schedular'] = config.getfloat('hyperparameters_model', 'variance_schedular', fallback=4)

    # section: hyperparameters_training
    params['hyperparameters_training']['batch_size'] = config.getint('hyperparameters_training', 'batch_size', fallback=128)
    params['hyperparameters_training']['n_epochs'] = config.getint('hyperparameters_training', 'n_epochs', fallback=100)
    params['hyperparameters_training']['lr'] = config.getfloat('hyperparameters_training', 'lr', fallback=2e-4)
    params['hyperparameters_training']['weight_decay'] = config.getfloat('hyperparameters_training', 'weight_decay', fallback=0.0)
    params['hyperparameters_training']['early_stopping'] = config.getboolean('hyperparameters_training', 'early_stopping', fallback=False)

    # section: hyperparameters_evaluation
    params['hyperparameters_evaluation']['n_samples'] = config.getint('hyperparameters_evaluation', 'n_samples', fallback=1000)

    # section: name
    params['name']['model_name'] = config.get('name', 'model_name', fallback='ddpm_mnist')

    return params

def create_folder(data_dir: str, model_dir: str, sample_dir: str, loss_dir: str) -> None:
    """!@brief Create folder for saving the generated samples, losses, and model.

    @param data_dir: The directory to save the data.
    @param model_dir: The directory to save the model.
    @param sample_dir: The directory to save the generated samples.
    @param loss_dir: The directory to save the loss.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        if not os.path.exists(os.path.join(data_dir, 'train')):
            os.makedirs(os.path.join(data_dir, 'train'))
        if not os.path.exists(os.path.join(data_dir, 'test')):
            os.makedirs(os.path.join(data_dir, 'test'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    print(f"Created folder for saving the dataset, generated samples, losses, and models.")
    return None

def save_losses(n_epoch: int, losses: list, psnr_vals: list, ssim_vals: list, loss_dir: str, name: str) -> None:
    """!@brief Save the loss values to a .txt file.

    @param n_epoch: The number of epochs for training.
    @param losses: The list of mean squared error (MSE) loss values for each epoch.
    @param psnr_vals: The list of Peak Signal-to-Noise Ratio (PSNR) values for each epoch.
    @param ssim_vals: The list of structural similarity (SSIM) values for each epoch.
    @param loss_dir: The directory to save the loss.
    @param name: The name of the model to specify the file name.
    """
    assert len(losses) == n_epoch, "The number of epochs does not match the length of the loss values."
    assert len(psnr_vals) == n_epoch, "The number of epochs does not match the length of the PSNR values."
    assert len(ssim_vals) == n_epoch, "The number of epochs does not match the length of the SSIM values."
    with open(os.path.join(loss_dir, f'{name}_losses.txt'), 'w') as f:
        for i in range(len(losses)):
            f.write(f"Epoch {i+1}, Loss: {losses[i]:.3f}, PSNR: {psnr_vals[i]:.3f}, SSIM: {ssim_vals[i]:.3f}\n")
    print(f"Saved the loss values to {loss_dir}/{name}_losses.txt.")
    return None

def visualize_losses(n_epoch: int, losses: list, psnr_vals: list, ssim_vals: list, loss_dir: str, name: str) -> None:
    """!@brief Visualize the loss values for each epoch.

    @param n_epoch: The number of epochs for training.
    @param losses: The list of mean squared error (MSE) loss values for each epoch.
    @param psnr_vals: The list of Peak Signal-to-Noise Ratio (PSNR) values for each epoch.
    @param ssim_vals: The list of structural similarity (SSIM) values for each epoch.
    @param loss_dir: The directory to save the loss.
    @param name: The name of the model to specify the file name.
    """
    # generated by the github copilot
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].plot(range(n_epoch), losses, label='Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(range(n_epoch), psnr_vals, label='PSNR')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('PSNR')
    ax[1].legend()
    ax[1].grid(True)
    ax[2].plot(range(n_epoch), ssim_vals, label='SSIM')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('SSIM')
    ax[2].legend()
    ax[2].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f'{name}_losses.png'))
    print(f"Visualized the loss values to {loss_dir}/{name}_losses.png.")
    return None
