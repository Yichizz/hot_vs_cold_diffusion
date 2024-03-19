"""!@mainpage Train DDPM
@brief Train Denoising Diffusion Probabilistic Model (DDPM) on MNIST dataset.

@details This script contains code to reproduce the results and figure of part 1 in the coursework.
For hyperparameter tuning, please change the parameters.ini file in the folder.
For training the model, please run the following command:
```bash
python train_ddpm.py parameters.ini
```
@author Yichi Zhang (yz870) on 19/03/2024
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from accelerate import Accelerator
from ddpm_mnist.utils import read_parameters, create_folder, save_losses, visualize_losses
from ddpm_mnist.model_builder import CNN, DDPM
from ddpm_mnist.engine import train


def main():
    # Read parameters from the .ini file
    if len(sys.argv) != 2:
        print("Please specify the .ini file.")
        print("Usage: python train_ddpm.py parameters.ini")
        sys.exit(1)
    param_file = sys.argv[1]
    params = read_parameters(param_file)
    # create folder for saving the generated samples, losses, and model
    create_folder(data_dir='./data', model_dir='./model', sample_dir='./samples', loss_dir='./losses')

    # Preprocess the data
    normalise_flag = params['hyperparameters_preprocess']['normalise']
    augmentation_flag = params['hyperparameters_preprocess']['augmentation']
    if normalise_flag and augmentation_flag:
        tf = transforms.Compose([transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    elif normalise_flag:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    elif augmentation_flag:
        tf = transforms.Compose([transforms.TrivialAugmentWide(), transforms.ToTensor()])
    else:
        tf = transforms.Compose([transforms.ToTensor()])

    trainset = MNIST("./data/train", train=True, download=True, transform=tf)
    testset = MNIST("./data/test", train=False, download=True, transform=tf)

    # Create the dataloader
    batch_size = params['hyperparameters_training']['batch_size']
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Build the model
    hidden_channels = params['hyperparameters_model']['hidden_channels']
    betas = params['hyperparameters_model']['betas']
    n_T = params['hyperparameters_model']['n_T']
    activation = params['hyperparameters_model']['activation']

    if activation == 'GELU':
        act = nn.GELU
    elif activation == 'ReLU':
        act = nn.ReLU
    elif activation == 'LeakyReLU':
        act = nn.LeakyReLU
    else:
        raise ValueError("Activation function not supported.")
    
    gt = CNN(in_channels=1, expected_shape= (28,28), n_hidden=hidden_channels, act=act)
    model = DDPM(gt, betas=betas, n_T=n_T)

    # Train the model
    lr = params['hyperparameters_training']['lr']
    weight_decay = params['hyperparameters_training']['weight_decay']
    # n_epochs = params['hyperparameters_training']['n_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    accelerator = Accelerator()
    model, optimizer, trainloader= accelerator.prepare(model, optimizer, trainloader)

    # Train the model
    n_epochs = 3 # for testing
    print("Start training...")
    losses, psnr_vals, ssim_vals = train(model, trainloader, optimizer, n_epochs, accelerator.device, './samples', './model', 'ddpm_mnist')
    print('Training finished.')

    # write losses to file
    save_losses(n_epochs, losses, psnr_vals, ssim_vals, './losses', 'ddpm_mnist')

    # visualize the average loss, PSNR, and SSIM per epoch
    visualize_losses(n_epochs, losses, psnr_vals, ssim_vals, './losses', 'ddpm_mnist')

if __name__ == "__main__":
    main()