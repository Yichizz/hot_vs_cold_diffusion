"""!@mainpage Train DDPM
@brief Train and Evaluate Denoising Diffusion Probabilistic Model (DDPM) on MNIST dataset.

@details This script contains code to reproduce the results and figure of part 1 in the coursework.
For hyperparameter tuning, please change the parameters.ini file in the folder.
For training the model, please run the following command:
```bash
python ddpm.py --ini_file parameters.ini --mode train
```
@author Yichi Zhang (yz870) on 19/03/2024
"""

import os
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from accelerate import Accelerator
from ddpm_mnist.utils import read_parameters, create_folder, save_losses, visualize_losses
from ddpm_mnist.model_builder import CNN, DDPM
from ddpm_mnist.engine import train
from ddpm_mnist.evaluate import compute_fid, evaluate_losses


def main_train():
    # Read parameters from the .ini file
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
    name = params['name']['model_name']

    # Train the model
    lr = params['hyperparameters_training']['lr']
    weight_decay = params['hyperparameters_training']['weight_decay']
    n_epochs = params['hyperparameters_training']['n_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_st = params['hyperparameters_training']['early_stopping']

    accelerator = Accelerator()
    model, optimizer, trainloader= accelerator.prepare(model, optimizer, trainloader)

    # Train the model
    print("Start training...")
    losses, psnr_vals, ssim_vals = train(model, trainloader, optimizer, n_epochs, accelerator.device, 
                                         './samples', './model', name, early_stopping=early_st)
    print('Training finished.')

    # write losses to file
    save_losses(n_epochs, losses, psnr_vals, ssim_vals, './losses', name=name)

    # visualize the average loss, PSNR, and SSIM per epoch
    visualize_losses(n_epochs, losses, psnr_vals, ssim_vals, './losses', name=name)


def main_evaluate():
    # intialized the model
    params = read_parameters(param_file)
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
    name = params['name']['model_name']

    # load state dict
    if not os.path.exists(f'./model/{name}.pth'):
        print(f'No model found with name {name}.')
        print('Please train the model first.')
        sys.exit(1)
    model.load_state_dict(torch.load(f'./model/{name}.pth'))
    print(f"Loaded model {name}.")

    # load test set to compare with generated samples
    n_samples = params['hyperparameters_evaluation']['n_samples']  
    normalise_flag = params['hyperparameters_preprocess']['normalise']
    if normalise_flag:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    else:
        tf = transforms.Compose([transforms.ToTensor()])
    
    testset = MNIST("./data/test", train=False, download=True, transform=tf)
    testloader = DataLoader(testset, batch_size=n_samples, shuffle=True, num_workers=4, drop_last=True)
    
    accelerator = Accelerator()
    model, testloader = accelerator.prepare(model, testloader)

    # generate samples
    model.eval()
    with torch.no_grad():
        xh = model.sample(n_samples, (1, 28, 28), accelerator.device)
        generated = xh.cpu().numpy()
        save first 50 samples
        for i in range(50):
            save_image(xh[i], f'./samples/{name}_final_sample_{i}.png')
    print(f"Generated {n_samples} samples and saved the first 50 samples.")

    # compute FID: we use first batch of testloader
    # we compute FID on CPU in case out of memory
    fid = compute_fid(xh.to('cpu'), next(iter(testloader))[0], device = 'cpu')
    print(f"Frechet Inception Distance (FID) is {fid:.3f}.")

    # compute mse, psnr, and ssim on test set using trained model
    test_mse, test_psnr, test_ssim = evaluate_losses(model, next(iter(testloader))[0], accelerator.device)
    print(f"Mean Squared Error (MSE) on {len(testset)} test samples is {test_mse:.3f}.")
    print(f"Peak Signal-to-Noise Ratio (PSNR) on {len(testset)} test samples is {test_psnr:.3f}.")
    print(f"Structural Similarity (SSIM) on {len(testset)} test samples is {test_ssim:.3f}.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ddpm.py --ini_file <parameters.ini> --mode <train/evaluate>")
        sys.exit(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ini_file", help="The .ini file containing the hyperparameters.")
    parser.add_argument("--mode", help="The mode of the script (train/evaluate).")
    args = parser.parse_args()
    param_file = args.ini_file
    mode = args.mode

    if mode == 'train':
        main_train()
    elif mode == 'evaluate':
        main_evaluate()
    else:
        raise ValueError("Mode not supported.")