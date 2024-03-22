"""!@file engine.py
@brief Training engine for the Denoising Diffusion Probabilistic Model (DDPM) on MNIST dataset.

@details This file contains the training function for the DDPM model. 
The trainign process records the loss, PSNR, and SSIM values for each epoch and saves the model and samples to the specified directory.
@author Yichi Zhang (yz870) on 19/03/2024
"""


import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

# Define the training function for the DDPM model
def train_ddpm(model, dataloader, optimizer, epochs, device, sampling_dir, model_dir, name, 
          early_stopping=False, patience=5, min_delta=1e-4):
    """!@brief Train the DDPM model on the MNIST dataset.

    @param model The DDPM model to be trained.
    @param dataloader The dataloader for the MNIST dataset.
    @param optimizer The optimizer for the model.
    @param epochs The number of epochs for training.
    @param device The device to run the training on.
    @param sampling_dir The directory to save the generated samples.
    @param model_dir The directory to save the trained model.
    @param name The name of the model to specify the file name.
    @param early_stopping Flag to enable early stopping.
    @param patience The number of epochs to wait before early stopping.
    @param min_delta The minimum change in loss to be considered as an improvement.

    @return losses The list of mean squared error (MSE) loss values for each epoch.
    @return psnr_vals The list of Peak Signal-to-Noise Ratio (PSNR) values for each epoch.
    @return ssim_vals The list of structural similarity (SSIM) values for each epoch.
    """
    losses, psnr_vals, ssim_vals = [], [], []
    loss_epoch, psnr_epoch, ssim_epoch = [], [], []
    n_epoch = epochs
    
    for i in range(n_epoch):
        model.train()
        pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar

        for x, _ in pbar:
            optimizer.zero_grad()

            loss, ssim_val, psnr_val = model(x)

            loss.backward()
            # ^Technically should be `accelerator.backward(loss)` but not necessary for local training

            losses.append(loss.item())
            psnr_vals.append(psnr_val.item())
            ssim_vals.append(ssim_val.item())
            avg_loss = np.average(losses[min(len(losses)-100, 0):])
            avg_psnr = np.average(psnr_vals[min(len(psnr_vals)-100, 0):])
            avg_ssim = np.average(ssim_vals[min(len(ssim_vals)-100, 0):])
            pbar.set_description(f"epoch {i+1}/{n_epoch}, loss: {avg_loss:.3f}, psnr: {avg_psnr:.3f}, ssim: {avg_ssim:.3f}")
            optimizer.step()
        # compute the average loss, PSNR, and SSIM for the current epoch
        # number of batches per epoch = len(dataloader)
        # losses per epoch = average of losses per batch
        loss_epoch.append(np.mean(losses[-len(dataloader):]))
        psnr_epoch.append(np.mean(psnr_vals[-len(dataloader):]))
        ssim_epoch.append(np.mean(ssim_vals[-len(dataloader):]))
        # early stopping
        if early_stopping and len(loss_epoch) > patience + 1:
            if loss_epoch[-1-patience] - loss_epoch[-1] <= min_delta:
                print(f"Early stopping at epoch {i+1}.")
                n_epoch = i+1
                break

        model.eval()
        with torch.no_grad():
            xh = model.sample(16, (1, 28, 28), device)  # Can get device explicitly with `accelerator.device`
            grid = make_grid(xh, nrow=4)

            # Save samples to `./contents` directory
            save_image(grid, sampling_dir + f"/{name}_epoch_{i+1}.png")

            # save model
            torch.save(model.state_dict(), model_dir + f"/{name}.pth")

    return loss_epoch, psnr_epoch, ssim_epoch, n_epoch

# Define the evaluation function for the DDM model
# main difference here is in sampling from the model
def train_ddm(model, dataloader, optimizer, epochs, device, sampling_dir, model_dir, name, 
          early_stopping=False, patience=5, min_delta=0.001):
    """!@brief Train the DDPM model on the MNIST dataset.

    @param model The DDPM model to be trained.
    @param dataloader The dataloader for the MNIST dataset.
    @param optimizer The optimizer for the model.
    @param epochs The number of epochs for training.
    @param device The device to run the training on.
    @param sampling_dir The directory to save the generated samples.
    @param model_dir The directory to save the trained model.
    @param name The name of the model to specify the file name.
    @param early_stopping Flag to enable early stopping.
    @param patience The number of epochs to wait before early stopping.
    @param min_delta The minimum change in loss to be considered as an improvement.

    @return losses The list of mean squared error (MSE) loss values for each epoch.
    @return psnr_vals The list of Peak Signal-to-Noise Ratio (PSNR) values for each epoch.
    @return ssim_vals The list of structural similarity (SSIM) values for each epoch.
    """
    losses, psnr_vals, ssim_vals = [], [], []
    loss_epoch, psnr_epoch, ssim_epoch = [], [], []
    n_epoch = epochs
    
    for i in range(n_epoch):
        model.train()
        pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar

        for x, _ in pbar:
            optimizer.zero_grad()

            loss, ssim_val, psnr_val = model(x)

            loss.backward()
            # ^Technically should be `accelerator.backward(loss)` but not necessary for local training

            losses.append(loss.item())
            psnr_vals.append(psnr_val.item())
            ssim_vals.append(ssim_val.item())
            avg_loss = np.average(losses[min(len(losses)-100, 0):])
            avg_psnr = np.average(psnr_vals[min(len(psnr_vals)-100, 0):])
            avg_ssim = np.average(ssim_vals[min(len(ssim_vals)-100, 0):])
            pbar.set_description(f"epoch {i+1}/{n_epoch}, loss: {avg_loss:.3f}, psnr: {avg_psnr:.3f}, ssim: {avg_ssim:.3f}")
            optimizer.step()
        # compute the average loss, PSNR, and SSIM for the current epoch
        # number of batches per epoch = len(dataloader)
        # losses per epoch = average of losses per batch
        loss_epoch.append(np.mean(losses[-len(dataloader):]))
        psnr_epoch.append(np.mean(psnr_vals[-len(dataloader):]))
        ssim_epoch.append(np.mean(ssim_vals[-len(dataloader):]))
        # early stopping
        if early_stopping and len(loss_epoch) > patience + 1:
            if loss_epoch[-1-patience] - loss_epoch[-1] < min_delta:
                print(f"Early stopping at epoch {i+1}.")
                n_epoch = i+1
                break

        model.eval()
        with torch.no_grad():
            # we use the sample method to generate samples from the model
            # original samples are the input images in the last batch (x)
            xh = model.sample(16, (1, 28, 28), original_sample = x, device=device)
            grid = make_grid(xh, nrow=4)

            # Save samples to `./contents` directory
            save_image(grid, sampling_dir + f"/{name}_epoch_{i+1}.png")

            # save model
            torch.save(model.state_dict(), model_dir + f"/{name}.pth")

    return loss_epoch, psnr_epoch, ssim_epoch, n_epoch