"""!@file evaluate.py
@brief Evaluation functions for the Denoising Diffusion Probabilistic Model (DDPM) on MNIST dataset.

@details This file contains evaluation functions including
- compute_fid: compute the Frechet Inception Distance (FID) for the generated samples
- compute_nll: compute the negative log-likelihood (NLL) for the generated samples

@author Yichi Zhang (yz870) on 19/03/2024
"""

import torch
import tqdm
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

def compute_fid(generated_samples, true_samples, device):
    """!@brief Compute the Frechet Inception Distance (FID) for the generated samples.

    @param generated_samples The generated samples from the trained model.
    @param true_samples The true samples from the MNIST dataset.
    @param device The device to run the computation on.

    @return fid The Frechet Inception Distance (FID) value.
    """
    # Preprocess the data
    # fid requires 3-channel images, all reshaped to 299x299
    # we can use torchvision.transforms to do this
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)), # expand to 3 channels
    ])
    generated_samples = torch.cat([transform(x) for x in generated_samples], 0)
    true_samples = torch.cat([transform(x) for x in true_samples], 0)
    # sample pixel values to [0,1]
    generated_samples = (generated_samples - generated_samples.min())/ (generated_samples.max() - generated_samples.min())
    true_samples = (true_samples - true_samples.min())/ (true_samples.max() - true_samples.min())

    generated_samples = generated_samples.view(-1, 3, 299, 299).to(device)
    true_samples = true_samples.view(-1, 3, 299, 299).to(device)

    # Compute the FID
    fid = FrechetInceptionDistance(feature=64, normalize=True)
    fid.update(true_samples, real=True)
    fid.update(generated_samples, real=False)
    print('Computing FID...')
    fid_val = fid.compute()
    fid_val = fid_val.item()
    return fid_val

def evaluate_losses(model, test_data, device):
    """!@brief Evaluate the loss values for the trained model.

    @param model The trained model.
    @param test_data The test dataset.
    @param device The device to run the computation on.

    @return mse The mean squared error (MSE) value.
    @return ssim The structural similarity (SSIM) value.
    @return psnr The peak signal-to-noise ratio (PSNR) value.
    """
    # Set the model to evaluation mode
    model.to(device)
    model.eval()
    mse_sum, ssim_sum, psnr_sum = 0, 0, 0
    
    for sample in tqdm.tqdm(test_data, desc='Evaluating'):
        sample = sample.unsqueeze(1).to(device)
        mse, ssim, psnr = model(sample)
        mse_sum += mse
        ssim_sum += ssim
        psnr_sum += psnr

    mse = mse_sum / len(test_data)
    ssim = ssim_sum / len(test_data)
    psnr = psnr_sum / len(test_data)

    return mse, psnr, ssim       